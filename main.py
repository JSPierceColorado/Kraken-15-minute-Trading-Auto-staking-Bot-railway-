# =============================
# main.py (tightened: no double-buys on held bases, safer signal, buylock sync)
# =============================

import os
import math
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional

import pandas as pd
import ccxt
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    UniqueConstraint, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

# -------------------------
# Settings
# -------------------------

def _get_bool(env: str, default: bool) -> bool:
    v = os.getenv(env)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _get_int(env: str, default: int) -> int:
    v = os.getenv(env)
    try:
        return int(v)
    except Exception:
        return default

def _get_float(env: str, default: float) -> float:
    v = os.getenv(env)
    try:
        return float(v)
    except Exception:
        return default

def _get_csv(env: str, default: List[str]) -> List[str]:
    v = os.getenv(env)
    if not v:
        return default
    return [x.strip() for x in v.split(',') if x.strip()]

class Settings:
    # --- General ---
    DRY_RUN = _get_bool("DRY_RUN", False)
    ENABLE_STAKING = _get_bool("ENABLE_STAKING", False)
    ENABLE_MOONSHOT = _get_bool("ENABLE_MOONSHOT", False)
    MOONSHOT_SELL_FRACTION = _get_float("MOONSHOT_SELL_FRACTION", 0.7)
    TAKE_PROFIT_PCT = _get_float("TAKE_PROFIT_PCT", 0.10)
    MAX_BUYS_PER_RUN = _get_int("MAX_BUYS_PER_RUN", 3)

    # DB
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")

    # --- API keys ---
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    # --- Universe / filters ---
    QUOTE_ASSETS   = _get_csv("QUOTE_ASSETS", ["USD", "USDT"])
    INCLUDE_CRYPTO = _get_bool("INCLUDE_CRYPTO", True)
    INCLUDE_XSTOCKS = _get_bool("INCLUDE_XSTOCKS", True)
    XSTOCKS_ONLY   = _get_bool("XSTOCKS_ONLY", False)
    XSTOCKS_SUFFIX = os.getenv("XSTOCKS_SUFFIX", "x")        # e.g., "AAPLx"
    XSTOCKS_BASES  = set(_get_csv("XSTOCKS_BASES", []))      # explicit whitelist as xStocks

    # Optional guardrails (not enforced by core flow)
    MIN_24H_VOLUME = _get_float("MIN_24H_VOLUME", 0.0)
    BASE_BLACKLIST = set(_get_csv("BASE_BLACKLIST", []))

    # --- Indicator settings ---
    RSI_LENGTH = _get_int("RSI_LENGTH", 14)
    FAST_MA    = _get_int("FAST_MA", 60)
    SLOW_MA    = _get_int("SLOW_MA", 240)
    # MACD defaults
    MACD_FAST  = 12
    MACD_SLOW  = 26
    MACD_SIG   = 9

SETTINGS = Settings()

# -------------------------
# DB models
# -------------------------
Base = declarative_base()
engine = create_engine(SETTINGS.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, index=True)
    base = Column(String, index=True)
    quote = Column(String, index=True)
    first_seen = Column(DateTime, server_default=func.now())

class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True)
    base = Column(String, index=True)
    quote = Column(String, index=True)
    amount = Column(Float)             # base amount held
    avg_cost = Column(Float)           # in quote (e.g., USD per base)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint('base', 'quote', name='uq_position_base_quote'),)

class BuyLock(Base):
    __tablename__ = "buy_locks"
    id = Column(Integer, primary_key=True)
    base = Column(String, unique=True, index=True)
    active = Column(Boolean, default=True)

class TradeLog(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    side = Column(String)              # BUY/SELL
    symbol = Column(String)
    base = Column(String)
    quote = Column(String)
    price = Column(Float)
    amount = Column(Float)
    notional = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(bind=engine)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# -------------------------
# Exchange wrapper (Kraken via CCXT)
# -------------------------
class KrakenClient:
    def __init__(self):
        self.x = ccxt.kraken({
            'apiKey': SETTINGS.KRAKEN_API_KEY,
            'secret': SETTINGS.KRAKEN_API_SECRET,
            'enableRateLimit': True,
        })
        self.markets = None

    def load_markets(self):
        if self.markets is None:
            self.markets = self.x.load_markets()
        return self.markets

    def get_market(self, symbol: str) -> Optional[dict]:
        self.load_markets()
        return self.markets.get(symbol)

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_ticker_price(self, symbol: str) -> float:
        t = self.x.fetch_ticker(symbol)
        return float(t['last'])

    def list_screenable_symbols(self) -> List[Tuple[str, str, str]]:
        """
        Return list of (symbol, base, quote) we might trade.
        Filters by QUOTE_ASSETS, blacklists, xStocks preferences, etc.
        """
        self.load_markets()
        out = []
        for sym, m in self.markets.items():
            base = m.get('base')
            quote = m.get('quote')
            if not base or not quote:
                continue

            if SETTINGS.BASE_BLACKLIST and base in SETTINGS.BASE_BLACKLIST:
                continue

            is_xstock = base.endswith(SETTINGS.XSTOCKS_SUFFIX)
            is_crypto = not is_xstock

            if SETTINGS.XSTOCKS_ONLY:
                if not is_xstock:
                    continue
            else:
                if is_xstock and not SETTINGS.INCLUDE_XSTOCKS:
                    continue
                if is_crypto and not SETTINGS.INCLUDE_CRYPTO:
                    continue

            if quote not in SETTINGS.QUOTE_ASSETS:
                continue

            out.append((sym, base, quote))

        # persist symbols table
        with session_scope() as s:
            for sym, base, quote in out:
                if not s.query(Symbol).filter_by(symbol=sym).first():
                    s.add(Symbol(symbol=sym, base=base, quote=quote))
        return out

    # -----------------
    # Orders + minimums
    # -----------------
    def get_minimums(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (min_cost, min_amount, amount_step).
        - min_cost: minimum quote notional
        - min_amount: minimum base amount
        - amount_step: smallest increment in base amount (derived from precision or explicit step)
        """
        m = self.get_market(symbol) or {}
        limits = m.get('limits') or {}
        cost_min = (limits.get('cost') or {}).get('min')
        amount_min = (limits.get('amount') or {}).get('min')

        # Derive an amount step:
        step = None
        # Some exchanges expose 'precision' as decimal places (int). For Kraken, this is usually decimals.
        prec = (m.get('precision') or {}).get('amount')
        if isinstance(prec, (int, float)):
            try:
                step = 10 ** (-int(prec))
            except Exception:
                step = None
        # If exchange provides explicit step size, prefer it
        step_size = (limits.get('amount') or {}).get('step')
        if step_size is not None:
            step = step_size

        return cost_min, amount_min, step

    def market_buy_quote(self, symbol: str, quote_amount: float) -> Tuple[float, float]:
        price = self.fetch_ticker_price(symbol)
        if SETTINGS.DRY_RUN:
            # Simulate fills greedily
            base_qty = quote_amount / max(price, 1e-18)
            return price, base_qty
        order = self.x.create_order(symbol, type='market', side='buy', amount=None, params={'cost': quote_amount})
        filled_base = float(order.get('filled', 0.0))
        return price, filled_base

    def market_sell_all(self, symbol: str, base_qty: float) -> Tuple[float, float]:
        # Respect minimum amount + amount step by rounding DOWN
        min_cost, min_amount, amount_step = self.get_minimums(symbol)

        # Apply step: round down to nearest step (or leave if no step)
        if amount_step and amount_step > 0:
            multiples = math.floor(base_qty / amount_step)
            base_qty = multiples * amount_step
            try:
                amt_str = self.x.amount_to_precision(symbol, base_qty)
                base_qty = float(amt_str)
            except Exception:
                pass

        price = self.fetch_ticker_price(symbol)
        if SETTINGS.DRY_RUN:
            return price, base_qty
        self.x.create_order(symbol, type='market', side='sell', amount=base_qty)
        return price, base_qty

    # Profit sink (ATOM) helpers
    def buy_atom_for_usd(self, usd_amount: float) -> Tuple[float, float]:
        # Buy ATOM/USD or ATOM/USDT depending on availability
        pair = None
        for q in SETTINGS.QUOTE_ASSETS:
            try_pair = f"ATOM/{q}"
            try:
                self.get_market(try_pair) or self.x.load_markets()
                self.get_market(try_pair)
                pair = try_pair
                break
            except Exception:
                continue
        if not pair:
            raise RuntimeError("No ATOM/* market available for quotes: " + ",".join(SETTINGS.QUOTE_ASSETS))

        price = self.fetch_ticker_price(pair)
        base_qty = usd_amount / max(price, 1e-18)
        if SETTINGS.DRY_RUN:
            return price, base_qty
        self.x.create_order(pair, type='market', side='buy', amount=None, params={'cost': usd_amount})
        return price, base_qty

    def stake_atom(self, base_qty: float) -> None:
        # Placeholder: Kraken Earn/Equities staking endpoints aren’t available via CCXT.
        pass


# -------------------------
# Indicators & signals
# -------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-18)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_indicators(df: pd.DataFrame, rsi_len: int, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out['rsi'] = rsi(out['close'], rsi_len)
    out['ma_fast'] = moving_average(out['close'], fast)
    out['ma_slow'] = moving_average(out['close'], slow)
    macd_line, macd_sig = macd(out['close'], SETTINGS.MACD_FAST, SETTINGS.MACD_SLOW, SETTINGS.MACD_SIG)
    out['macd'] = macd_line
    out['macd_signal'] = macd_sig
    return out

def buy_signal_rowwise(row) -> bool:
    # kept for compatibility if needed elsewhere
    return (row['rsi'] <= 30) and (row['ma_fast'] < row['ma_slow'])

def buy_signal_df(ind: pd.DataFrame) -> bool:
    """
    Tightened buy: oversold + bearish, but require early turn and MACD bull cross:
      - RSI <= 30
      - MA_fast < MA_slow (downtrend)
      - RSI rising vs previous bar
      - Close is a green bar vs previous close
      - MACD crosses above signal
    """
    if len(ind) < 2:
        return False
    last2 = ind.iloc[-2:]
    rsi_now = float(last2['rsi'].iloc[-1])
    rsi_prev = float(last2['rsi'].iloc[-2])
    ma_fast_now = float(last2['ma_fast'].iloc[-1])
    ma_slow_now = float(last2['ma_slow'].iloc[-1])
    close_now = float(last2['close'].iloc[-1])
    close_prev = float(last2['close'].iloc[-2])
    macd_now = float(last2['macd'].iloc[-1])
    macd_prev = float(last2['macd'].iloc[-2])
    sig_now = float(last2['macd_signal'].iloc[-1])
    sig_prev = float(last2['macd_signal'].iloc[-2])

    basic = (
        (rsi_now <= 30.0) and
        (ma_fast_now < ma_slow_now) and
        (rsi_now > rsi_prev) and
        (close_now > close_prev)
    )
    macd_bull_cross = (macd_prev <= sig_prev) and (macd_now > sig_now)
    return basic and macd_bull_cross


def compute_order_notional(kr: 'KrakenClient', quote: str) -> float:
    # Simple allocation: flat spend per trade per run
    # Could use balance checks here; for now a fixed notional per buy
    per_trade = _get_float("PER_TRADE_NOTIONAL", 25.0)
    return per_trade


def bump_spend_to_exchange_minimums(kr: 'KrakenClient', symbol: str, desired_quote: float) -> Tuple[float, bool]:
    min_cost, min_amount, amount_step = kr.get_minimums(symbol)
    adjusted = desired_quote

    # Ensure spend meets min cost
    if min_cost is not None:
        adjusted = max(adjusted, float(min_cost))

    # Ensure base amount meets min amount and step after rounding UP
    if price := kr.fetch_ticker_price(symbol):
        base_qty = adjusted / price
        if (min_amount is not None) and (base_qty < float(min_amount) - 1e-18):
            base_qty = float(min_amount)
        if amount_step and amount_step > 0:
            multiples = math.ceil(base_qty / amount_step)
            base_qty = multiples * amount_step
        adjusted = base_qty * price

    bumped = adjusted > desired_quote + 1e-12
    return adjusted, bumped


def _estimate_24h_quote_volume(kr: 'KrakenClient', symbol: str) -> float:
    """
    Approximate last-24h quote-notional volume by summing hour bars: sum(close * volume) over last 24 hours.
    """
    try:
        df_h = kr.fetch_ohlcv_df(symbol, timeframe='1h', limit=30)
        last_24 = df_h.tail(24)
        return float((last_24['close'] * last_24['volume']).sum())
    except Exception:
        return 0.0


# -------------------------
# Core flows
# -------------------------

def screen_and_buy_signals(kr: KrakenClient):
    syms = kr.list_screenable_symbols()
    buys_this_run = 0
    for sym, base, quote in syms:
        if buys_this_run >= SETTINGS.MAX_BUYS_PER_RUN:
            break

        # Hard skip if we already hold this base OR we have an active lock
        with session_scope() as s:
            if s.query(Position).filter_by(base=base).first():
                # guarantee a lock exists for legacy positions
                if not s.query(BuyLock).filter_by(base=base).first():
                    s.add(BuyLock(base=base, active=True))
                continue
            if s.query(BuyLock).filter_by(base=base, active=True).first():
                continue

        try:
            df = kr.fetch_ohlcv_df(sym, timeframe='15m',
                                   limit=max(SETTINGS.SLOW_MA + 5, 260))
            ind = compute_indicators(df, SETTINGS.RSI_LENGTH,
                                     SETTINGS.FAST_MA, SETTINGS.SLOW_MA)
            # Enforce minimum 24h volume (quote notional)
            vol_24h_quote = _estimate_24h_quote_volume(kr, sym)
            if SETTINGS.MIN_24H_VOLUME > 0 and vol_24h_quote < SETTINGS.MIN_24H_VOLUME:
                continue

            if buy_signal_df(ind):
                desired = compute_order_notional(kr, quote)
                adjusted, bumped = bump_spend_to_exchange_minimums(kr, sym, desired)
                if bumped:
                    print(f"{sym}: bumped spend {desired:.8f} -> {adjusted:.8f} {quote} (min cost/amount/precision)")

                price, qty = kr.market_buy_quote(sym, adjusted)
                with session_scope() as s:
                    s.add(BuyLock(base=base, active=True))
                    pos = s.query(Position).filter_by(base=base, quote=quote).first()
                    if pos:
                        new_amount = pos.amount + qty
                        new_cost = (pos.avg_cost * pos.amount + price * qty) / max(new_amount, 1e-9)
                        pos.amount = new_amount
                        pos.avg_cost = new_cost
                    else:
                        s.add(Position(base=base, quote=quote, amount=qty, avg_cost=price))
                    s.add(TradeLog(side="BUY", symbol=sym, base=base, quote=quote,
                                   price=price, amount=qty, notional=price * qty))
                buys_this_run += 1

                last = ind.iloc[-1]
                ma_fast = float(last['ma_fast'])
                ma_slow = float(last['ma_slow'])
                print(
                    f"Signal BUY {sym}: rsi={last['rsi']:.1f} (rising), "
                    f"ma{SETTINGS.FAST_MA}={ma_fast:.6f} < ma{SETTINGS.SLOW_MA}={ma_slow:.6f}, macd_cross=YES, 24h_vol≈{vol_24h_quote:.2f} {quote}, price={price:.10f}, "
                    f"spend={adjusted:.8f} {quote}"
                )
        except Exception as e:
            # Permission errors and anything else are just logged—no blacklisting
            print(f"Screening error {sym}: {e}")


def take_profits_and_sink(kr: KrakenClient):
    """
    SELL RULES:
      - If ENABLE_MOONSHOT is False: sell ALL when price >= avg_cost * (1 + TAKE_PROFIT_PCT).
      - If ENABLE_MOONSHOT is True: require RSI>=70 and MA_fast>MA_slow and price >= target; sell fraction.
      - Price >= avg_cost * (1 + TAKE_PROFIT_PCT)
    Never sell ATOM (profit sink asset).
    If ENABLE_MOONSHOT, sell only a fraction and keep remainder.
    """
    total_profit_usd = 0.0

    # Sell loop over current positions
    with session_scope() as s:
        positions: List[Position] = s.query(Position).all()

    for pos in positions:
        base = pos.base
        quote = pos.quote
        if base == 'ATOM':
            continue  # never sell sink asset

        # Must have an active lock to avoid double buys while held
        with session_scope() as s:
            if not s.query(BuyLock).filter_by(base=base).first():
                s.add(BuyLock(base=base, active=True))

        sym = f"{base}/{quote}"
        try:
            df = kr.fetch_ohlcv_df(sym, timeframe='15m',
                                   limit=max(SETTINGS.SLOW_MA + 5, 260))
            ind = compute_indicators(df, SETTINGS.RSI_LENGTH,
                                     SETTINGS.FAST_MA, SETTINGS.SLOW_MA)
            last = ind.iloc[-1]
            price = float(last['close'])
            rsi_val = float(last['rsi'])
            ma_fast = float(last['ma_fast'])
            ma_slow = float(last['ma_slow'])
        except Exception as e:
            print(f"Sell check skipped for {sym} (indicator fetch failed): {e}")
            continue

        target_price = pos.avg_cost * (1.0 + SETTINGS.TAKE_PROFIT_PCT)
        cond_profit = price >= target_price

        enable_moonshot = bool(getattr(SETTINGS, "ENABLE_MOONSHOT", False))
        if not enable_moonshot:
            should_take_profit = cond_profit
        else:
            cond_rsi = rsi_val >= 70.0
            cond_ma = ma_fast > ma_slow
            should_take_profit = cond_profit and cond_rsi and cond_ma

        if should_take_profit:
            sell_fraction = float(getattr(SETTINGS, "MOONSHOT_SELL_FRACTION", 0.7)) if enable_moonshot else 1.0
            sell_fraction = min(max(sell_fraction, 0.0), 1.0)

            sell_qty = pos.amount * sell_fraction
            if sell_fraction < 1.0 and sell_qty <= 0:
                continue

            sell_price, actually_sold = kr.market_sell_all(sym, sell_qty)
            notional = sell_price * actually_sold
            cost = pos.avg_cost * actually_sold
            pnl = notional - cost
            if pos.quote in ("USD", "USDT"):
                total_profit_usd += max(pnl, 0.0)

            with session_scope() as s:
                # Update or remove position
                if sell_fraction >= 0.9999 or abs(pos.amount - actually_sold) <= 1e-12:
                    # sold all
                    s.query(Position).filter_by(base=base, quote=quote).delete()
                    # Keep lock active to prevent re-buy in the same run; optional: deactivate on cooldown elsewhere
                    print(f"SELL ALL {sym}: sold={actually_sold:.10f}, pnl≈{pnl:.2f}")
                else:
                    remaining = max(pos.amount - actually_sold, 0.0)
                    pos.amount = remaining
                    print(
                        f"PARTIAL SELL {sym}: sold={actually_sold:.10f}, remain={remaining:.10f}, "
                        f"pnl≈{pnl:.2f}. Moonshot portion left to run."
                    )

    if total_profit_usd > 0.0:
        try:
            price, qty = kr.buy_atom_for_usd(total_profit_usd)
            print(f"Profit sink: bought {qty:.10f} ATOM at {price:.10f} for ${total_profit_usd:.2f}")
            if SETTINGS.ENABLE_STAKING:
                kr.stake_atom(qty)  # placeholder
                print("(Attempted to stake ATOM via placeholder—verify API support)")
        except Exception as e:
            print(f"Failed profit sink conversion to ATOM: {e}")


# -------------------------
# Entrypoint
# -------------------------

def main():
    kr = KrakenClient()
    screen_and_buy_signals(kr)
    take_profits_and_sink(kr)

if __name__ == "__main__":
    main()
