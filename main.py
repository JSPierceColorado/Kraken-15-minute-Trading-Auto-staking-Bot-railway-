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
def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

def _get_csv(name: str, default_list):
    val = os.getenv(name)
    if not val:
        return list(default_list)
    return [x.strip() for x in val.split(",") if x.strip()]

class Settings:
    # --- Database ---
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

    # --- Order sizing ---
    ORDER_SIZE_MODE = os.getenv("ORDER_SIZE_MODE", "USD")    # "USD" or "PCT"
    ORDER_SIZE_USD  = _get_float("ORDER_SIZE_USD", 20.0)
    ORDER_SIZE_PCT  = _get_float("ORDER_SIZE_PCT", 0.05)     # 5% of free quote

    # --- Profit taking ---
    TAKE_PROFIT_PCT = _get_float("TAKE_PROFIT_PCT", 0.10)    # 10%

    # --- Profit sink ---
    PROFIT_SINK_SYMBOL = os.getenv("PROFIT_SINK_SYMBOL", "ATOM")
    ENABLE_STAKING     = _get_bool("ENABLE_STAKING", False)

    # --- Execution / dry-run ---
    DRY_RUN = _get_bool("DRY_RUN", True)

    # --- Loop / pacing ---
    LOOP_SLEEP_SECONDS = _get_int("LOOP_SLEEP_SECONDS", 60)
    PER_SYMBOL_DELAY_S = _get_int("PER_SYMBOL_DELAY_S", 0)

    # --- Misc defaults (not used in live flow) ---
    TAKER_FEE_PCT      = _get_float("TAKER_FEE_PCT", 0.001)  # 0.10%
    MIN_NOTIONAL       = _get_float("MIN_NOTIONAL", 5.0)
    BACKTEST_SEED_CASH = _get_float("BACKTEST_SEED_CASH", 10_000.0)

    # --- Moonshot mode (partial take-profit) ---
    ENABLE_MOONSHOT = _get_bool("ENABLE_MOONSHOT", False)
    MOONSHOT_SELL_FRACTION = _get_float("MOONSHOT_SELL_FRACTION", 0.70)

    # --- Safety cap for buys per run ---
    MAX_BUYS_PER_RUN = _get_int("MAX_BUYS_PER_RUN", 3)

SETTINGS = Settings()


# -------------------------
# DB models
# -------------------------
Base = declarative_base()
engine = create_engine(SETTINGS.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class SeenMarket(Base):
    __tablename__ = "seen_markets"
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
    __table_args__ = (UniqueConstraint("base", "quote", name="uix_base_quote"),)

class BuyLock(Base):
    __tablename__ = "buy_locks"
    id = Column(Integer, primary_key=True)
    base = Column(String, index=True)
    active = Column(Boolean, default=True)  # one active buy at a time per base
    __table_args__ = (UniqueConstraint("base", name="uix_buylock_base"),)

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
    pnl = Column(Float, nullable=True) # for sells
    ts = Column(DateTime, server_default=func.now())

def init_db():
    Base.metadata.create_all(engine)


# -------------------------
# Exchange wrapper
# -------------------------
class KrakenClient:
    def __init__(self):
        self.x = ccxt.kraken({
            'apiKey': SETTINGS.KRAKEN_API_KEY,
            'secret': SETTINGS.KRAKEN_API_SECRET,
            'enableRateLimit': True,
            'options': {'fetchOHLCVWarning': False}
        })

    def load_markets(self) -> Dict:
        return self.x.load_markets(reload=True)

    def get_market(self, symbol: str) -> dict:
        markets = self.x.load_markets()  # cached
        return markets.get(symbol, {})

    def get_min_cost_amount_step(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
                # If prec is decimals, step = 10^-prec
                step = pow(10.0, -float(prec))
            except Exception:
                step = None
        # If CCXT exposes a direct step in limits.amount.step, prefer it
        step_direct = (limits.get('amount') or {}).get('step') if limits.get('amount') else None
        if step_direct is not None:
            step = float(step_direct)

        return (
            float(cost_min) if cost_min is not None else None,
            float(amount_min) if amount_min is not None else None,
            float(step) if step is not None else None,
        )

    def _is_xstock(self, base: str) -> bool:
        if not base:
            return False
        if base.lower().endswith(SETTINGS.XSTOCKS_SUFFIX.lower()):
            return True
        if base.upper() in SETTINGS.XSTOCKS_BASES:
            return True
        return False

    def list_screenable_symbols(self) -> List[Tuple[str, str, str]]:
        mkts = self.load_markets()
        out: List[Tuple[str, str, str]] = []
        for sym, m in mkts.items():
            base = m.get('base')
            quote = m.get('quote')
            if not base or not quote:
                continue
            if quote not in SETTINGS.QUOTE_ASSETS:
                continue
            if not m.get('active', True):
                continue
            if base in SETTINGS.BASE_BLACKLIST:
                continue

            is_x = self._is_xstock(base)
            if SETTINGS.XSTOCKS_ONLY:
                if SETTINGS.INCLUDE_XSTOCKS and is_x:
                    out.append((sym, base, quote))
                continue

            if is_x and SETTINGS.INCLUDE_XSTOCKS:
                out.append((sym, base, quote))
            elif (not is_x) and SETTINGS.INCLUDE_CRYPTO:
                out.append((sym, base, quote))
        return out

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_ticker_price(self, symbol: str) -> float:
        t = self.x.fetch_ticker(symbol)
        return float(t['last'])

    def get_free_balance(self, currency: str) -> float:
        try:
            bal = self.x.fetch_free_balance()
            return float(bal.get(currency, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _amount_to_precision_ceil(self, symbol: str, amount: float, step: Optional[float]) -> float:
        """
        Convert amount to a valid tradable amount, rounding UP to the next step if needed.
        Uses CCXT's amount_to_precision to enforce decimal places, then adjusts upward
        to meet a known step if provided.
        """
        if amount <= 0:
            return 0.0
        # First adhere to decimal precision CCXT expects
        try:
            amt_str = self.x.amount_to_precision(symbol, amount)
            amt = float(amt_str)
        except Exception:
            amt = amount

        if step and step > 0:
            # ceil to next multiple of 'step'
            multiples = math.ceil(amt / step)
            amt = multiples * step
            # Re-apply precision formatting after stepping
            try:
                amt_str = self.x.amount_to_precision(symbol, amt)
                amt = float(amt_str)
            except Exception:
                pass
        return amt

    def market_buy_quote(self, symbol: str, quote_notional: float) -> Tuple[float, float]:
        """
        Market buy spending 'quote_notional' units of the QUOTE currency (USD/USDT).
        Returns (price, base_qty) actually requested.
        """
        price = self.fetch_ticker_price(symbol)
        base_qty = quote_notional / price if price > 0 else 0.0

        # Enforce precision/step by rounding UP
        _, min_amount, amount_step = self.get_min_cost_amount_step(symbol)
        base_qty = self._amount_to_precision_ceil(symbol, base_qty, amount_step)

        if SETTINGS.DRY_RUN:
            return price, base_qty

        self.x.create_order(symbol, type='market', side='buy', amount=base_qty)
        return price, base_qty

    def market_sell_all(self, symbol: str, base_qty: float) -> Tuple[float, float]:
        # For sells, round DOWN a hair to avoid “too much” errors, but still respect precision
        _, _, amount_step = self.get_min_cost_amount_step(symbol)
        try:
            amt_str = self.x.amount_to_precision(symbol, base_qty)
            base_qty = float(amt_str)
        except Exception:
            pass
        if amount_step and amount_step > 0:
            # floor to step
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

    def buy_atom_for_usd(self, usd_amount: float) -> Tuple[float, float]:
        # Prefer USD quote if available, else USDT
        for q in SETTINGS.QUOTE_ASSETS:
            sym = f"{SETTINGS.PROFIT_SINK_SYMBOL}/{q}"
            try:
                return self.market_buy_quote(sym, usd_amount)
            except Exception:
                continue
        raise RuntimeError("No ATOM/* market available for quotes: " + ",".join(SETTINGS.QUOTE_ASSETS))

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
    rs = roll_up / (roll_down + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def compute_indicators(df: pd.DataFrame, rsi_len: int, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out['rsi'] = rsi(out['close'], rsi_len)
    out['ma_fast'] = moving_average(out['close'], fast)
    out['ma_slow'] = moving_average(out['close'], slow)
    return out

def buy_signal_rowwise(row) -> bool:
    # kept for compatibility if needed elsewhere
    return (row['rsi'] <= 30) and (row['ma_fast'] < row['ma_slow'])

def buy_signal_df(ind: pd.DataFrame) -> bool:
    """
    Tightened buy: oversold + bearish, but require early turn:
      - RSI <= 30
      - MA_fast < MA_slow (downtrend)
      - RSI rising vs previous bar
      - Close is a green bar vs previous close
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

    return (
        (rsi_now <= 30.0) and
        (ma_fast_now < ma_slow_now) and
        (rsi_now > rsi_prev) and
        (close_now > close_prev)
    )


# -------------------------
# Orchestration helpers
# -------------------------
BUMP_BUFFER = 1.01  # 1% safety buffer above calculated minimums

def required_spend_for_amount(symbol: str, price: float, amount: float) -> float:
    if price and price > 0:
        return amount * price
    return 0.0

def bump_spend_to_exchange_minimums(kr: KrakenClient, symbol: str, desired_spend: float) -> Tuple[float, bool]:
    """
    Bump spend to satisfy:
      - min cost (quote notional)
      - min amount (base)
      - amount step/precision (by rounding UP to next valid step)
    Returns (adjusted_spend, was_bumped)
    """
    if desired_spend <= 0:
        return 0.0, False

    min_cost, min_amount, amount_step = kr.get_min_cost_amount_step(symbol)
    try:
        price = kr.fetch_ticker_price(symbol)
    except Exception:
        price = 0.0

    adjusted = desired_spend

    # Ensure spend meets min cost
    if min_cost is not None:
        adjusted = max(adjusted, float(min_cost))

    # Ensure base amount meets min amount and step after rounding UP
    if price and price > 0:
        # target base amount from current spend
        base_qty = adjusted / price

        # apply min amount if any
        if (min_amount is not None) and (base_qty < float(min_amount) - 1e-18):
            base_qty = float(min_amount)

        # apply step: round UP to next valid increment
        if amount_step and amount_step > 0:
            base_qty = math.ceil(base_qty / amount_step) * amount_step

        # re-compute spend from the rounded/stepped amount
        adjusted = required_spend_for_amount(symbol, price, base_qty)

    # Add tiny buffer to avoid boundary rejects
    adjusted *= BUMP_BUFFER
    return adjusted, (adjusted > desired_spend + 1e-12)


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def validate_settings():
    assert SETTINGS.FAST_MA < SETTINGS.SLOW_MA, "FAST_MA must be < SLOW_MA"
    if getattr(SETTINGS, "ENABLE_MOONSHOT", False):
        frac = float(getattr(SETTINGS, "MOONSHOT_SELL_FRACTION", 0.7))
        assert 0.0 < frac < 1.0, "MOONSHOT_SELL_FRACTION must be in (0,1)"

def compute_order_notional(kr: KrakenClient, quote: str) -> float:
    """
    Returns how much QUOTE (USD/USDT) to spend for this order based on env:
      - ORDER_SIZE_MODE=USD -> ORDER_SIZE_USD
      - ORDER_SIZE_MODE=PCT -> ORDER_SIZE_PCT * free balance of QUOTE
    """
    mode = (SETTINGS.ORDER_SIZE_MODE or "USD").upper()
    if mode == "PCT":
        avail = kr.get_free_balance(quote)
        spend = max(0.0, avail * SETTINGS.ORDER_SIZE_PCT)
        return spend
    return SETTINGS.ORDER_SIZE_USD

def seed_markets_if_empty(kr: KrakenClient) -> bool:
    """
    If the seen_markets table is empty, seed it with ALL current markets (no buys).
    Returns True if we seeded, False if not.
    """
    syms = kr.list_screenable_symbols()
    with session_scope() as s:
        count = s.query(SeenMarket).count()
        if count > 0:
            return False
        for sym, base, quote in syms:
            s.add(SeenMarket(symbol=sym, base=base, quote=quote))
    print(f"Bootstrap: seeded {len(syms)} markets — no buys on first run.")
    return True

def find_new_markets(kr: KrakenClient):
    """
    Return only markets that are NOT in seen_markets yet (true new listings since last run).
    """
    syms = kr.list_screenable_symbols()
    new_symbols = []
    with session_scope() as s:
        for sym, base, quote in syms:
            if not s.query(SeenMarket).filter_by(symbol=sym).first():
                s.add(SeenMarket(symbol=sym, base=base, quote=quote))
                new_symbols.append((sym, base, quote))
    return new_symbols

def buy_new_listings(kr: KrakenClient, new_symbols):
    buys_this_run = 0
    for sym, base, quote in new_symbols:
        if buys_this_run >= SETTINGS.MAX_BUYS_PER_RUN:
            break
        try:
            # Hard skip if we already hold this base (any quote)
            with session_scope() as s:
                if s.query(Position).filter_by(base=base).first():
                    # ensure a lock exists for legacy positions
                    if not s.query(BuyLock).filter_by(base=base).first():
                        s.add(BuyLock(base=base, active=True))
                    continue

            desired = compute_order_notional(kr, quote)
            adjusted, bumped = bump_spend_to_exchange_minimums(kr, sym, desired)
            if bumped:
                print(f"{sym}: bumped spend {desired:.8f} -> {adjusted:.8f} {quote} (min cost/amount/precision)")

            price, qty = kr.market_buy_quote(sym, adjusted)
            with session_scope() as s:
                if not s.query(BuyLock).filter_by(base=base).first():
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
            print(f"New listing buy: {sym} spend={adjusted:.8f} {quote} at {price:.10f}")
        except Exception as e:
            print(f"Buy error {sym}: {e}")

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
                    f"ma{SETTINGS.FAST_MA}={ma_fast:.6f} < ma{SETTINGS.SLOW_MA}={ma_slow:.6f}, price={price:.10f}, "
                    f"spend={adjusted:.8f} {quote}"
                )
        except Exception as e:
            # Permission errors and anything else are just logged—no blacklisting
            print(f"Screening error {sym}: {e}")

def take_profits_and_sink(kr: KrakenClient):
    """
    SELL RULE (all must be true):
      - RSI(14) >= 70
      - MA60 > MA240  (15m bars)
      - Price >= avg_cost * (1 + TAKE_PROFIT_PCT)
    Never sell ATOM (profit sink asset).
    If ENABLE_MOONSHOT, sell only a fraction and keep remainder.
    """
    total_profit_usd = 0.0
    with session_scope() as s:
        positions = s.query(Position).all()
        for pos in positions:
            if pos.base == SETTINGS.PROFIT_SINK_SYMBOL:
                continue

            sym = f"{pos.base}/{pos.quote}"
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

            cond_rsi = rsi_val >= 70.0
            cond_ma = ma_fast > ma_slow
            target_price = pos.avg_cost * (1.0 + SETTINGS.TAKE_PROFIT_PCT)
            cond_profit = price >= target_price
            should_take_profit = cond_rsi and cond_ma and cond_profit

            if should_take_profit:
                enable_moonshot = bool(getattr(SETTINGS, "ENABLE_MOONSHOT", False))
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
                    total_profit_usd += pnl

                s.add(TradeLog(side="SELL", symbol=sym, base=pos.base, quote=pos.quote,
                               price=sell_price, amount=actually_sold, notional=notional, pnl=pnl))

                remaining = pos.amount - actually_sold
                if remaining <= 0:
                    s.delete(pos)
                    bl = s.query(BuyLock).filter_by(base=pos.base, active=True).first()
                    if bl:
                        s.delete(bl)
                    profit_pct = (sell_price / pos.avg_cost - 1.0) * 100.0
                    print(
                        f"SELL {sym}: rsi={rsi_val:.1f} (>=70), ma{SETTINGS.FAST_MA}={ma_fast:.6f} > "
                        f"ma{SETTINGS.SLOW_MA}={ma_slow:.6f}, price={sell_price:.10f} (avg={pos.avg_cost:.10f}, "
                        f"+{profit_pct:.2f}%), pnl≈{pnl:.2f}"
                    )
                else:
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
            print(f"Failed profit sink buy ATOM: {e}")

def sync_buylocks_with_positions():
    """
    Ensure a BuyLock exists for every held base across any quote.
    This prevents accidental additional buys if legacy runs lacked locks.
    """
    with session_scope() as s:
        held_bases = {p.base for p in s.query(Position).all() if (p.amount or 0) > 0}
        for b in held_bases:
            if not s.query(BuyLock).filter_by(base=b).first():
                s.add(BuyLock(base=b, active=True))

def run_once():
    validate_settings()
    init_db()
    kr = KrakenClient()

    # One-time sanity: guarantee locks for held positions
    sync_buylocks_with_positions()

    bootstrapped = seed_markets_if_empty(kr)
    new_syms = find_new_markets(kr)

    if (not bootstrapped) and new_syms:
        buy_new_listings(kr, new_syms)

    screen_and_buy_signals(kr)
    take_profits_and_sink(kr)

if __name__ == "__main__":
    run_once()
    # Uncomment to loop:
    # import time
    # while True:
    #     try:
    #         run_once()
    #     except Exception as e:
    #         print("Run error:", e)
    #     time.sleep(SETTINGS.LOOP_SLEEP_SECONDS)
