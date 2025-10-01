# =============================
# main.py (all-in-one, no blacklisting, auto-bump to exchange minimums)
# =============================

import os
from contextlib import contextmanager
from typing import List, Dict, Tuple

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
        markets = self.x.load_markets()
        return markets.get(symbol, {})

    def get_min_cost_and_amount(self, symbol: str) -> Tuple[float, float]:
        """
        Returns (min_cost, min_amount) if provided by exchange metadata, else (None, None).
        """
        m = self.get_market(symbol) or {}
        limits = m.get('limits') or {}
        cost_min = (limits.get('cost') or {}).get('min')
        amount_min = (limits.get('amount') or {}).get('min')
        return (float(cost_min) if cost_min is not None else None,
                float(amount_min) if amount_min is not None else None)

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

    def market_buy_quote(self, symbol: str, quote_notional: float) -> Tuple[float, float]:
        """
        Market buy spending 'quote_notional' units of the QUOTE currency (USD/USDT).
        Returns (price, base_qty).
        """
        price = self.fetch_ticker_price(symbol)
        base_qty = quote_notional / price if price > 0 else 0.0
        if SETTINGS.DRY_RUN:
            return price, base_qty
        self.x.create_order(symbol, type='market', side='buy', amount=base_qty)
        return price, base_qty

    def market_sell_all(self, symbol: str, base_qty: float) -> Tuple[float, float]:
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

def buy_signal(row) -> bool:
    return (row['rsi'] <= 30) and (row['ma_fast'] < row['ma_slow'])


# -------------------------
# Orchestration helpers
# -------------------------
BaseBumpBuffer = 1.01  # bump 1% above calculated minimum to avoid boundary rejects

def bump_spend_to_exchange_minimums(kr: KrakenClient, symbol: str, desired_spend: float, buffer: float = BaseBumpBuffer) -> Tuple[float, bool]:
    """
    Ensure spend meets exchange minimums. Returns (adjusted_spend, bumped_flag).
    If exchange defines min cost or min amount, we raise spend accordingly and add a small buffer.
    """
    if desired_spend <= 0:
        return 0.0, False

    min_cost, min_amount = kr.get_min_cost_and_amount(symbol)
    price = 0.0
    try:
        price = kr.fetch_ticker_price(symbol)
    except Exception:
        pass

    required_spend = desired_spend
    if min_cost is not None:
        required_spend = max(required_spend, float(min_cost))
    if min_amount is not None and price and price > 0:
        required_spend = max(required_spend, float(min_amount) * price)

    if required_spend > desired_spend + 1e-12:
        return required_spend * (buffer if buffer > 1.0 else 1.0), True
    return desired_spend, False


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
    for sym, base, quote in new_symbols:
        try:
            order_notional = compute_order_notional(kr, quote)
            bumped, was_bumped = bump_spend_to_exchange_minimums(kr, sym, order_notional)
            if was_bumped:
                print(f"{sym}: bumped spend {order_notional:.2f} -> {bumped:.2f} {quote} to satisfy exchange minimums")
            order_notional = bumped

            price, qty = kr.market_buy_quote(sym, order_notional)
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
            print(f"New listing buy: {sym} spend={order_notional:.2f} {quote} at {price:.6f}")
        except Exception as e:
            # Log and continue; do NOT blacklist restricted assets
            print(f"Buy error {sym}: {e}")

def screen_and_buy_signals(kr: KrakenClient):
    syms = kr.list_screenable_symbols()
    for sym, base, quote in syms:
        # Only one active buy per base at a time
        with session_scope() as s:
            if s.query(BuyLock).filter_by(base=base, active=True).first():
                continue
        try:
            df = kr.fetch_ohlcv_df(sym, timeframe='15m',
                                   limit=max(SETTINGS.SLOW_MA + 5, 260))
            ind = compute_indicators(df, SETTINGS.RSI_LENGTH,
                                     SETTINGS.FAST_MA, SETTINGS.SLOW_MA)
            last = ind.iloc[-1]
            if buy_signal(last):
                order_notional = compute_order_notional(kr, quote)
                bumped, was_bumped = bump_spend_to_exchange_minimums(kr, sym, order_notional)
                if was_bumped:
                    print(f"{sym}: bumped spend {order_notional:.2f} -> {bumped:.2f} {quote} to satisfy exchange minimums")
                order_notional = bumped

                price, qty = kr.market_buy_quote(sym, order_notional)
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
                ma_fast = float(last['ma_fast'])
                ma_slow = float(last['ma_slow'])
                print(
                    f"Signal BUY {sym}: rsi={last['rsi']:.1f}, "
                    f"ma{SETTINGS.FAST_MA}={ma_fast:.6f} > ma{SETTINGS.SLOW_MA}={ma_slow:.6f}, price={price:.6f}, "
                    f"spend={order_notional:.2f} {quote}"
                )
        except Exception as e:
            # This will include permission errors; we just log them and move on
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
                        f"ma{SETTINGS.SLOW_MA}={ma_slow:.6f}, price={sell_price:.6f} (avg={pos.avg_cost:.6f}, "
                        f"+{profit_pct:.2f}%), pnl≈{pnl:.2f}"
                    )
                else:
                    pos.amount = remaining
                    print(
                        f"PARTIAL SELL {sym}: sold={actually_sold:.6f}, remain={remaining:.6f}, "
                        f"pnl≈{pnl:.2f}. Moonshot portion left to run."
                    )

    if total_profit_usd > 0.0:
        try:
            price, qty = kr.buy_atom_for_usd(total_profit_usd)
            print(f"Profit sink: bought {qty:.6f} ATOM at {price:.6f} for ${total_profit_usd:.2f}")
            if SETTINGS.ENABLE_STAKING:
                kr.stake_atom(qty)  # placeholder
                print("(Attempted to stake ATOM via placeholder—verify API support)")
        except Exception as e:
            print(f"Failed profit sink buy ATOM: {e}")

def run_once():
    validate_settings()
    init_db()
    kr = KrakenClient()

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
