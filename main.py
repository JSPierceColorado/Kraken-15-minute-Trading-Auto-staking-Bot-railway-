# =============================
# main.py (simplified buy logic; removed MACD/green candle dependencies)
# =============================

import os
import sys
import math
import logging
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
# Logging
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


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_EVERY_SYMBOL = _get_bool("LOG_EVERY_SYMBOL", True)

_logger = logging.getLogger("aletheia")
if not _logger.handlers:
    _logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.propagate = False

# -------------------------
# Settings
# -------------------------

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
    XSTOCKS_SUFFIX = os.getenv("XSTOCKS_SUFFIX", "x")
    XSTOCKS_BASES  = set(_get_csv("XSTOCKS_BASES", []))

    # Optional guardrails
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

_logger.info(
    "Settings: DRY_RUN=%s MAX_BUYS_PER_RUN=%s TAKE_PROFIT_PCT=%.3f MOONSHOT=%s QUOTES=%s MIN_24H_VOL=%.2f",
    SETTINGS.DRY_RUN, SETTINGS.MAX_BUYS_PER_RUN, SETTINGS.TAKE_PROFIT_PCT,
    SETTINGS.ENABLE_MOONSHOT, ",".join(SETTINGS.QUOTE_ASSETS), SETTINGS.MIN_24H_VOLUME,
)

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
    amount = Column(Float)
    avg_cost = Column(Float)
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
    side = Column(String)
    symbol = Column(String)
    base = Column(String)
    quote = Column(String)
    price = Column(Float)
    amount = Column(Float)
    notional = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(bind=engine)
_logger.info("Database initialized at %s", SETTINGS.DATABASE_URL)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        _logger.exception("DB session rolled back due to: %s", e)
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
        _logger.info("Kraken client initialized (rate limited=%s, dry_run=%s)", True, SETTINGS.DRY_RUN)

    def load_markets(self):
        if self.markets is None:
            _logger.info("Loading markets from Kraken…")
            self.markets = self.x.load_markets()
            _logger.info("Loaded %d markets", len(self.markets or {}))
        return self.markets

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_ticker_price(self, symbol: str) -> float:
        t = self.x.fetch_ticker(symbol)
        return float(t['last'])

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
    return 100 - (100 / (1 + rs))

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

def buy_signal_df(ind: pd.DataFrame) -> bool:
    """
    Simplified buy signal:
      - RSI <= 30 (oversold)
      - MA_fast < MA_slow (downtrend)
      - RSI rising vs previous bar
    """
    if len(ind) < 2:
        return False
    last2 = ind.iloc[-2:]
    rsi_now = float(last2['rsi'].iloc[-1])
    rsi_prev = float(last2['rsi'].iloc[-2])
    ma_fast_now = float(last2['ma_fast'].iloc[-1])
    ma_slow_now = float(last2['ma_slow'].iloc[-1])

    return (rsi_now <= 30.0) and (ma_fast_now < ma_slow_now) and (rsi_now > rsi_prev))

# -------------------------
# Core flows
# -------------------------

def screen_and_buy_signals(kr: KrakenClient):
    syms = kr.load_markets().keys()
    for sym in syms:
        try:
            df = kr.fetch_ohlcv_df(sym, timeframe='15m', limit=max(SETTINGS.SLOW_MA + 5, 260))
            ind = compute_indicators(df, SETTINGS.RSI_LENGTH, SETTINGS.FAST_MA, SETTINGS.SLOW_MA)
            if buy_signal_df(ind):
                _logger.info(f"BUY signal for {sym}")
        except Exception as e:
            _logger.warning(f"Error screening {sym}: {e}")

# -------------------------
# Entrypoint
# -------------------------

def main():
    _logger.info("=== Aletheia run start ===")
    kr = KrakenClient()
    try:
        screen_and_buy_signals(kr)
    except Exception:
        _logger.exception("screen_and_buy_signals crashed")
    _logger.info("=== Aletheia run end ===")

if __name__ == "__main__":
    main()
