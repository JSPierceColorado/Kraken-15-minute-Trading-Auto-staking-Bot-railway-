# =============================
# main.py (simplified indicators and buy logic)
# =============================

import os
import sys
import math
import logging
from contextlib import contextmanager
from typing import List, Tuple, Optional
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
    DRY_RUN = _get_bool("DRY_RUN", False)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")
    QUOTE_ASSETS = _get_csv("QUOTE_ASSETS", ["USD", "USDT"])
    RSI_LENGTH = _get_int("RSI_LENGTH", 14)
    FAST_MA = _get_int("FAST_MA", 60)
    SLOW_MA = _get_int("SLOW_MA", 240)
    MAX_BUYS_PER_RUN = _get_int("MAX_BUYS_PER_RUN", 3)

SETTINGS = Settings()

_logger.info(
    "Settings: DRY_RUN=%s MAX_BUYS_PER_RUN=%s QUOTES=%s",
    SETTINGS.DRY_RUN, SETTINGS.MAX_BUYS_PER_RUN, ",".join(SETTINGS.QUOTE_ASSETS),
)

# -------------------------
# DB setup
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

Base.metadata.create_all(bind=engine)

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
# Exchange client
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

# -------------------------
# Indicators
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

def compute_indicators(df: pd.DataFrame, rsi_len: int, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out['rsi'] = rsi(out['close'], rsi_len)
    out['ma_fast'] = moving_average(out['close'], fast)
    out['ma_slow'] = moving_average(out['close'], slow)
    return out

# -------------------------
# Buy signal
# -------------------------

def buy_signal_df(ind: pd.DataFrame) -> bool:
    """
    Buy when:
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

    return (rsi_now <= 30.0) and (ma_fast_now < ma_slow_now) and (rsi_now > rsi_prev)

# -------------------------
# Core flow
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
