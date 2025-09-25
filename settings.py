import os
from dataclasses import dataclass


@dataclass
class Settings:
# Kraken keys (use CCXT for trading)
KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET: str = os.getenv("KRAKEN_API_SECRET", "")


# Database
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./state.db")


# Trading params
QUOTE_ASSETS: tuple = tuple(os.getenv("QUOTE_ASSETS", "USD,USDT").split(","))
MIN_NOTIONAL_USD: float = float(os.getenv("MIN_NOTIONAL_USD", "1.0"))
RSI_LENGTH: int = int(os.getenv("RSI_LENGTH", "14"))
FAST_MA: int = int(os.getenv("FAST_MA", "60")) # 60×15m
SLOW_MA: int = int(os.getenv("SLOW_MA", "240")) # 240×15m
TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.075"))


# Scheduling
LOOP_SLEEP_SECONDS: int = int(os.getenv("LOOP_SLEEP_SECONDS", str(15*60)))


# Staking / Profit sink
PROFIT_SINK_SYMBOL: str = os.getenv("PROFIT_SINK_SYMBOL", "ATOM")
ENABLE_STAKING: bool = os.getenv("ENABLE_STAKING", "false").lower() in ("1","true","yes")


# Safety
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("1","true","yes")


SETTINGS = Settings()
