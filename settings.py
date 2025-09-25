# =============================
# settings.py
# =============================
# Centralized configuration & env parsing

import os
from dataclasses import dataclass

def _to_bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes")

@dataclass
class Settings:
    # Kraken keys (use CCXT for trading)
    KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET: str = os.getenv("KRAKEN_API_SECRET", "")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./state.db")

    # Trading params
    QUOTE_ASSETS: tuple = tuple(os.getenv("QUOTE_ASSETS", "USD,USDT").split(","))
    # Legacy min notional kept for compatibility (no longer used when ORDER_SIZE_* present)
    MIN_NOTIONAL_USD: float = float(os.getenv("MIN_NOTIONAL_USD", "1.0"))
    RSI_LENGTH: int = int(os.getenv("RSI_LENGTH", "14"))
    FAST_MA: int = int(os.getenv("FAST_MA", "60"))      # 60×15m bars
    SLOW_MA: int = int(os.getenv("SLOW_MA", "240"))     # 240×15m bars
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.075"))

    # --- NEW: Order sizing ---
    # Mode: "USD" (fixed spend) or "PCT" (percent of available quote balance)
    ORDER_SIZE_MODE: str = os.getenv("ORDER_SIZE_MODE", "USD")
    ORDER_SIZE_USD: float = float(os.getenv("ORDER_SIZE_USD", "1.0"))
    # 0.01 = 1% of available funds in the quote currency (USD/USDT)
    ORDER_SIZE_PCT: float = float(os.getenv("ORDER_SIZE_PCT", "0.01"))

    # Scheduling
    LOOP_SLEEP_SECONDS: int = int(os.getenv("LOOP_SLEEP_SECONDS", str(15 * 60)))

    # Staking / Profit sink
    PROFIT_SINK_SYMBOL: str = os.getenv("PROFIT_SINK_SYMBOL", "ATOM")
    ENABLE_STAKING: bool = _to_bool(os.getenv("ENABLE_STAKING", "false"))

    # Safety
    DRY_RUN: bool = _to_bool(os.getenv("DRY_RUN", "true"))

    # ---------- Asset-class filters (for Kraken tokenized equities = xStocks) ----------
    # Scan normal crypto pairs?
    INCLUDE_CRYPTO: bool = _to_bool(os.getenv("INCLUDE_CRYPTO", "true"))
    # Scan tokenized equities (xStocks like AAPLx, TSLAx, SPYx)?
    INCLUDE_XSTOCKS: bool = _to_bool(os.getenv("INCLUDE_XSTOCKS", "true"))
    # If true, only scan xStocks (ignore crypto)
    XSTOCKS_ONLY: bool = _to_bool(os.getenv("XSTOCKS_ONLY", "false"))
    # Suffix Kraken uses for tokenized equities base symbols (e.g., AAPLx)
    XSTOCKS_SUFFIX: str = os.getenv("XSTOCKS_SUFFIX", "x")
    # Optional explicit whitelist of xStocks bases, comma-separated (e.g., "AAPLx,TSLAx,SPYx")
    XSTOCKS_BASES: tuple = tuple(
        b.strip().upper() for b in os.getenv("XSTOCKS_BASES", "").split(",") if b.strip()
    )

SETTINGS = Settings()
