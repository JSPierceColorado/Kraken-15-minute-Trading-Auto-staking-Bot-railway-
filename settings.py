# =============================
# settings.py
# =============================
import os

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

    # --- Universe / filters used by KrakenClient.list_screenable_symbols() ---
    QUOTE_ASSETS   = _get_csv("QUOTE_ASSETS", ["USD", "USDT"])
    INCLUDE_CRYPTO = _get_bool("INCLUDE_CRYPTO", True)
    INCLUDE_XSTOCKS = _get_bool("INCLUDE_XSTOCKS", True)
    XSTOCKS_ONLY   = _get_bool("XSTOCKS_ONLY", False)
    XSTOCKS_SUFFIX = os.getenv("XSTOCKS_SUFFIX", "x")        # e.g., "AAPLx"
    XSTOCKS_BASES  = set(_get_csv("XSTOCKS_BASES", []))      # explicit whitelist as xStocks

    # Optional liquidity guardrails (not required by code, safe to keep)
    MIN_24H_VOLUME = _get_float("MIN_24H_VOLUME", 0.0)       # in quote units
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
    DRY_RUN = _get_bool("DRY_RUN", True)                     # simulate orders by default

    # --- Loop / pacing ---
    LOOP_SLEEP_SECONDS = _get_int("LOOP_SLEEP_SECONDS", 60)
    PER_SYMBOL_DELAY_S = _get_int("PER_SYMBOL_DELAY_S", 0)

    # --- Fees & backtest defaults ---
    TAKER_FEE_PCT      = _get_float("TAKER_FEE_PCT", 0.001)  # 0.10%
    MIN_NOTIONAL       = _get_float("MIN_NOTIONAL", 5.0)
    BACKTEST_SEED_CASH = _get_float("BACKTEST_SEED_CASH", 10_000.0)

    # --- Moonshot mode (partial take-profit) ---
    ENABLE_MOONSHOT = _get_bool("ENABLE_MOONSHOT", False)    # off by default
    MOONSHOT_SELL_FRACTION = _get_float("MOONSHOT_SELL_FRACTION", 0.70)

SETTINGS = Settings()
