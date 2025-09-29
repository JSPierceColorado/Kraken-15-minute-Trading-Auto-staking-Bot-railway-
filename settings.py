import os

class Settings:
    # --- Database ---
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")

    # --- API keys ---
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    # --- Indicator settings ---
    RSI_LENGTH = 14
    FAST_MA = 60
    SLOW_MA = 240

    # --- Order sizing ---
    ORDER_SIZE_MODE = "USD"       # "USD" or "PCT"
    ORDER_SIZE_USD = 20.0
    ORDER_SIZE_PCT = 0.05

    # --- Profit taking ---
    TAKE_PROFIT_PCT = 0.10

    # --- Profit sink ---
    PROFIT_SINK_SYMBOL = "ATOM"
    ENABLE_STAKING = False

    # --- Loop settings ---
    LOOP_SLEEP_SECONDS = 60

    # --- Fees & backtest defaults ---
    TAKER_FEE_PCT = 0.001
    MIN_NOTIONAL = 5.0
    BACKTEST_SEED_CASH = 10_000.0

    # --- Moonshot mode ---
    ENABLE_MOONSHOT = False
    MOONSHOT_SELL_FRACTION = 0.70

SETTINGS = Settings()
