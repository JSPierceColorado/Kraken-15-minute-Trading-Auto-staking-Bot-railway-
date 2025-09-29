# =============================
# settings.py
# =============================

class Settings:
    # --- Indicator settings ---
    RSI_LENGTH = 14
    FAST_MA = 60
    SLOW_MA = 240

    # --- Order sizing ---
    # ORDER_SIZE_MODE = "USD" or "PCT"
    ORDER_SIZE_MODE = "USD"
    ORDER_SIZE_USD = 20.0         # used if mode=USD
    ORDER_SIZE_PCT = 0.05         # used if mode=PCT (fraction of free balance)

    # --- Profit taking ---
    TAKE_PROFIT_PCT = 0.10        # 10% profit target

    # --- Profit sink ---
    PROFIT_SINK_SYMBOL = "ATOM"   # asset to accumulate profits
    ENABLE_STAKING = False        # placeholder; depends on exchange API

    # --- Loop settings ---
    LOOP_SLEEP_SECONDS = 60       # delay between run_once() iterations

    # --- Fees & backtest defaults ---
    TAKER_FEE_PCT = 0.001         # 0.1% fee assumption
    MIN_NOTIONAL = 5.0            # fallback if exchange rules unavailable
    BACKTEST_SEED_CASH = 10_000.0

    # --- NEW: moonshot mode ---
    ENABLE_MOONSHOT = False       # False = behave as before
    MOONSHOT_SELL_FRACTION = 0.70 # fraction to sell at TP if enabled (0.7 = 70%)

SETTINGS = Settings()
