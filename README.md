# Kraken 15‑Minute Trading & Auto-Staking Bot (Railway)


## What it does
- Every 15 minutes, screens all active Kraken markets with quotes in USD/USDT.
- **New listings**: first time a market is seen, buys $1 notional.
- **Signal buys**: Buys $1 when RSI(14) <= 30 **and** MA(60) < MA(240) on 15‑minute candles.
- Ensures **one active buy lock per base asset**.
- Takes profit by selling full position at **+7.5%** over avg cost.
- Redirects realized profits to **ATOM**; optionally attempts staking (placeholder).
- Never sells ATOM.


## Environment Variables
- `KRAKEN_API_KEY`, `KRAKEN_API_SECRET` – Kraken credentials
- `DATABASE_URL` – e.g., `postgresql://user:pass@host:5432/db` or default `sqlite:///./state.db`
- `QUOTE_ASSETS` – Comma list of quote currencies to scan (default `USD,USDT`)
- `MIN_NOTIONAL_USD` – Per-buy spend (default `1.0`)
- `RSI_LENGTH` – RSI period (default `14`)
- `FAST_MA` – Fast MA window in bars (default `60`)
- `SLOW_MA` – Slow MA window in bars (default `240`)
- `TAKE_PROFIT_PCT` – e.g., `0.075` for 7.5%
- `LOOP_SLEEP_SECONDS` – if using looped runner (default 900)
- `PROFIT_SINK_SYMBOL` – default `ATOM`
- `ENABLE_STAKING` – `true/false` to attempt staking placeholder
- `DRY_RUN` – `true/false`; when `true` no orders are sent


## Railway setup
1. Create a new Railway service from this repo.
2. Add environment variables (`KRAKEN_API_KEY`, `KRAKEN_API_SECRET`, etc.).
3. For persistence, prefer a Railway **Postgres** plugin; set `DATABASE_URL` accordingly.
4. Enable **Cron** with `*/15 * * * *` running `python main.py`.


## Notes & Caveats
- CCXT abstracts Kraken trading & candles. Staking (Kraken Earn) is **not** supported via CCXT; the `stake_atom` call is a placeholder. Depending on your jurisdiction, Earn/Staking may be restricted—verify with Kraken before enabling.
- Profit calculation uses our tracked average cost per position. If you have external deposits/trades, reconcile or reset the DB to avoid mismatched PnL.
- “Newly listed” is defined as the first time this bot observes a symbol in `load_markets()`; it will buy $1 immediately. If you redeploy without DB persistence, it may re-buy. Use a persisted DB.
- This bot does **market** orders for simplicity and tiny notionals; for larger sizes consider slippage controls.
- DRY_RUN defaults to `true`. Set `DRY_RUN=false` to trade live.
