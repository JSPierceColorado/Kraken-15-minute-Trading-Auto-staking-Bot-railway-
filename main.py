import math
except Exception as e:
print(f"Screening error {sym}: {e}")




def take_profits_and_sink(kr: KrakenClient):
total_profit_usd = 0.0
with session_scope() as s:
positions = s.query(Position).all()
for pos in positions:
if pos.base == SETTINGS.PROFIT_SINK_SYMBOL:
continue # never sell ATOM (principal stays)
sym = f"{pos.base}/{pos.quote}"
try:
price = kr.fetch_ticker_price(sym)
except Exception:
continue
if price >= pos.avg_cost*(1.0 + SETTINGS.TAKE_PROFIT_PCT):
# sell all
sell_price, sold_qty = kr.market_sell_all(sym, pos.amount)
notional = sell_price * sold_qty
cost = pos.avg_cost * sold_qty
pnl = notional - cost
total_profit_usd += pnl if pos.quote in ("USD", "USDT") else 0.0
s.add(TradeLog(side="SELL", symbol=sym, base=pos.base, quote=pos.quote, price=sell_price, amount=sold_qty, notional=notional, pnl=pnl))
# clear position & buy lock
s.delete(pos)
bl = s.query(BuyLock).filter_by(base=pos.base, active=True).first()
if bl:
s.delete(bl)
print(f"TP SELL {sym}: sold {sold_qty:.8f} at {sell_price:.6f} pnl≈{pnl:.2f}")


# sink profits to ATOM
if total_profit_usd > 0.0:
try:
price, qty = kr.buy_atom_for_usd(total_profit_usd)
print(f"Profit sink: bought {qty:.6f} ATOM at {price:.6f} for ${total_profit_usd:.2f}")
if SETTINGS.ENABLE_STAKING:
kr.stake_atom(qty)
print("(Attempted to stake ATOM via placeholder—verify API support)")
except Exception as e:
print(f"Failed profit sink buy ATOM: {e}")




def run_once():
init_db()
kr = KrakenClient()
new_syms = ensure_seen_markets(kr)
if new_syms:
buy_new_listings(kr, new_syms)
screen_and_buy_signals(kr)
take_profits_and_sink(kr)


if __name__ == "__main__":
# Option A: single run (use Railway Cron)
run_once()
# Option B: looped runner (unset if using Cron)
# import time
# while True:
# try:
# run_once()
# except Exception as e:
# print("Run error:", e)
# time.sleep(SETTINGS.LOOP_SLEEP_SECONDS)
