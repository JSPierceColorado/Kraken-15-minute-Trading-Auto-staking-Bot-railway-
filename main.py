# =============================
# main.py
# =============================
# Orchestrates screening, trading, selling, and profit sink.

from contextlib import contextmanager
from settings import SETTINGS
from db import init_db, SessionLocal, SeenMarket, Position, BuyLock, TradeLog
from exchange_ccxt import KrakenClient
from strategy import compute_indicators, buy_signal

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


# --- NEW: bootstrap + new-market detection -----------------------------------
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
# -----------------------------------------------------------------------------


def buy_new_listings(kr: KrakenClient, new_symbols):
    for sym, base, quote in new_symbols:
        try:
            price, qty = kr.market_buy_usd(sym, SETTINGS.MIN_NOTIONAL_USD)
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
            print(f"New listing buy: {sym} ${SETTINGS.MIN_NOTIONAL_USD:.2f} at {price:.6f}")
        except Exception as e:
            print(f"Failed new-listing buy for {sym}: {e}")


def screen_and_buy_signals(kr: KrakenClient):
    syms = kr.list_screenable_symbols()
    for sym, base, quote in syms:
        # Only one active buy per base asset at a time
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
                price, qty = kr.market_buy_usd(sym, SETTINGS.MIN_NOTIONAL_USD)
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
                print(f"Signal BUY {sym}: rsi={last['rsi']:.1f} ma60<{last['ma_slow']:.6f}? at {price:.6f}")
        except Exception as e:
            print(f"Screening error {sym}: {e}")


def take_profits_and_sink(kr: KrakenClient):
    total_profit_usd = 0.0
    with session_scope() as s:
        positions = s.query(Position).all()
        for pos in positions:
            # Never sell ATOM (profit sink asset)
            if pos.base == SETTINGS.PROFIT_SINK_SYMBOL:
                continue
            sym = f"{pos.base}/{pos.quote}"
            try:
                price = kr.fetch_ticker_price(sym)
            except Exception:
                continue
            if price >= pos.avg_cost * (1.0 + SETTINGS.TAKE_PROFIT_PCT):
                sell_price, sold_qty = kr.market_sell_all(sym, pos.amount)
                notional = sell_price * sold_qty
                cost = pos.avg_cost * sold_qty
                pnl = notional - cost
                if pos.quote in ("USD", "USDT"):
                    total_profit_usd += pnl
                s.add(TradeLog(side="SELL", symbol=sym, base=pos.base, quote=pos.quote,
                               price=sell_price, amount=sold_qty,
                               notional=notional, pnl=pnl))
                # Clear position and its buy-lock
                s.delete(pos)
                bl = s.query(BuyLock).filter_by(base=pos.base, active=True).first()
                if bl:
                    s.delete(bl)
                print(f"TP SELL {sym}: sold {sold_qty:.8f} at {sell_price:.6f} pnl≈{pnl:.2f}")

    # Sink realized USD profits into ATOM
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
    init_db()
    kr = KrakenClient()

    # First-ever run: seed the market list and do NOT buy.
    bootstrapped = seed_markets_if_empty(kr)

    # On subsequent runs: only these are truly new listings.
    new_syms = find_new_markets(kr)

    if (not bootstrapped) and new_syms:
        buy_new_listings(kr, new_syms)

    screen_and_buy_signals(kr)
    take_profits_and_sink(kr)


if __name__ == "__main__":
    run_once()
    # Uncomment for continuous loop mode:
    # import time
    # while True:
    #     try:
    #         run_once()
    #     except Exception as e:
    #         print("Run error:", e)
    #     time.sleep(SETTINGS.LOOP_SLEEP_SECONDS)
