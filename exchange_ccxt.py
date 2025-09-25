# =============================
# exchange_ccxt.py
# =============================
# Kraken wrapper with CCXT for tickers, candles, orders.

import ccxt
import pandas as pd
from typing import List, Dict, Tuple
from settings import SETTINGS

class KrakenClient:
    def __init__(self):
        self.x = ccxt.kraken({
            'apiKey': SETTINGS.KRAKEN_API_KEY,
            'secret': SETTINGS.KRAKEN_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'fetchOHLCVWarning': False,
            }
        })

    def load_markets(self) -> Dict:
        return self.x.load_markets(reload=True)

    def _is_xstock(self, base: str) -> bool:
        if not base:
            return False
        # Treat bases ending with the configured suffix (e.g., "x") as xStocks
        if base.lower().endswith(SETTINGS.XSTOCKS_SUFFIX.lower()):
            return True
        # Or bases explicitly whitelisted via env
        if base.upper() in SETTINGS.XSTOCKS_BASES:
            return True
        return False

    def list_screenable_symbols(self) -> List[Tuple[str, str, str]]:
        mkts = self.load_markets()
        out = []
        for sym, m in mkts.items():
            base = m.get('base')
            quote = m.get('quote')
            if not base or not quote:
                continue
            if quote not in SETTINGS.QUOTE_ASSETS:
                continue
            if not m.get('active', True):
                continue

            is_x = self._is_xstock(base)

            # xStocks-only mode
            if SETTINGS.XSTOCKS_ONLY:
                if SETTINGS.INCLUDE_XSTOCKS and is_x:
                    out.append((sym, base, quote))
                continue

            # Mixed mode with toggles
            if is_x and SETTINGS.INCLUDE_XSTOCKS:
                out.append((sym, base, quote))
            elif (not is_x) and SETTINGS.INCLUDE_CRYPTO:
                out.append((sym, base, quote))
        return out

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_ticker_price(self, symbol: str) -> float:
        t = self.x.fetch_ticker(symbol)
        return float(t['last'])

    def market_buy_usd(self, symbol: str, usd_amount: float) -> Tuple[float, float]:
        # market buy spending usd_amount of QUOTE
        price = self.fetch_ticker_price(symbol)
        base_qty = usd_amount / price if price > 0 else 0.0
        if SETTINGS.DRY_RUN:
            return price, base_qty
        # Kraken CCXT: amount is base currency units
        self.x.create_order(symbol, type='market', side='buy', amount=base_qty)
        return price, base_qty

    def market_sell_all(self, symbol: str, base_qty: float) -> Tuple[float, float]:
        price = self.fetch_ticker_price(symbol)
        if SETTINGS.DRY_RUN:
            return price, base_qty
        self.x.create_order(symbol, type='market', side='sell', amount=base_qty)
        return price, base_qty

    def buy_atom_for_usd(self, usd_amount: float) -> Tuple[float, float]:
        # Prefer USD quote if available, else USDT
        for q in SETTINGS.QUOTE_ASSETS:
            sym = f"{SETTINGS.PROFIT_SINK_SYMBOL}/{q}"
            try:
                return self.market_buy_usd(sym, usd_amount)
            except Exception:
                continue
        raise RuntimeError("No ATOM/* market available for quotes: " + ",".join(SETTINGS.QUOTE_ASSETS))

    def stake_atom(self, base_qty: float) -> None:
        # Placeholder: Kraken Earn/Equities staking endpoints aren’t available via CCXT.
        # If/when Kraken exposes this, wire it here.
        pass
