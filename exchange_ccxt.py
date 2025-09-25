import time
from typing import List, Dict, Tuple
import ccxt
import pandas as pd
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


def list_screenable_symbols(self) -> List[Tuple[str,str,str]]:
mkts = self.load_markets()
out = []
for sym, m in mkts.items():
base = m['base']
quote = m['quote']
if quote in SETTINGS.QUOTE_ASSETS and m.get('active', True):
out.append((sym, base, quote))
return out


def fetch_ohlcv_df(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
return df


def fetch_ticker_price(self, symbol: str) -> float:
t = self.x.fetch_ticker(symbol)
return float(t['last'])


def market_buy_usd(self, symbol: str, usd_amount: float) -> Tuple[float,float]:
# market buy spending usd_amount of QUOTE
price = self.fetch_ticker_price(symbol)
base_qty = usd_amount / price
if SETTINGS.DRY_RUN:
return price, base_qty
order = self.x.create_order(symbol, type='market', side='buy', amount=base_qty)
return price, base_qty


def market_sell_all(self, symbol: str, base_qty: float) -> Tuple[float,float]:
price = self.fetch_ticker_price(symbol)
if SETTINGS.DRY_RUN:
return price, base_qty
order = self.x.create_order(symbol, type='market', side='sell', amount=base_qty)
return price, base_qty


def buy_atom_for_usd(self, usd_amount: float) -> Tuple[float,float]:
# Prefer USD quote if available, else USDT
for q in SETTINGS.QUOTE_ASSETS:
sym = f"{SETTINGS.PROFIT_SINK_SYMBOL}/{q}"
try:
return self.market_buy_usd(sym, usd_amount)
except Exception:
continue
raise RuntimeError("No ATOM/* market available for quotes: " + ",".join(SETTINGS.QUOTE_ASSETS))


def stake_atom(self, base_qty: float) -> None:
# Placeholder: Kraken Earn API may not be available via CCXT. Keep here for future integration.
# In practice: use Kraken's native API endpoint if/when accessible, else leave ATOM on spot and stake manually.
pass
