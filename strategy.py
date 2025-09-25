import pandas as pd


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
delta = series.diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.ewm(alpha=1/length, adjust=False).mean()
roll_down = down.ewm(alpha=1/length, adjust=False).mean()
rs = roll_up / (roll_down + 1e-9)
rsi = 100 - (100 / (1 + rs))
return rsi




def moving_average(series: pd.Series, window: int) -> pd.Series:
return series.rolling(window=window, min_periods=window).mean()




def compute_indicators(df: pd.DataFrame, rsi_len: int, fast: int, slow: int) -> pd.DataFrame:
out = df.copy()
out['rsi'] = rsi(out['close'], rsi_len)
out['ma_fast'] = moving_average(out['close'], fast)
out['ma_slow'] = moving_average(out['close'], slow)
return out




def buy_signal(row) -> bool:
return (row['rsi'] <= 30) and (row['ma_fast'] < row['ma_slow'])
