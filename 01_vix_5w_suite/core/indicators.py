import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_3_10(series: pd.Series) -> pd.Series:
    fast = ema(series, 3)
    slow = ema(series, 10)
    return fast - slow

def roc(series: pd.Series, period: int = 2) -> pd.Series:
    return (series / series.shift(period) - 1.0) * 100.0

def rolling_percentile(series: pd.Series, window: int = 52) -> pd.Series:
    r = series.rolling(window=window, min_periods=4)
    return r.rank(pct=True)
