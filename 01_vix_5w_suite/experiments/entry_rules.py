import numpy as np
import pandas as pd

def entry_signal_percentile(i: int, prices: np.ndarray, pct_array: np.ndarray, threshold: float) -> bool:
    if i < 5:
        return False
    pct_i = pct_array[i]
    if np.isnan(pct_i):
        return False
    return pct_i <= threshold

def entry_signal_osc_roc(i: int, macd: pd.Series, roc2: pd.Series) -> bool:
    if i < 2:
        return False
    m = macd.iloc[i]
    m1 = macd.iloc[i - 1]
    m2 = macd.iloc[i - 2]
    r2 = roc2.iloc[i]
    if any(pd.isna(x) for x in (m, m1, m2, r2)):
        return False
    return (m < 0.0) and (m > m1) and (m1 <= m2) and (r2 <= 0.0)
