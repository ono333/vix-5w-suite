"""
Regime Detection Engine
-----------------------
Computes VIX percentile over a rolling lookback window and maps it
to one of five trading regimes.  Also fetches UVXY / VXX prices and
detects the 1-week VIX trend.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ── Regime definitions ────────────────────────────────────────────────────────

REGIME_BANDS: List[Tuple[float, float, str, str, str]] = [
    (0,  20, "ULTRA_LOW", "🟢", "#00d67c"),
    (20, 40, "LOW",       "🔵", "#3b82f6"),
    (40, 60, "MEDIUM",    "🟡", "#f59e0b"),
    (60, 80, "HIGH",      "🟠", "#f97316"),
    (80, 101,"EXTREME",   "🔴", "#ef4444"),
]

REGIME_DESC = {
    "ULTRA_LOW": "VIX complacency zone. Premium sellers' paradise; decay is rapid but spikes are dangerous.",
    "LOW":       "Below-average volatility. Strong income harvesting environment with moderate cushion.",
    "MEDIUM":    "Average volatility. Balanced regime. Most strategies are viable.",
    "HIGH":      "Elevated volatility. Mean-reversion setups attractive; size down on new entries.",
    "EXTREME":   "Crisis / spike. Shock absorbers and tail hunters activate. Existing positions need triage.",
}


@dataclass
class RegimeState:
    vix_current:    float
    vix_percentile: float
    regime:         str
    regime_color:   str
    regime_emoji:   str
    uvxy_price:     float | None
    vxx_price:      float | None
    vix_1w_ago:     float
    vix_1m_ago:     float
    vix_trend:      str   # RISING | FALLING | STABLE
    vix_1w_chg_pct: float
    vix_series:     pd.Series          # full lookback window for charts
    percentile_series: pd.Series       # rolling percentile for charts
    as_of:          datetime
    lookback_days:  int

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def vix_level(self) -> str:
        """Human-readable VIX level string."""
        v = self.vix_current
        if   v < 14:  return "Historically Low"
        elif v < 20:  return "Below Average"
        elif v < 25:  return "Average"
        elif v < 35:  return "Elevated"
        elif v < 50:  return "Stressed"
        else:          return "Crisis"

    @property
    def trend_arrow(self) -> str:
        if self.vix_trend == "RISING":  return "↑"
        if self.vix_trend == "FALLING": return "↓"
        return "→"

    @property
    def trend_color(self) -> str:
        if self.vix_trend == "RISING":  return "#ef4444"
        if self.vix_trend == "FALLING": return "#22c55e"
        return "#94a3b8"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pct_rank_latest(series: pd.Series) -> float:
    """Return the percentile rank (0-100) of the last observation."""
    arr = series.dropna().values
    if len(arr) < 2:
        return 50.0
    last = arr[-1]
    return float(np.sum(arr <= last) / len(arr) * 100)


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank of `series` using a trailing `window`."""
    def rank_last(s):
        v = s.values
        return np.sum(v <= v[-1]) / len(v) * 100
    return series.rolling(window, min_periods=30).apply(rank_last, raw=False)


def _regime_from_pct(pct: float) -> Tuple[str, str, str]:
    for lo, hi, name, emoji, color in REGIME_BANDS:
        if lo <= pct < hi:
            return name, emoji, color
    return "EXTREME", "🔴", "#ef4444"


# ── Public API ────────────────────────────────────────────────────────────────

_CACHE: dict = {}
_CACHE_TTL = 300  # seconds


def fetch_market_data(lookback_days: int = 252) -> RegimeState:
    """
    Download VIX, UVXY, VXX from yfinance and compute regime state.
    Results are cached for 5 minutes to avoid hammering the API.
    """
    cache_key = lookback_days
    now = time.time()
    if cache_key in _CACHE:
        ts, result = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return result

    extra = 60  # buffer for weekends / holidays
    start = datetime.today() - timedelta(days=lookback_days + extra)
    end   = datetime.today()

    # ── download ──────────────────────────────────────────────────────────
    vix_raw  = yf.download("^VIX",  start=start, end=end, progress=False, auto_adjust=True)
    uvxy_raw = yf.download("UVXY",  start=start, end=end, progress=False, auto_adjust=True)
    vxx_raw  = yf.download("VXX",   start=start, end=end, progress=False, auto_adjust=True)

    # Handle both old (Close) and new (multi-level) yfinance column formats
    def _close(df: pd.DataFrame) -> pd.Series:
        if "Close" in df.columns:
            return df["Close"].dropna().squeeze()
        # Multi-level (ticker, field)
        for col in df.columns:
            if isinstance(col, tuple) and "Close" in col:
                return df[col].dropna().squeeze()
        return pd.Series(dtype=float)

    vix_s  = _close(vix_raw)
    uvxy_s = _close(uvxy_raw)
    vxx_s  = _close(vxx_raw)

    if vix_s.empty:
        raise RuntimeError("Failed to download VIX data from yfinance.")

    # ── compute regime ─────────────────────────────────────────────────────
    vix_window   = vix_s.iloc[-lookback_days:]
    pct          = _pct_rank_latest(vix_window)
    pct_series   = _rolling_percentile(vix_s, lookback_days).iloc[-lookback_days:]

    vix_current  = float(vix_s.iloc[-1])
    vix_1w_ago   = float(vix_s.iloc[-6])  if len(vix_s) > 5  else vix_current
    vix_1m_ago   = float(vix_s.iloc[-22]) if len(vix_s) > 21 else vix_current

    chg_1w_pct   = (vix_current - vix_1w_ago) / vix_1w_ago * 100
    if   chg_1w_pct >  8:  trend = "RISING"
    elif chg_1w_pct < -8:  trend = "FALLING"
    else:                   trend = "STABLE"

    regime, emoji, color = _regime_from_pct(pct)

    state = RegimeState(
        vix_current    = vix_current,
        vix_percentile = pct,
        regime         = regime,
        regime_color   = color,
        regime_emoji   = emoji,
        uvxy_price     = float(uvxy_s.iloc[-1]) if not uvxy_s.empty else None,
        vxx_price      = float(vxx_s.iloc[-1])  if not vxx_s.empty  else None,
        vix_1w_ago     = vix_1w_ago,
        vix_1m_ago     = vix_1m_ago,
        vix_trend      = trend,
        vix_1w_chg_pct = chg_1w_pct,
        vix_series     = vix_window,
        percentile_series = pct_series,
        as_of          = datetime.now(),
        lookback_days  = lookback_days,
    )

    _CACHE[cache_key] = (now, state)
    return state


def get_regime_description(regime: str) -> str:
    return REGIME_DESC.get(regime, "")


def all_regimes() -> List[str]:
    return [r[2] for r in REGIME_BANDS]
