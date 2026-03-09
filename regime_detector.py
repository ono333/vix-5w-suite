"""
Regime Detector for VIX 5% Weekly Suite
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from enums import VolatilityRegime

@dataclass
class RegimeState:
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    confidence: float = 0.8
    vix_slope: float = 0.0
    term_structure: str = "normal"
    regime_age_days: int = 0

def _calculate_percentile(data: pd.Series, lookback: int = 52) -> float:
    if len(data) < 2:
        return 0.5
    current = float(data.iloc[-1])
    window = data.iloc[-lookback:] if len(data) >= lookback else data
    return float((window < current).mean())

def _calculate_confidence(data_length: int, vix_percentile: float) -> float:
    base = min(0.9, 0.4 + (data_length / 100) * 0.5)
    boundaries = [0.10, 0.25, 0.50, 0.75, 0.90]
    min_dist = min(abs(vix_percentile - b) for b in boundaries)
    boundary_conf = min(1.0, min_dist * 4)
    return 0.6 * base + 0.4 * boundary_conf

def _fetch_vix_percentile_252() -> float:
    """Fetch ^VIX and return 252-day rank-based percentile."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end   = datetime.now()
        start = end - timedelta(days=400)
        df = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if len(closes) < 10:
            return 0.5
        today = float(closes.iloc[-1])
        window = closes.iloc[-252:] if len(closes) >= 252 else closes
        pct = float((window < today).sum()) / len(window)
        return round(max(0.0, min(1.0, pct)), 4)
    except Exception:
        return 0.5


def classify_regime(
    data: Union[pd.Series, float],
    vix_percentile: Optional[float] = None,
    lookback: int = 252,
) -> RegimeState:
    if isinstance(data, pd.Series) and len(data) > 0:
        vix_level = float(data.iloc[-1])
        # Always use ^VIX rank unless caller explicitly provides percentile
        pct = vix_percentile if vix_percentile is not None else _fetch_vix_percentile_252()
        conf = _calculate_confidence(len(data), pct)
    else:
        vix_level = float(data) if isinstance(data, (int, float)) else 20.0
        pct = vix_percentile if vix_percentile is not None else _fetch_vix_percentile_252()
        conf = 0.5

    # Regime thresholds based on VIX percentile rank
    if pct <= 0.35:
        regime = VolatilityRegime.CALM       # 0–35%: income harvesting
    elif pct <= 0.55:
        regime = VolatilityRegime.RISING     # 35–55%: balanced
    elif pct <= 0.75:
        regime = VolatilityRegime.STRESSED   # 55–75%: defensive
    elif pct <= 0.90:
        regime = VolatilityRegime.DECLINING  # 75–90%: expansion/crisis
    else:
        regime = VolatilityRegime.EXTREME    # 90–100%: panic/tail

    return RegimeState(
        regime=regime,
        vix_level=vix_level,
        vix_percentile=pct,
        confidence=conf,
        vix_slope=0.0,
        term_structure="normal",
        regime_age_days=0,
    )

def get_regime_color(regime: VolatilityRegime) -> str:
    colors = {
        VolatilityRegime.CALM: "#4CAF50",
        VolatilityRegime.RISING: "#FF9800",
        VolatilityRegime.STRESSED: "#f44336",
        VolatilityRegime.DECLINING: "#2196F3",
        VolatilityRegime.EXTREME: "#9C27B0",
    }
    return colors.get(regime, "#757575")

def get_regime_description(regime: VolatilityRegime) -> str:
    descriptions = {
        VolatilityRegime.CALM: "Low volatility - ideal for income harvesting",
        VolatilityRegime.RISING: "Volatility increasing - caution advised",
        VolatilityRegime.STRESSED: "High volatility - defensive positioning",
        VolatilityRegime.DECLINING: "Volatility decreasing - mean reversion opportunity",
        VolatilityRegime.EXTREME: "Crisis levels - tail risk protection",
    }
    return descriptions.get(regime, "Unknown regime")
