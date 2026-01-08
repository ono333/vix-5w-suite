"""
Regime Detector for VIX 5% Weekly Suite

Exports:
    - classify_regime
    - RegimeState
    - VolatilityRegime (re-exported from enums)
    - get_regime_color
    - get_regime_description
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Re-export VolatilityRegime for convenience
from enums import VolatilityRegime

# Regime percentile boundaries
REGIME_THRESHOLDS = {
    VolatilityRegime.CALM: (0.00, 0.25),
    VolatilityRegime.RISING: (0.25, 0.50),
    VolatilityRegime.STRESSED: (0.50, 0.75),
    VolatilityRegime.DECLINING: (0.75, 0.90),
    VolatilityRegime.EXTREME: (0.90, 1.01),
}


@dataclass
class RegimeState:
    """Current regime state."""
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    timestamp: dt.datetime
    previous_regime: Optional[VolatilityRegime] = None
    regime_duration_days: int = 0
    is_transition: bool = False
    vix_change_1w: float = 0.0
    vix_change_1m: float = 0.0


def classify_regime(
    vix_level: float,
    vix_percentile: float,
    vix_series: Optional[pd.Series] = None,
    previous_state: Optional[RegimeState] = None,
) -> RegimeState:
    """
    Classify current volatility regime.
    
    Args:
        vix_level: Current VIX level
        vix_percentile: Current VIX percentile (0-1)
        vix_series: Optional historical VIX data for change calculations
        previous_state: Previous regime state for transition detection
    
    Returns:
        RegimeState with current regime info
    """
    # Determine regime from percentile
    regime = VolatilityRegime.CALM  # default
    for r, (low, high) in REGIME_THRESHOLDS.items():
        if low <= vix_percentile < high:
            regime = r
            break
    
    # Calculate VIX changes if series provided
    vix_change_1w = 0.0
    vix_change_1m = 0.0
    if vix_series is not None and len(vix_series) > 0:
        if len(vix_series) >= 1:
            vix_1w = vix_series.iloc[-1]
            vix_change_1w = (vix_level - vix_1w) / vix_1w if vix_1w > 0 else 0.0
        if len(vix_series) >= 4:
            vix_1m = vix_series.iloc[-4]
            vix_change_1m = (vix_level - vix_1m) / vix_1m if vix_1m > 0 else 0.0
    
    # Check for regime transition
    is_transition = False
    previous_regime = None
    regime_duration_days = 0
    
    if previous_state:
        previous_regime = previous_state.regime
        is_transition = (regime != previous_regime)
        if not is_transition:
            regime_duration_days = previous_state.regime_duration_days + 1
    
    return RegimeState(
        regime=regime,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        timestamp=dt.datetime.now(),
        previous_regime=previous_regime,
        regime_duration_days=regime_duration_days,
        is_transition=is_transition,
        vix_change_1w=vix_change_1w,
        vix_change_1m=vix_change_1m,
    )


def get_regime_color(regime: VolatilityRegime) -> str:
    """Get display color for a regime."""
    colors = {
        VolatilityRegime.CALM: "#2ECC71",      # Green
        VolatilityRegime.RISING: "#F1C40F",    # Yellow
        VolatilityRegime.STRESSED: "#E67E22",  # Orange
        VolatilityRegime.DECLINING: "#3498DB", # Blue
        VolatilityRegime.EXTREME: "#E74C3C",   # Red
    }
    return colors.get(regime, "#95A5A6")


def get_regime_description(regime: VolatilityRegime) -> str:
    """Get description for a regime."""
    descriptions = {
        VolatilityRegime.CALM: "Low volatility - ideal for income harvesting",
        VolatilityRegime.RISING: "Volatility increasing - reduce exposure",
        VolatilityRegime.STRESSED: "Elevated volatility - hedge activation",
        VolatilityRegime.DECLINING: "Post-spike decay - mean reversion opportunities",
        VolatilityRegime.EXTREME: "Crisis conditions - tail strategies active",
    }
    return descriptions.get(regime, "Unknown regime")


def get_regime_emoji(regime: VolatilityRegime) -> str:
    """Get emoji for a regime."""
    emojis = {
        VolatilityRegime.CALM: "🟢",
        VolatilityRegime.RISING: "🟡",
        VolatilityRegime.STRESSED: "🟠",
        VolatilityRegime.DECLINING: "🔵",
        VolatilityRegime.EXTREME: "🔴",
    }
    return emojis.get(regime, "⚪")


def calculate_vix_percentile(
    vix_series: pd.Series,
    current_vix: float,
    lookback_weeks: int = 52,
) -> float:
    """Calculate VIX percentile within lookback window."""
    if vix_series is None or len(vix_series) < lookback_weeks:
        # Fallback heuristics
        if current_vix < 15:
            return 0.15
        elif current_vix < 20:
            return 0.35
        elif current_vix < 25:
            return 0.55
        elif current_vix < 35:
            return 0.75
        else:
            return 0.95
    
    window = vix_series.iloc[-lookback_weeks:].values
    return float((window < current_vix).mean())
