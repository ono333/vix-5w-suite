#!/usr/bin/env python3
"""
Regime Detector for VIX/UVXY Suite

Classifies the current volatility environment into distinct regimes:
- CALM: Low volatility, stable markets
- RISING: Volatility increasing, potential stress ahead
- STRESSED: High volatility, crisis conditions
- DECLINING: Post-spike, volatility normalizing
- EXTREME: Tail event conditions

Each regime has specific implications for variant activation and parameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    CALM = "calm"
    RISING = "rising"
    STRESSED = "stressed"
    DECLINING = "declining"
    EXTREME = "extreme"


@dataclass
class RegimeState:
    """Current regime state with supporting metrics."""
    regime: VolatilityRegime
    confidence: float  # 0-1, how confident in this classification
    vix_level: float
    vix_percentile: float  # 0-1, percentile over lookback
    vix_slope: float  # Rate of change
    term_structure: float  # Contango/backwardation indicator
    regime_age_days: int  # How long in this regime
    previous_regime: Optional[VolatilityRegime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "vix_slope": self.vix_slope,
            "term_structure": self.term_structure,
            "regime_age_days": self.regime_age_days,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
        }


def compute_percentile(series: pd.Series, lookback: int = 52) -> float:
    """Compute current value's percentile over lookback period."""
    if len(series) < lookback:
        lookback = len(series)
    if lookback < 2:
        return 0.5
    
    window = series.iloc[-lookback:]
    current = series.iloc[-1]
    return float((window < current).mean())


def compute_slope(series: pd.Series, window: int = 5) -> float:
    """Compute slope (rate of change) over recent period."""
    if len(series) < window + 1:
        return 0.0
    
    recent = series.iloc[-window:]
    if recent.iloc[0] == 0:
        return 0.0
    
    return float((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0])


def compute_acceleration(series: pd.Series, window: int = 5) -> float:
    """Compute acceleration (second derivative) of volatility."""
    if len(series) < window * 2:
        return 0.0
    
    slope_recent = compute_slope(series, window)
    slope_prior = compute_slope(series.iloc[:-window], window)
    
    return slope_recent - slope_prior


def classify_regime(
    vix_series: pd.Series,
    lookback_weeks: int = 52,
    slope_window: int = 5,
    previous_regime: Optional[VolatilityRegime] = None,
) -> RegimeState:
    """
    Classify current volatility regime based on VIX/UVXY series.
    
    Parameters
    ----------
    vix_series : pd.Series
        Weekly VIX or UVXY close prices
    lookback_weeks : int
        Lookback period for percentile calculation
    slope_window : int
        Window for slope calculation
    previous_regime : Optional[VolatilityRegime]
        Previous regime for transition detection
    
    Returns
    -------
    RegimeState
        Current regime classification with supporting metrics
    """
    if vix_series is None or len(vix_series) < 10:
        return RegimeState(
            regime=VolatilityRegime.CALM,
            confidence=0.0,
            vix_level=0.0,
            vix_percentile=0.5,
            vix_slope=0.0,
            term_structure=0.0,
            regime_age_days=0,
            previous_regime=previous_regime,
        )
    
    current_level = float(vix_series.iloc[-1])
    percentile = compute_percentile(vix_series, lookback_weeks)
    slope = compute_slope(vix_series, slope_window)
    acceleration = compute_acceleration(vix_series, slope_window)
    
    # Term structure approximation (would need VIX futures for real calculation)
    # Using short-term vs longer-term volatility as proxy
    if len(vix_series) >= 20:
        short_term_avg = vix_series.iloc[-5:].mean()
        longer_term_avg = vix_series.iloc[-20:-5].mean()
        if longer_term_avg > 0:
            term_structure = float((short_term_avg - longer_term_avg) / longer_term_avg)
        else:
            term_structure = 0.0
    else:
        term_structure = 0.0
    
    # Classification logic
    regime = VolatilityRegime.CALM
    confidence = 0.5
    
    # EXTREME: Very high percentile with acceleration
    if percentile >= 0.95 and slope > 0.15:
        regime = VolatilityRegime.EXTREME
        confidence = min(0.95, percentile)
    
    # STRESSED: High percentile or sharp increase
    elif percentile >= 0.75 or (percentile >= 0.60 and slope > 0.10):
        regime = VolatilityRegime.STRESSED
        confidence = 0.7 + 0.2 * percentile
    
    # RISING: Increasing volatility from low/medium levels
    elif slope > 0.05 and percentile < 0.75:
        regime = VolatilityRegime.RISING
        confidence = 0.6 + 0.3 * min(slope / 0.15, 1.0)
    
    # DECLINING: Falling volatility from high levels
    elif slope < -0.03 and percentile > 0.40:
        regime = VolatilityRegime.DECLINING
        confidence = 0.6 + 0.3 * min(abs(slope) / 0.10, 1.0)
    
    # CALM: Low volatility, stable
    else:
        regime = VolatilityRegime.CALM
        confidence = 0.7 + 0.2 * (1 - percentile)
    
    # Estimate regime age (simplified)
    regime_age_days = 7  # Default to 1 week
    
    return RegimeState(
        regime=regime,
        confidence=confidence,
        vix_level=current_level,
        vix_percentile=percentile,
        vix_slope=slope,
        term_structure=term_structure,
        regime_age_days=regime_age_days,
        previous_regime=previous_regime,
    )


def regime_transition_alert(
    current: RegimeState,
    previous: RegimeState,
) -> Optional[str]:
    """
    Generate alert message for significant regime transitions.
    
    Returns None if no alert needed.
    """
    if current.regime == previous.regime:
        return None
    
    # Define critical transitions
    critical_transitions = {
        (VolatilityRegime.CALM, VolatilityRegime.RISING): "⚠️ ALERT: Volatility regime shifting from CALM to RISING",
        (VolatilityRegime.RISING, VolatilityRegime.STRESSED): "🚨 ALERT: Entering STRESSED regime - review positions",
        (VolatilityRegime.STRESSED, VolatilityRegime.EXTREME): "🔴 CRITICAL: EXTREME volatility conditions",
        (VolatilityRegime.STRESSED, VolatilityRegime.DECLINING): "📉 Volatility peaked - entering DECLINING regime",
        (VolatilityRegime.DECLINING, VolatilityRegime.CALM): "✅ Volatility normalized - CALM regime restored",
    }
    
    key = (previous.regime, current.regime)
    if key in critical_transitions:
        return critical_transitions[key]
    
    return f"ℹ️ Regime changed: {previous.regime.value} → {current.regime.value}"


def get_regime_color(regime: VolatilityRegime) -> str:
    """Return display color for regime."""
    colors = {
        VolatilityRegime.CALM: "#28a745",      # Green
        VolatilityRegime.RISING: "#ffc107",    # Yellow
        VolatilityRegime.STRESSED: "#fd7e14",  # Orange
        VolatilityRegime.DECLINING: "#17a2b8", # Cyan
        VolatilityRegime.EXTREME: "#dc3545",   # Red
    }
    return colors.get(regime, "#6c757d")


def get_regime_description(regime: VolatilityRegime) -> str:
    """Return human-readable description of regime."""
    descriptions = {
        VolatilityRegime.CALM: "Low volatility, stable markets. Favor income/carry strategies.",
        VolatilityRegime.RISING: "Volatility increasing. Reduce short exposure, prepare hedges.",
        VolatilityRegime.STRESSED: "High volatility. Focus on hedges and convex positions.",
        VolatilityRegime.DECLINING: "Post-spike decay. Favor mean reversion strategies.",
        VolatilityRegime.EXTREME: "Tail event conditions. Maximize convex exposure.",
    }
    return descriptions.get(regime, "Unknown regime")
