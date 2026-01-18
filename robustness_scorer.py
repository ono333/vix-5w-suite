"""
Robustness Scorer for VIX 5% Weekly Suite

Calculates a robustness score for variant signals based on multiple factors.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class RobustnessResult:
    """Result of robustness calculation."""
    total_score: float = 50.0
    regime_score: float = 50.0
    percentile_score: float = 50.0
    timing_score: float = 50.0
    structure_score: float = 50.0
    notes: str = ""


def calculate_robustness(variant: Any, regime: Any) -> RobustnessResult:
    """
    Calculate robustness score for a variant given current regime.
    
    Factors considered:
    - Regime alignment (is variant active in current regime?)
    - Percentile position (how close to optimal entry?)
    - Timing (day of week, market hours)
    - Structure (spread width, DTE balance)
    """
    score = 50.0  # Base score
    
    # Regime alignment
    regime_score = 0.0
    if hasattr(variant, 'active_in_regimes') and hasattr(regime, 'regime'):
        if regime.regime in variant.active_in_regimes:
            regime_score = 80.0
        else:
            regime_score = 20.0
    else:
        regime_score = 50.0
    
    # Percentile score - better when lower percentile for most strategies
    percentile_score = 50.0
    if hasattr(regime, 'vix_percentile') and hasattr(variant, 'entry_percentile'):
        if regime.vix_percentile <= variant.entry_percentile:
            # Good - we're at or below entry threshold
            percentile_score = 80.0 + (variant.entry_percentile - regime.vix_percentile) * 20
        else:
            # Not ideal - above entry threshold
            percentile_score = max(20.0, 60.0 - (regime.vix_percentile - variant.entry_percentile) * 50)
    
    # Confidence score
    confidence_score = 50.0
    if hasattr(regime, 'confidence'):
        confidence_score = regime.confidence * 100
    
    # Timing score (simplified)
    timing_score = 60.0  # Base timing score
    
    # Calculate total
    total_score = (
        regime_score * 0.35 +
        percentile_score * 0.30 +
        confidence_score * 0.20 +
        timing_score * 0.15
    )
    
    return RobustnessResult(
        total_score=min(100, max(0, total_score)),
        regime_score=regime_score,
        percentile_score=percentile_score,
        timing_score=timing_score,
        structure_score=50.0,
        notes=""
    )


def batch_score_variants(variants: list, regime: Any) -> list:
    """Score multiple variants at once."""
    return [calculate_robustness(v, regime) for v in variants]


def get_robustness_color(score: float) -> str:
    """Get color for robustness score display."""
    if score >= 80:
        return "#4CAF50"  # Green
    elif score >= 60:
        return "#8BC34A"  # Light green
    elif score >= 40:
        return "#FFC107"  # Amber
    elif score >= 20:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def get_robustness_label(score: float) -> str:
    """Get label for robustness score."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    elif score >= 20:
        return "Weak"
    else:
        return "Poor"
