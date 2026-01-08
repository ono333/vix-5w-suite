#!/usr/bin/env python3
"""
Robustness Scorer for VIX/UVXY Suite

Calculates execution survivability scores for variant signals.

The robustness score is NOT a performance metric.
It answers: "If this signal is executed later, worse, or not at all - does it still make sense?"

Components:
- Bid-ask spread tightness (slippage tolerance)
- Credit envelope width (entry price sensitivity)
- Liquidity (OI, volume) - fill probability
- Time-to-expiry buffer (delayed execution survivability)
- Structure sensitivity (vertical vs diagonal)
- Regime stability (likelihood of regime flip)

Each component is normalized 0-1 and weighted to produce:
    robustness_score ∈ [0, 100]

Display interpretation:
- Green ≥ 70: Execution-friendly
- Yellow 40-69: Caution
- Red < 40: Theoretical only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from regime_detector import RegimeState, VolatilityRegime
from variant_generator import VariantParams, VariantRole


@dataclass
class RobustnessComponents:
    """Individual components of the robustness score."""
    spread_score: float = 0.0       # Bid-ask spread tightness
    envelope_score: float = 0.0     # Credit envelope width
    liquidity_score: float = 0.0    # OI and volume
    time_buffer_score: float = 0.0  # DTE buffer for delayed execution
    structure_score: float = 0.0    # Structure complexity
    regime_score: float = 0.0       # Regime stability
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spread_score": self.spread_score,
            "envelope_score": self.envelope_score,
            "liquidity_score": self.liquidity_score,
            "time_buffer_score": self.time_buffer_score,
            "structure_score": self.structure_score,
            "regime_score": self.regime_score,
        }


@dataclass
class RobustnessResult:
    """Complete robustness assessment."""
    total_score: float              # 0-100
    components: RobustnessComponents
    weakest_component: str
    strongest_component: str
    recommendation: str             # Human-readable guidance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "components": self.components.to_dict(),
            "weakest_component": self.weakest_component,
            "strongest_component": self.strongest_component,
            "recommendation": self.recommendation,
        }


# Component weights (should sum to 1.0)
WEIGHTS = {
    "spread": 0.20,
    "envelope": 0.15,
    "liquidity": 0.25,
    "time_buffer": 0.15,
    "structure": 0.10,
    "regime": 0.15,
}


def score_spread(
    bid: float,
    ask: float,
    threshold_pct: float = 0.20,
) -> float:
    """
    Score bid-ask spread tightness.
    
    Tighter spread = higher score.
    Spread > threshold_pct = 0 score.
    """
    if bid <= 0 or ask <= bid:
        return 0.0
    
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    
    if spread_pct >= threshold_pct:
        return 0.0
    
    # Linear scaling: 0% spread = 1.0, threshold = 0.0
    return 1.0 - (spread_pct / threshold_pct)


def score_credit_envelope(
    credit_low: float,
    credit_high: float,
    min_acceptable: float = 0.20,
) -> float:
    """
    Score credit envelope width.
    
    Wider envelope with good minimum = higher score.
    """
    if credit_low <= 0:
        return 0.0
    
    if credit_low < min_acceptable:
        # Below minimum acceptable credit
        return 0.3 * (credit_low / min_acceptable)
    
    # Envelope width relative to mid
    mid = (credit_low + credit_high) / 2
    if mid <= 0:
        return 0.0
    
    width = credit_high - credit_low
    width_pct = width / mid
    
    # Narrower envelope = higher score (less sensitive to fill)
    # But also need decent credit level
    credit_score = min(1.0, credit_low / (min_acceptable * 2))
    width_score = max(0.0, 1.0 - width_pct)
    
    return 0.6 * credit_score + 0.4 * width_score


def score_liquidity(
    open_interest: int,
    volume: int,
    min_oi: int = 100,
    min_vol: int = 50,
) -> float:
    """
    Score liquidity based on OI and volume.
    
    Higher OI and volume = higher score.
    """
    oi_score = min(1.0, open_interest / (min_oi * 5)) if open_interest >= min_oi else 0.5 * (open_interest / min_oi)
    vol_score = min(1.0, volume / (min_vol * 5)) if volume >= min_vol else 0.5 * (volume / min_vol) if min_vol > 0 else 0.5
    
    return 0.6 * oi_score + 0.4 * vol_score


def score_time_buffer(
    dte_days: int,
    execution_delay_days: int = 3,
    min_buffer_days: int = 7,
) -> float:
    """
    Score time buffer for delayed execution.
    
    Longer DTE = higher score (more buffer for delayed execution).
    """
    effective_dte = dte_days - execution_delay_days
    
    if effective_dte <= 0:
        return 0.0
    
    if effective_dte < min_buffer_days:
        return 0.5 * (effective_dte / min_buffer_days)
    
    # Diminishing returns after 30 days
    return min(1.0, 0.5 + 0.5 * (effective_dte / 30))


def score_structure(structure: str, role: VariantRole) -> float:
    """
    Score structure complexity and execution difficulty.
    
    Simpler structures = higher score.
    """
    # Base structure scores
    structure_scores = {
        "long_only": 0.95,      # Simplest
        "calendar": 0.80,
        "diagonal": 0.70,
        "credit_spread": 0.60,  # Most sensitive to entry price
        "meta": 1.0,            # V5 (no trade)
    }
    
    base = structure_scores.get(structure, 0.5)
    
    # Role-specific adjustments
    if role == VariantRole.CONVEX:
        # Convex positions are inherently less sensitive to entry timing
        base = min(1.0, base + 0.1)
    elif role == VariantRole.INCOME:
        # Income positions need precise entry
        base = max(0.0, base - 0.1)
    
    return base


def score_regime_stability(
    regime: RegimeState,
    variant_active_regimes: List[VolatilityRegime],
) -> float:
    """
    Score regime stability and alignment with variant.
    
    Higher score if:
    - High regime confidence
    - Current regime matches variant's preferred regimes
    - Low probability of regime flip
    """
    # Base: regime confidence
    base = regime.confidence
    
    # Alignment with variant's preferred regimes
    if regime.regime in variant_active_regimes:
        alignment = 1.0
    else:
        alignment = 0.3
    
    # Stability indicator (low slope = more stable)
    slope_abs = abs(regime.vix_slope)
    stability = max(0.0, 1.0 - slope_abs * 5)  # Penalize high slope
    
    return 0.4 * base + 0.3 * alignment + 0.3 * stability


def calculate_robustness(
    variant: VariantParams,
    regime: RegimeState,
    market_data: Optional[Dict[str, Any]] = None,
) -> RobustnessResult:
    """
    Calculate comprehensive robustness score for a variant.
    
    Parameters
    ----------
    variant : VariantParams
        The variant to score
    regime : RegimeState
        Current regime state
    market_data : Optional[Dict]
        Market data including:
        - bid, ask: Option prices
        - open_interest, volume: Liquidity metrics
        - dte_days: Days to expiration
        
    Returns
    -------
    RobustnessResult
        Complete robustness assessment
    """
    # Default market data if not provided
    if market_data is None:
        market_data = {
            "bid": 0.50,
            "ask": 0.60,
            "open_interest": 200,
            "volume": 50,
            "dte_days": variant.long_dte_weeks * 7,
        }
    
    components = RobustnessComponents()
    
    # Calculate each component
    components.spread_score = score_spread(
        market_data.get("bid", 0),
        market_data.get("ask", 0),
    )
    
    # Estimate credit envelope from bid/ask
    bid = market_data.get("bid", 0)
    ask = market_data.get("ask", 0)
    components.envelope_score = score_credit_envelope(
        credit_low=bid,
        credit_high=bid + 0.35 * (ask - bid),  # Conservative fill estimate
    )
    
    components.liquidity_score = score_liquidity(
        market_data.get("open_interest", 0),
        market_data.get("volume", 0),
    )
    
    components.time_buffer_score = score_time_buffer(
        market_data.get("dte_days", variant.long_dte_weeks * 7),
    )
    
    components.structure_score = score_structure(
        variant.structure,
        variant.role,
    )
    
    components.regime_score = score_regime_stability(
        regime,
        variant.active_in_regimes,
    )
    
    # Calculate weighted total
    total = (
        WEIGHTS["spread"] * components.spread_score +
        WEIGHTS["envelope"] * components.envelope_score +
        WEIGHTS["liquidity"] * components.liquidity_score +
        WEIGHTS["time_buffer"] * components.time_buffer_score +
        WEIGHTS["structure"] * components.structure_score +
        WEIGHTS["regime"] * components.regime_score
    ) * 100
    
    # Find weakest and strongest components
    component_scores = {
        "spread": components.spread_score,
        "envelope": components.envelope_score,
        "liquidity": components.liquidity_score,
        "time_buffer": components.time_buffer_score,
        "structure": components.structure_score,
        "regime": components.regime_score,
    }
    
    weakest = min(component_scores, key=component_scores.get)
    strongest = max(component_scores, key=component_scores.get)
    
    # Generate recommendation
    if total >= 70:
        recommendation = f"Execution-friendly. Strength: {strongest}."
    elif total >= 40:
        recommendation = f"Proceed with caution. Weakness: {weakest}."
    else:
        recommendation = f"Theoretical only. Critical weakness: {weakest}."
    
    return RobustnessResult(
        total_score=total,
        components=components,
        weakest_component=weakest,
        strongest_component=strongest,
        recommendation=recommendation,
    )


def get_robustness_color(score: float) -> str:
    """Get display color for robustness score."""
    if score >= 70:
        return "#28a745"  # Green
    elif score >= 40:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_robustness_label(score: float) -> str:
    """Get label for robustness score."""
    if score >= 70:
        return "EXECUTION-FRIENDLY"
    elif score >= 40:
        return "CAUTION"
    else:
        return "THEORETICAL ONLY"


def batch_score_variants(
    variants: List[VariantParams],
    regime: RegimeState,
    market_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, RobustnessResult]:
    """
    Score multiple variants at once.
    
    Parameters
    ----------
    variants : List[VariantParams]
    regime : RegimeState
    market_data : Optional[Dict]
        Dict mapping variant_id to market data
    
    Returns
    -------
    Dict[str, RobustnessResult]
        Mapping of variant_id to robustness result
    """
    results = {}
    
    for variant in variants:
        variant_market_data = None
        if market_data and variant.variant_id in market_data:
            variant_market_data = market_data[variant.variant_id]
        
        result = calculate_robustness(variant, regime, variant_market_data)
        results[variant.variant_id] = result
        
        # Update variant's robustness score
        variant.robustness_score = result.total_score
    
    return results
