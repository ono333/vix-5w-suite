#!/usr/bin/env python3
"""
Robustness Scoring Module for VIX 5% Weekly Suite

Robustness score is an EXECUTION SURVIVABILITY metric, not a performance metric.

It answers: "If this signal is executed later, worse, or not at all — does it still make sense?"

Components:
    1. Bid-ask tightness (slippage tolerance)
    2. Credit envelope width (entry price sensitivity)
    3. Liquidity (OI / volume)
    4. Time-to-expiry buffer (delayed execution survivability)
    5. Structure sensitivity (vertical vs diagonal vs calendar)
    6. Regime stability (likelihood of regime flip)

Output: robustness_score ∈ [0, 100]
    - Green ≥ 70: Execution-friendly
    - Yellow 40-69: Caution
    - Red < 40: Theoretical only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np

from regime_detector import VolatilityRegime, RegimeState


@dataclass
class RobustnessComponents:
    """Individual components of the robustness score."""
    bid_ask_score: float  # 0-100
    credit_envelope_score: float  # 0-100
    liquidity_score: float  # 0-100
    dte_buffer_score: float  # 0-100
    structure_score: float  # 0-100
    regime_stability_score: float  # 0-100
    
    # Weights (sum to 1.0)
    WEIGHT_BID_ASK = 0.20
    WEIGHT_CREDIT = 0.15
    WEIGHT_LIQUIDITY = 0.20
    WEIGHT_DTE = 0.15
    WEIGHT_STRUCTURE = 0.15
    WEIGHT_REGIME = 0.15
    
    @property
    def total_score(self) -> float:
        """Weighted composite score."""
        return (
            self.bid_ask_score * self.WEIGHT_BID_ASK +
            self.credit_envelope_score * self.WEIGHT_CREDIT +
            self.liquidity_score * self.WEIGHT_LIQUIDITY +
            self.dte_buffer_score * self.WEIGHT_DTE +
            self.structure_score * self.WEIGHT_STRUCTURE +
            self.regime_stability_score * self.WEIGHT_REGIME
        )
    
    @property
    def weakest_component(self) -> str:
        """Identify the weakest scoring component."""
        scores = {
            "bid_ask": self.bid_ask_score,
            "credit_envelope": self.credit_envelope_score,
            "liquidity": self.liquidity_score,
            "dte_buffer": self.dte_buffer_score,
            "structure": self.structure_score,
            "regime_stability": self.regime_stability_score,
        }
        return min(scores, key=scores.get)
    
    @property
    def strongest_component(self) -> str:
        """Identify the strongest scoring component."""
        scores = {
            "bid_ask": self.bid_ask_score,
            "credit_envelope": self.credit_envelope_score,
            "liquidity": self.liquidity_score,
            "dte_buffer": self.dte_buffer_score,
            "structure": self.structure_score,
            "regime_stability": self.regime_stability_score,
        }
        return max(scores, key=scores.get)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_ask_score": round(self.bid_ask_score, 1),
            "credit_envelope_score": round(self.credit_envelope_score, 1),
            "liquidity_score": round(self.liquidity_score, 1),
            "dte_buffer_score": round(self.dte_buffer_score, 1),
            "structure_score": round(self.structure_score, 1),
            "regime_stability_score": round(self.regime_stability_score, 1),
            "total_score": round(self.total_score, 1),
            "weakest": self.weakest_component,
            "strongest": self.strongest_component,
        }


def score_bid_ask_spread(
    bid: float,
    ask: float,
    threshold_tight: float = 0.10,  # 10% spread = 100 score
    threshold_wide: float = 0.30,   # 30% spread = 0 score
) -> float:
    """
    Score based on bid-ask spread tightness.
    
    Spread < 10%: 100 (excellent)
    Spread 10-30%: linear interpolation
    Spread > 30%: 0 (untradeable)
    """
    if bid <= 0 or ask <= bid:
        return 0.0
    
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    
    if spread_pct <= threshold_tight:
        return 100.0
    elif spread_pct >= threshold_wide:
        return 0.0
    else:
        # Linear interpolation
        return 100.0 * (threshold_wide - spread_pct) / (threshold_wide - threshold_tight)


def score_credit_envelope(
    credit_low: float,
    credit_mid: float,
    credit_high: float,
    min_acceptable: float = 0.20,
) -> float:
    """
    Score based on credit envelope width.
    
    Narrower envelope = more predictable execution = higher score.
    Also penalizes if mid credit is below minimum acceptable.
    """
    if credit_mid <= 0:
        return 50.0  # neutral for debit spreads
    
    # Width penalty
    if credit_high > 0:
        width_ratio = (credit_high - credit_low) / credit_mid
    else:
        width_ratio = 1.0
    
    # Narrower is better (ratio < 0.3 is excellent)
    if width_ratio <= 0.3:
        width_score = 100.0
    elif width_ratio >= 1.0:
        width_score = 30.0
    else:
        width_score = 100.0 - 70.0 * (width_ratio - 0.3) / 0.7
    
    # Credit adequacy
    if credit_mid >= min_acceptable:
        adequacy_score = 100.0
    elif credit_low >= min_acceptable * 0.5:
        adequacy_score = 70.0
    else:
        adequacy_score = 40.0
    
    return (width_score + adequacy_score) / 2


def score_liquidity(
    open_interest: Optional[int] = None,
    volume: Optional[int] = None,
    min_oi: int = 100,
    min_volume: int = 50,
) -> float:
    """
    Score based on option liquidity metrics.
    
    Higher OI and volume = better liquidity = higher score.
    """
    score = 50.0  # baseline
    
    if open_interest is not None:
        if open_interest >= min_oi * 5:
            score += 25.0
        elif open_interest >= min_oi * 2:
            score += 15.0
        elif open_interest >= min_oi:
            score += 5.0
        else:
            score -= 15.0
    
    if volume is not None:
        if volume >= min_volume * 5:
            score += 25.0
        elif volume >= min_volume * 2:
            score += 15.0
        elif volume >= min_volume:
            score += 5.0
        else:
            score -= 10.0
    
    return max(0.0, min(100.0, score))


def score_dte_buffer(
    dte_days: int,
    execution_delay_hours: int = 72,  # up to 3 days delay
) -> float:
    """
    Score based on time-to-expiry buffer.
    
    Longer DTE = more buffer for delayed execution = higher score.
    """
    # Minimum DTE for any score
    min_dte = 7
    
    if dte_days < min_dte:
        return 20.0  # very risky
    
    # Ideal DTE for paper trading with delayed execution
    ideal_dte = 14 + (execution_delay_hours // 24)
    
    if dte_days >= ideal_dte:
        return 100.0
    else:
        # Scale from min to ideal
        return 20.0 + 80.0 * (dte_days - min_dte) / (ideal_dte - min_dte)


def score_structure_sensitivity(
    structure_type: str,
    long_dte: Optional[int] = None,
    short_dte: Optional[int] = None,
) -> float:
    """
    Score based on structure type.
    
    Some structures are more sensitive to entry timing than others.
    
    long_only: Least sensitive (100)
    diagonal: Moderate sensitivity (75-90)
    calendar: High sensitivity (50-70)
    vertical (same DTE): Very high sensitivity (30-50)
    """
    base_scores = {
        "long_only": 100.0,
        "diagonal": 85.0,
        "calendar": 60.0,
        "vertical": 40.0,
    }
    
    base = base_scores.get(structure_type, 50.0)
    
    # Adjust for DTE mismatch in diagonals
    if structure_type == "diagonal" and long_dte and short_dte:
        dte_diff = abs(long_dte - short_dte)
        if dte_diff > 20:
            base += 10  # wider diagonal is more robust
        elif dte_diff < 7:
            base -= 15  # narrow diagonal is fragile
    
    return max(0.0, min(100.0, base))


def score_regime_stability(
    regime: RegimeState,
    regime_duration_weeks: int = 0,
) -> float:
    """
    Score based on regime stability.
    
    Stable regime = lower chance of invalidation = higher score.
    Transitional regimes score lower.
    """
    # Base score by regime type
    regime_base = {
        VolatilityRegime.CALM: 90.0,
        VolatilityRegime.RISING: 50.0,  # transitional
        VolatilityRegime.STRESSED: 60.0,
        VolatilityRegime.DECLINING: 55.0,  # transitional
        VolatilityRegime.EXTREME: 40.0,  # volatile
    }
    
    base = regime_base.get(regime.regime, 50.0)
    
    # Confidence adjustment
    base *= regime.confidence
    
    # Duration bonus (longer in regime = more stable)
    if regime_duration_weeks >= 4:
        base += 15.0
    elif regime_duration_weeks >= 2:
        base += 5.0
    
    # Slope penalty (high slope = unstable)
    if abs(regime.vix_slope) > 0.20:
        base -= 20.0
    elif abs(regime.vix_slope) > 0.10:
        base -= 10.0
    
    return max(0.0, min(100.0, base))


def compute_robustness_score(
    # Option market data
    bid: float = 0.0,
    ask: float = 0.0,
    open_interest: Optional[int] = None,
    volume: Optional[int] = None,
    
    # Credit envelope
    credit_low: float = 0.0,
    credit_mid: float = 0.0,
    credit_high: float = 0.0,
    min_acceptable_credit: float = 0.20,
    
    # Structure
    structure_type: str = "diagonal",
    long_dte: Optional[int] = None,
    short_dte: Optional[int] = None,
    
    # Regime
    regime: Optional[RegimeState] = None,
    regime_duration_weeks: int = 0,
    
    # Execution context
    execution_delay_hours: int = 72,
) -> RobustnessComponents:
    """
    Compute complete robustness score with all components.
    
    Returns RobustnessComponents with individual and total scores.
    """
    # 1. Bid-ask score
    if bid > 0 and ask > bid:
        bid_ask = score_bid_ask_spread(bid, ask)
    else:
        bid_ask = 50.0  # neutral if no data
    
    # 2. Credit envelope score
    credit_score = score_credit_envelope(
        credit_low, credit_mid, credit_high, min_acceptable_credit
    )
    
    # 3. Liquidity score
    liquidity = score_liquidity(open_interest, volume)
    
    # 4. DTE buffer score
    dte = long_dte or short_dte or 14
    dte_score = score_dte_buffer(dte, execution_delay_hours)
    
    # 5. Structure score
    structure = score_structure_sensitivity(structure_type, long_dte, short_dte)
    
    # 6. Regime stability
    if regime:
        regime_score = score_regime_stability(regime, regime_duration_weeks)
    else:
        regime_score = 50.0
    
    return RobustnessComponents(
        bid_ask_score=bid_ask,
        credit_envelope_score=credit_score,
        liquidity_score=liquidity,
        dte_buffer_score=dte_score,
        structure_score=structure,
        regime_stability_score=regime_score,
    )


def get_robustness_rating(score: float) -> str:
    """Get rating label from score."""
    if score >= 70:
        return "GREEN"
    elif score >= 40:
        return "YELLOW"
    else:
        return "RED"


def get_robustness_description(score: float) -> str:
    """Get human-readable description."""
    if score >= 80:
        return "Highly execution-friendly, robust to delays and slippage"
    elif score >= 70:
        return "Execution-friendly, acceptable for paper trading"
    elif score >= 55:
        return "Moderate robustness, exercise caution"
    elif score >= 40:
        return "Low robustness, theoretical only unless improved"
    else:
        return "Poor robustness, not recommended for execution"


def get_robustness_color(score: float) -> str:
    """Get color for UI display."""
    if score >= 70:
        return "#2ECC71"  # Green
    elif score >= 40:
        return "#F39C12"  # Orange/Yellow
    else:
        return "#E74C3C"  # Red


# --- Batch scoring utilities ---

def score_signal_batch(
    signals: List[Dict[str, Any]],
    regime: Optional[RegimeState] = None,
) -> List[Dict[str, Any]]:
    """
    Score robustness for a batch of signals.
    
    Returns list of dicts with signal data + robustness info.
    """
    results = []
    
    for signal in signals:
        components = compute_robustness_score(
            bid=signal.get("bid", 0),
            ask=signal.get("ask", 0),
            open_interest=signal.get("open_interest"),
            volume=signal.get("volume"),
            credit_low=signal.get("expected_credit_low", 0),
            credit_mid=signal.get("expected_credit_mid", 0),
            credit_high=signal.get("expected_credit_high", 0),
            min_acceptable_credit=signal.get("min_acceptable_credit", 0.20),
            structure_type=signal.get("structure_type", "diagonal"),
            long_dte=signal.get("long_dte_weeks", 0) * 7 if signal.get("long_dte_weeks") else None,
            short_dte=signal.get("short_dte_weeks", 0) * 7 if signal.get("short_dte_weeks") else None,
            regime=regime,
        )
        
        results.append({
            **signal,
            "robustness_score": components.total_score,
            "robustness_components": components.to_dict(),
            "robustness_rating": get_robustness_rating(components.total_score),
            "robustness_color": get_robustness_color(components.total_score),
        })
    
    return results
