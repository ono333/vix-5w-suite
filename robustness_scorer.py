"""
Robustness Scorer for VIX 5% Weekly Suite
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enums import VolatilityRegime, VariantRole

@dataclass
class RobustnessResult:
    total_score: float
    regime_alignment: float = 0.0
    liquidity_score: float = 0.0
    timing_score: float = 0.0
    risk_reward_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

def calculate_robustness(variant, regime_state) -> RobustnessResult:
    if hasattr(regime_state, 'regime'):
        regime = regime_state.regime
        pct = regime_state.vix_percentile
        conf = getattr(regime_state, 'confidence', 0.8)
    else:
        regime = VolatilityRegime.CALM
        pct = 0.5
        conf = 0.7
    
    is_active = regime in getattr(variant, 'active_in_regimes', [])
    regime_score = 90 if is_active else 40
    
    liquidity = 70 + (1 - pct) * 20
    timing = 60 + conf * 30
    
    tp = getattr(variant, 'tp_pct', 0.2)
    sl = getattr(variant, 'sl_pct', 0.5)
    rr = tp / sl if sl > 0 else 1.0
    risk_reward = min(90, 50 + rr * 20)
    
    total = (regime_score * 0.35 + liquidity * 0.25 + timing * 0.20 + risk_reward * 0.20)
    
    return RobustnessResult(
        total_score=total,
        regime_alignment=regime_score,
        liquidity_score=liquidity,
        timing_score=timing,
        risk_reward_score=risk_reward,
    )

def batch_score_variants(variants: List, regime_state) -> List[RobustnessResult]:
    return [calculate_robustness(v, regime_state) for v in variants]

def get_robustness_color(score: float) -> str:
    if score >= 80:
        return "#4CAF50"
    elif score >= 60:
        return "#FF9800"
    else:
        return "#f44336"

def get_robustness_label(score: float) -> str:
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"
