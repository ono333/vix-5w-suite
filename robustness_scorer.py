"""
Robustness Scorer for VIX 5% Weekly Suite

Exports:
    - calculate_robustness
    - batch_score_variants
    - RobustnessResult
    - get_robustness_color
    - get_robustness_label
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from enums import VolatilityRegime, VariantRole


@dataclass
class RobustnessResult:
    """Result of robustness scoring."""
    total_score: float  # 0-100
    confidence: str     # "low", "medium", "high"
    
    # Component scores
    liquidity_score: float = 0.0
    regime_score: float = 0.0
    strike_score: float = 0.0
    timing_score: float = 0.0
    structure_score: float = 0.0
    
    # Feedback
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "confidence": self.confidence,
            "liquidity_score": self.liquidity_score,
            "regime_score": self.regime_score,
            "strike_score": self.strike_score,
            "timing_score": self.timing_score,
            "structure_score": self.structure_score,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


def get_robustness_color(score: float) -> str:
    """Get color based on robustness score."""
    if score >= 80:
        return "#27AE60"  # Green
    elif score >= 60:
        return "#F1C40F"  # Yellow
    elif score >= 40:
        return "#E67E22"  # Orange
    else:
        return "#E74C3C"  # Red


def get_robustness_label(score: float) -> str:
    """Get label based on robustness score."""
    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Moderate"
    elif score >= 40:
        return "Weak"
    else:
        return "Poor"


def _score_liquidity(
    contracts: int,
    underlying: str,
    estimated_debit: float,
) -> tuple[float, List[str]]:
    """Score liquidity factors."""
    warnings = []
    score = 80.0
    
    if underlying in ("UVXY", "VXX"):
        score -= 10
        if contracts > 10:
            warnings.append("UVXY/VXX may have liquidity issues for larger positions")
            score -= 10
    
    if contracts > 20:
        warnings.append("Large position - may face execution challenges")
        score -= 15
    
    if estimated_debit < 100:
        warnings.append("Very small position - commission impact high")
        score -= 10
    elif estimated_debit > 50000:
        warnings.append("Large debit - consider scaling in")
        score -= 5
    
    return max(0, min(100, score)), warnings


def _score_regime(
    regime: VolatilityRegime,
    role: VariantRole,
    vix_percentile: float,
) -> tuple[float, List[str]]:
    """Score regime alignment."""
    warnings = []
    
    ideal = {
        VariantRole.INCOME: [VolatilityRegime.CALM],
        VariantRole.DECAY: [VolatilityRegime.DECLINING],
        VariantRole.HEDGE: [VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        VariantRole.CONVEX: [VolatilityRegime.EXTREME],
        VariantRole.ADAPTIVE: list(VolatilityRegime),
    }
    
    if regime in ideal.get(role, []):
        score = 90.0
    else:
        score = 60.0
        warnings.append(f"{role.value} not ideal for {regime.value} regime")
    
    if vix_percentile < 0.05:
        warnings.append("Extremely low VIX - potential complacency")
    elif vix_percentile > 0.95:
        warnings.append("Extremely high VIX - potential mean reversion")
    
    return max(0, min(100, score)), warnings


def _score_strike(
    vix_level: float,
    long_strike: float,
    short_strike: Optional[float],
    position_type: str,
) -> tuple[float, List[str]]:
    """Score strike selection."""
    warnings = []
    score = 80.0
    
    otm_distance = long_strike - vix_level
    otm_pct = otm_distance / vix_level if vix_level > 0 else 0
    
    if otm_distance < 0:
        warnings.append("Long strike is ITM - higher cost")
        score -= 20
    
    if otm_pct > 0.50:
        warnings.append("Long strike very far OTM - low delta")
        score -= 15
    elif otm_pct > 0.30:
        score -= 5
    
    if short_strike and position_type == "diagonal":
        if short_strike - vix_level < otm_distance:
            warnings.append("Short strike closer than long - inverted diagonal")
            score -= 25
    
    return max(0, min(100, score)), warnings


def _score_timing(
    long_dte_days: int,
    short_dte_days: Optional[int],
    position_type: str,
) -> tuple[float, List[str]]:
    """Score timing factors."""
    warnings = []
    score = 80.0
    
    if long_dte_days < 14:
        warnings.append("Very short DTE on long leg - high theta decay")
        score -= 20
    elif long_dte_days < 30:
        warnings.append("Short DTE on long leg")
        score -= 10
    elif long_dte_days > 180:
        score += 5
    
    if position_type == "diagonal" and short_dte_days:
        if short_dte_days < 5:
            warnings.append("Short leg close to expiration - roll needed")
            score -= 5
    
    return max(0, min(100, score)), warnings


def _score_structure(
    position_type: str,
    role: VariantRole,
    target_mult: float,
) -> tuple[float, List[str]]:
    """Score trade structure."""
    warnings = []
    score = 75.0
    
    if target_mult < 1.10:
        warnings.append("Low profit target - tight margin")
        score -= 10
    elif target_mult > 3.0:
        score += 5
    
    ideal_structures = {
        VariantRole.INCOME: ["diagonal"],
        VariantRole.DECAY: ["diagonal"],
        VariantRole.HEDGE: ["long_call"],
        VariantRole.CONVEX: ["long_call"],
        VariantRole.ADAPTIVE: ["diagonal", "long_call", "adaptive"],
    }
    
    if position_type in ideal_structures.get(role, []):
        score += 10
    
    return max(0, min(100, score)), warnings


def calculate_robustness(
    role: VariantRole,
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying: str,
    position_type: str,
    long_strike: float,
    long_dte_days: int,
    short_strike: Optional[float] = None,
    short_dte_days: Optional[int] = None,
    contracts: int = 1,
    estimated_debit: float = 0.0,
    target_mult: float = 1.20,
) -> RobustnessResult:
    """Calculate robustness score for a signal."""
    
    all_warnings = []
    all_recommendations = []
    
    # Score components
    liq_score, liq_warn = _score_liquidity(contracts, underlying, estimated_debit)
    all_warnings.extend(liq_warn)
    
    reg_score, reg_warn = _score_regime(regime, role, vix_percentile)
    all_warnings.extend(reg_warn)
    
    strike_score, strike_warn = _score_strike(vix_level, long_strike, short_strike, position_type)
    all_warnings.extend(strike_warn)
    
    time_score, time_warn = _score_timing(long_dte_days, short_dte_days, position_type)
    all_warnings.extend(time_warn)
    
    struct_score, struct_warn = _score_structure(position_type, role, target_mult)
    all_warnings.extend(struct_warn)
    
    # Weighted total
    total = (
        liq_score * 0.15 +
        reg_score * 0.25 +
        strike_score * 0.20 +
        time_score * 0.20 +
        struct_score * 0.20
    )
    
    # Confidence
    if total >= 80:
        confidence = "high"
        all_recommendations.append("Signal looks strong - proceed with normal sizing")
    elif total >= 60:
        confidence = "medium"
        all_recommendations.append("Signal acceptable - consider reducing size")
    else:
        confidence = "low"
        all_recommendations.append("Signal weak - consider skipping or minimal size")
    
    if len(all_warnings) > 3:
        all_recommendations.append("Multiple concerns - extra caution advised")
    
    return RobustnessResult(
        total_score=round(total, 1),
        confidence=confidence,
        liquidity_score=round(liq_score, 1),
        regime_score=round(reg_score, 1),
        strike_score=round(strike_score, 1),
        timing_score=round(time_score, 1),
        structure_score=round(struct_score, 1),
        warnings=all_warnings,
        recommendations=all_recommendations,
    )


def batch_score_variants(signals: List[Any]) -> Dict[str, RobustnessResult]:
    """Score multiple signals at once."""
    results = {}
    
    for sig in signals:
        result = calculate_robustness(
            role=sig.role,
            regime=sig.regime,
            vix_level=sig.vix_level,
            vix_percentile=sig.vix_percentile,
            underlying=sig.underlying,
            position_type=sig.position_type,
            long_strike=sig.long_strike,
            long_dte_days=sig.long_dte_days,
            short_strike=sig.short_strike,
            short_dte_days=sig.short_dte_days,
            contracts=sig.contracts,
            estimated_debit=sig.estimated_debit,
            target_mult=sig.target_mult,
        )
        results[sig.signal_id] = result
    
    return results
