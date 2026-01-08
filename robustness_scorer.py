"""
Robustness Scorer for VIX 5% Weekly Suite

Scores trading signals based on multiple factors to assess
execution survivability and trade quality.

Factors considered:
- Liquidity (bid-ask spread, volume)
- Market conditions alignment
- Strike distance from spot
- Time to expiration
- Regime stability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import datetime as dt

from enums import VolatilityRegime, VariantRole


@dataclass
class RobustnessScore:
    """Detailed robustness scoring breakdown."""
    total_score: float  # 0-100
    confidence: str     # "low", "medium", "high"
    
    # Component scores (0-100 each)
    liquidity_score: float
    regime_score: float
    strike_score: float
    timing_score: float
    structure_score: float
    
    # Flags
    warnings: list
    recommendations: list
    
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


def score_liquidity(
    estimated_debit: float,
    suggested_contracts: int,
    underlying_symbol: str = "^VIX",
) -> tuple[float, list]:
    """
    Score based on estimated liquidity.
    
    Returns:
        (score, warnings)
    """
    warnings = []
    
    # VIX options generally liquid, UVXY less so
    if underlying_symbol in ("UVXY", "VXX"):
        base_score = 70  # Less liquid
        if suggested_contracts > 10:
            warnings.append("UVXY/VXX may have liquidity issues for larger positions")
            base_score -= 10
    else:
        base_score = 85  # VIX options typically liquid
    
    # Penalize very large positions
    if suggested_contracts > 20:
        warnings.append("Large position size may face execution challenges")
        base_score -= 15
    elif suggested_contracts > 10:
        base_score -= 5
    
    # Penalize very small or very large debits
    if estimated_debit < 100:
        warnings.append("Very small position - commission impact significant")
        base_score -= 10
    elif estimated_debit > 50000:
        warnings.append("Large debit - consider scaling in")
        base_score -= 5
    
    return max(0, min(100, base_score)), warnings


def score_regime(
    regime: VolatilityRegime,
    variant_role: VariantRole,
    vix_percentile: float,
    regime_duration_days: int = 0,
) -> tuple[float, list]:
    """
    Score based on regime alignment with variant strategy.
    
    Returns:
        (score, warnings)
    """
    warnings = []
    
    # Define ideal regime/variant alignments
    ideal_alignments = {
        VariantRole.INCOME: [VolatilityRegime.CALM],
        VariantRole.DECAY: [VolatilityRegime.DECLINING],
        VariantRole.HEDGE: [VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        VariantRole.CONVEX: [VolatilityRegime.EXTREME],
        VariantRole.ADAPTIVE: list(VolatilityRegime),  # All regimes
    }
    
    ideal = ideal_alignments.get(variant_role, [])
    
    if regime in ideal:
        base_score = 90
    else:
        base_score = 60
        warnings.append(f"{variant_role.value} not ideal for {regime.value} regime")
    
    # Bonus for regime stability
    if regime_duration_days > 30:
        base_score += 5
    elif regime_duration_days < 7:
        base_score -= 5
        warnings.append("Recent regime transition - increased uncertainty")
    
    # Check percentile edge cases
    if vix_percentile < 0.05:
        warnings.append("Extremely low VIX - potential complacency")
    elif vix_percentile > 0.95:
        warnings.append("Extremely high VIX - potential mean reversion")
    
    return max(0, min(100, base_score)), warnings


def score_strike_selection(
    vix_level: float,
    long_strike: float,
    short_strike: Optional[float],
    position_type: str,
) -> tuple[float, list]:
    """
    Score based on strike selection relative to spot.
    
    Returns:
        (score, warnings)
    """
    warnings = []
    
    # Calculate OTM distance
    otm_distance = long_strike - vix_level
    otm_pct = otm_distance / vix_level if vix_level > 0 else 0
    
    base_score = 80
    
    # Penalize ITM long calls
    if otm_distance < 0:
        warnings.append("Long strike is ITM - higher cost, lower leverage")
        base_score -= 20
    
    # Penalize very far OTM
    if otm_pct > 0.50:  # More than 50% OTM
        warnings.append("Long strike very far OTM - low delta, may expire worthless")
        base_score -= 15
    elif otm_pct > 0.30:
        base_score -= 5
    
    # For diagonals, check short strike
    if short_strike is not None and position_type == "diagonal":
        short_distance = short_strike - vix_level
        if short_distance < otm_distance:
            warnings.append("Short strike closer than long - inverted diagonal")
            base_score -= 25
    
    return max(0, min(100, base_score)), warnings


def score_timing(
    long_dte_days: int,
    short_dte_days: Optional[int],
    position_type: str,
    signal_time: dt.datetime,
) -> tuple[float, list]:
    """
    Score based on option timing and DTE.
    
    Returns:
        (score, warnings)
    """
    warnings = []
    base_score = 80
    
    # Check long leg DTE
    if long_dte_days < 14:
        warnings.append("Very short DTE on long leg - high theta decay")
        base_score -= 20
    elif long_dte_days < 30:
        warnings.append("Short DTE on long leg - accelerated decay")
        base_score -= 10
    elif long_dte_days > 180:
        base_score += 5  # More time = more flexibility
    
    # For diagonals, check short leg timing
    if position_type == "diagonal" and short_dte_days is not None:
        if short_dte_days < 5:
            warnings.append("Short leg close to expiration - roll needed soon")
            base_score -= 5
        elif short_dte_days > 14:
            base_score += 5  # More premium capture time
    
    # Check day of week (prefer Thurs/Fri signals for weekly options)
    dow = signal_time.weekday()
    if dow in (3, 4):  # Thursday or Friday
        base_score += 5
    elif dow == 0:  # Monday
        base_score -= 5
        warnings.append("Monday signal - wait for mid-week for better pricing")
    
    return max(0, min(100, base_score)), warnings


def score_structure(
    position_type: str,
    variant_role: VariantRole,
    estimated_debit: float,
    max_loss: float,
    target_mult: float,
) -> tuple[float, list]:
    """
    Score based on trade structure and risk/reward.
    
    Returns:
        (score, warnings)
    """
    warnings = []
    base_score = 75
    
    # Check risk/reward
    if target_mult < 1.10:
        warnings.append("Low profit target - tight margin for error")
        base_score -= 10
    elif target_mult > 3.0:
        base_score += 5  # High reward target
    
    # Check structure alignment with variant
    structure_match = {
        VariantRole.INCOME: ["diagonal"],
        VariantRole.DECAY: ["diagonal"],
        VariantRole.HEDGE: ["long_call", "put_spread"],
        VariantRole.CONVEX: ["long_call"],
        VariantRole.ADAPTIVE: ["diagonal", "long_call", "adaptive"],
    }
    
    ideal_structures = structure_match.get(variant_role, [])
    if position_type in ideal_structures:
        base_score += 10
    else:
        warnings.append(f"Structure '{position_type}' may not be ideal for {variant_role.value}")
    
    # Check max loss vs debit
    if max_loss > estimated_debit * 1.5:
        warnings.append("Max loss exceeds debit significantly - check position sizing")
        base_score -= 10
    
    return max(0, min(100, base_score)), warnings


def calculate_robustness_score(
    # Signal details
    variant_role: VariantRole,
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying_symbol: str,
    
    # Position details
    position_type: str,
    long_strike: float,
    long_dte_days: int,
    short_strike: Optional[float] = None,
    short_dte_days: Optional[int] = None,
    
    # Sizing
    suggested_contracts: int = 1,
    estimated_debit: float = 0.0,
    max_loss: float = 0.0,
    target_mult: float = 1.20,
    
    # Context
    regime_duration_days: int = 0,
    signal_time: Optional[dt.datetime] = None,
) -> RobustnessScore:
    """
    Calculate comprehensive robustness score for a trading signal.
    
    Returns:
        RobustnessScore with detailed breakdown
    """
    if signal_time is None:
        signal_time = dt.datetime.now()
    
    all_warnings = []
    all_recommendations = []
    
    # Score each component
    liquidity, liq_warnings = score_liquidity(
        estimated_debit, suggested_contracts, underlying_symbol
    )
    all_warnings.extend(liq_warnings)
    
    regime_sc, reg_warnings = score_regime(
        regime, variant_role, vix_percentile, regime_duration_days
    )
    all_warnings.extend(reg_warnings)
    
    strike, strike_warnings = score_strike_selection(
        vix_level, long_strike, short_strike, position_type
    )
    all_warnings.extend(strike_warnings)
    
    timing, timing_warnings = score_timing(
        long_dte_days, short_dte_days, position_type, signal_time
    )
    all_warnings.extend(timing_warnings)
    
    structure, struct_warnings = score_structure(
        position_type, variant_role, estimated_debit, max_loss, target_mult
    )
    all_warnings.extend(struct_warnings)
    
    # Calculate weighted total
    weights = {
        "liquidity": 0.15,
        "regime": 0.25,
        "strike": 0.20,
        "timing": 0.20,
        "structure": 0.20,
    }
    
    total = (
        liquidity * weights["liquidity"] +
        regime_sc * weights["regime"] +
        strike * weights["strike"] +
        timing * weights["timing"] +
        structure * weights["structure"]
    )
    
    # Determine confidence
    if total >= 80:
        confidence = "high"
        all_recommendations.append("Signal looks strong - proceed with normal sizing")
    elif total >= 60:
        confidence = "medium"
        all_recommendations.append("Signal acceptable - consider reducing size")
    else:
        confidence = "low"
        all_recommendations.append("Signal weak - consider skipping or minimal size")
    
    # Add specific recommendations based on warnings
    if len(all_warnings) > 3:
        all_recommendations.append("Multiple concerns - extra caution advised")
    
    return RobustnessScore(
        total_score=round(total, 1),
        confidence=confidence,
        liquidity_score=round(liquidity, 1),
        regime_score=round(regime_sc, 1),
        strike_score=round(strike, 1),
        timing_score=round(timing, 1),
        structure_score=round(structure, 1),
        warnings=all_warnings,
        recommendations=all_recommendations,
    )


def format_robustness_display(score: RobustnessScore) -> str:
    """Format robustness score for display."""
    # Emoji based on confidence
    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(score.confidence, "⚪")
    
    lines = [
        f"{emoji} **Robustness: {score.total_score}/100** ({score.confidence.upper()})",
        "",
        "**Component Scores:**",
        f"- Liquidity: {score.liquidity_score}/100",
        f"- Regime: {score.regime_score}/100",
        f"- Strike: {score.strike_score}/100",
        f"- Timing: {score.timing_score}/100",
        f"- Structure: {score.structure_score}/100",
    ]
    
    if score.warnings:
        lines.extend(["", "**⚠️ Warnings:**"])
        for w in score.warnings:
            lines.append(f"- {w}")
    
    if score.recommendations:
        lines.extend(["", "**💡 Recommendations:**"])
        for r in score.recommendations:
            lines.append(f"- {r}")
    
    return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing robustness scorer...")
    
    score = calculate_robustness_score(
        variant_role=VariantRole.INCOME,
        regime=VolatilityRegime.CALM,
        vix_level=14.5,
        vix_percentile=0.15,
        underlying_symbol="^VIX",
        position_type="diagonal",
        long_strike=20.0,
        long_dte_days=180,
        short_strike=20.0,
        short_dte_days=7,
        suggested_contracts=5,
        estimated_debit=2500.0,
        max_loss=2500.0,
        target_mult=1.20,
        regime_duration_days=30,
    )
    
    print(format_robustness_display(score))
