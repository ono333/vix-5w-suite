#!/usr/bin/env python3
"""
Variant Generator for VIX/UVXY Suite

Generates 5 distinct strategy variants based on current regime conditions.
Each variant has a specific ROLE, not just different parameters.

Variant Roles:
- V1 (INCOME): Income Harvester - stability anchor, frequent small gains
- V2 (DECAY): Mean Reversion Accelerator - post-spike decay capture  
- V3 (HEDGE): Shock Absorber - crisis hedge, drawdown reduction
- V4 (CONVEX): Convex Tail Hunter - maximum gain, rare explosive payoffs
- V5 (ADAPTIVE): Regime-Aware Allocator - controls sizing of other variants

Each variant is generated with:
- Role-specific parameter constraints
- Regime-conditional activation
- Robustness scoring
- Suggested TP/SL levels
"""

from __future__ import annotations

from pandas import Series
from enums import VolatilityRegime



from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

import numpy as np

from regime_detector import VolatilityRegime, RegimeState

from enums import VolatilityRegime
from utils.regime_utils import extract_current_regime

from enums import VolatilityRegime
from utils.regime_utils import extract_current_regime


def generate_all_variants(uvxy_data, regime):
    regime = extract_current_regime(regime)

    # existing code continues below


class VariantRole(Enum):
    """Strategy variant roles."""
    INCOME = "income"      # V1: Income Harvester
    DECAY = "decay"        # V2: Mean Reversion Accelerator
    HEDGE = "hedge"        # V3: Shock Absorber
    CONVEX = "convex"      # V4: Convex Tail Hunter
    ADAPTIVE = "adaptive"  # V5: Regime-Aware Allocator


@dataclass
class VariantParams:
    """Generated parameters for a single variant."""
    variant_id: str
    role: VariantRole
    
    # Structure
    structure: str  # "diagonal", "long_only", "credit_spread", "calendar"
    
    # Entry parameters
    entry_percentile: float
    entry_lookback_weeks: int
    
    # Strike selection
    long_strike_offset: float  # Points OTM for long leg
    short_strike_offset: float  # Points OTM for short leg
    
    # DTE selection
    long_dte_weeks: int
    short_dte_weeks: int
    
    # Position sizing
    alloc_pct: float  # Fraction of equity
    max_contracts: int
    
    # Exit parameters
    target_mult: float  # Take profit multiple
    exit_mult: float    # Stop loss multiple
    max_hold_weeks: int
    
    # Volatility scaling
    sigma_mult: float
    
    # Risk management
    tp_pct: float  # Take profit %
    sl_pct: float  # Stop loss %
    
    # Regime conditions
    active_in_regimes: List[VolatilityRegime]
    suppressed_in_regimes: List[VolatilityRegime]
    
    # Scoring
    robustness_score: float = 0.0
    liquidity_score: float = 0.0
    
    # Status
    status: str = "TRADE"  # "TRADE", "NO_TRADE", "SUPPRESSED"
    status_reason: str = ""
    
    # Metadata
    generated_at: str = ""
    signal_batch_id: str = ""
    valid_until: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "role": self.role.value,
            "structure": self.structure,
            "entry_percentile": self.entry_percentile,
            "entry_lookback_weeks": self.entry_lookback_weeks,
            "long_strike_offset": self.long_strike_offset,
            "short_strike_offset": self.short_strike_offset,
            "long_dte_weeks": self.long_dte_weeks,
            "short_dte_weeks": self.short_dte_weeks,
            "alloc_pct": self.alloc_pct,
            "max_contracts": self.max_contracts,
            "target_mult": self.target_mult,
            "exit_mult": self.exit_mult,
            "max_hold_weeks": self.max_hold_weeks,
            "sigma_mult": self.sigma_mult,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "active_in_regimes": [r.value for r in self.active_in_regimes],
            "suppressed_in_regimes": [r.value for r in self.suppressed_in_regimes],
            "robustness_score": self.robustness_score,
            "liquidity_score": self.liquidity_score,
            "status": self.status,
            "status_reason": self.status_reason,
            "generated_at": self.generated_at,
            "signal_batch_id": self.signal_batch_id,
            "valid_until": self.valid_until,
        }


@dataclass
class SignalBatch:
    """A batch of variant signals generated at a specific time."""
    batch_id: str
    generated_at: datetime
    valid_until: datetime
    regime_state: RegimeState
    variants: List[VariantParams]
    frozen: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "regime_state": self.regime_state.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "frozen": self.frozen,
        }


def _generate_batch_id(timestamp: datetime) -> str:
    """Generate unique batch ID."""
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    hash_input = f"{ts_str}_{np.random.randint(10000)}"
    return f"SIG_{ts_str}_{hashlib.md5(hash_input.encode()).hexdigest()[:6]}"


def _generate_variant_id(role: VariantRole, batch_id: str) -> str:
    """Generate unique variant ID."""
    role_num = {
        VariantRole.INCOME: 1,
        VariantRole.DECAY: 2,
        VariantRole.HEDGE: 3,
        VariantRole.CONVEX: 4,
        VariantRole.ADAPTIVE: 5,
    }
    return f"V{role_num[role]}_{batch_id}"


def generate_income_variant(
    regime: RegimeState,
    base_config: Dict[str, Any],
    batch_id: str,
) -> VariantParams:
    """
    V1: Income Harvester
    
    Objective: Maximize theta, frequent small gains, stability anchor.
    - Narrow spreads
    - Fast exits
    - Hard profit targets
    - Suppressed during rising/stressed regimes
    """
    variant_id = _generate_variant_id(VariantRole.INCOME, batch_id)
    
    # Regime-specific adjustments
    if regime == VolatilityRegime.CALM:
        alloc_pct = 0.35
        status = "TRADE"
        status_reason = "Optimal conditions for income harvesting"
    elif regime == VolatilityRegime.DECLINING:
        alloc_pct = 0.25
        status = "TRADE"
        status_reason = "Decay conditions favorable"
    elif regime == VolatilityRegime.RISING:
        alloc_pct = 0.10
        status = "TRADE"
        status_reason = "Reduced size due to rising volatility"
    else:
        alloc_pct = 0.0
        status = "SUPPRESSED"
        status_reason = f"Suppressed in {regime.regime.value} regime"
    
    return VariantParams(
        variant_id=variant_id,
        role=VariantRole.INCOME,
        structure="diagonal",
        entry_percentile=0.30,  # Enter at relatively low vol
        entry_lookback_weeks=52,
        long_strike_offset=5.0,   # Closer to ATM
        short_strike_offset=3.0,  # Near ATM for max theta
        long_dte_weeks=13,        # Shorter dated
        short_dte_weeks=1,        # Weekly rolls
        alloc_pct=alloc_pct,
        max_contracts=20,
        target_mult=1.15,         # Quick profit taking
        exit_mult=0.70,           # Tighter stop
        max_hold_weeks=4,
        sigma_mult=0.8,
        tp_pct=0.15,              # 15% take profit
        sl_pct=0.30,              # 30% stop loss
        active_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        status=status,
        status_reason=status_reason,
        generated_at=datetime.utcnow().isoformat(),
        signal_batch_id=batch_id,
    )


def generate_decay_variant(
    regime: RegimeState,
    base_config: Dict[str, Any],
    batch_id: str,
) -> VariantParams:
    """
    V2: Mean Reversion Accelerator
    
    Objective: Capture post-spike decay, not spikes themselves.
    - Entry delayed by regime condition (wait for peak confirmation)
    - Heavier short exposure
    - Minimal long hedge
    - Time-stop dominant
    """
    variant_id = _generate_variant_id(VariantRole.DECAY, batch_id)
    
    # Only activate after vol has peaked
    if regime.regime == VolatilityRegime.DECLINING:
        alloc_pct = 0.25
        status = "TRADE"
        status_reason = "Post-spike decay conditions optimal"
    elif regime.regime == VolatilityRegime.STRESSED and regime.vix_slope < -0.02:
        alloc_pct = 0.15
        status = "TRADE"
        status_reason = "Vol deceleration detected"
    else:
        alloc_pct = 0.0
        status = "NO_TRADE"
        status_reason = "Waiting for post-spike conditions"
    
    return VariantParams(
        variant_id=variant_id,
        role=VariantRole.DECAY,
        structure="diagonal",
        entry_percentile=0.70,    # Enter at higher vol (post-spike)
        entry_lookback_weeks=26,
        long_strike_offset=15.0,  # Far OTM for cheap hedge
        short_strike_offset=5.0,  # Closer ATM for max decay
        long_dte_weeks=8,
        short_dte_weeks=1,
        alloc_pct=alloc_pct,
        max_contracts=15,
        target_mult=1.25,
        exit_mult=0.60,
        max_hold_weeks=6,         # Time-based exit
        sigma_mult=1.0,
        tp_pct=0.25,
        sl_pct=0.40,
        active_in_regimes=[VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.RISING],
        status=status,
        status_reason=status_reason,
        generated_at=datetime.utcnow().isoformat(),
        signal_batch_id=batch_id,
    )


def generate_hedge_variant(
    regime: RegimeState,
    base_config: Dict[str, Any],
    batch_id: str,
) -> VariantParams:
    """
    V3: Shock Absorber
    
    Objective: Smooth equity curve, reduce drawdowns during stress.
    - Wider hedges
    - Earlier long activation
    - Net long vega during high stress
    - Lower position size
    """
    variant_id = _generate_variant_id(VariantRole.HEDGE, batch_id)
    
    # Scale up as stress increases
    if regime.regime == VolatilityRegime.CALM:
        alloc_pct = 0.05
        status = "TRADE"
        status_reason = "Maintaining small hedge position"
    elif regime.regime == VolatilityRegime.RISING:
        alloc_pct = 0.15
        status = "TRADE"
        status_reason = "Increasing hedge allocation"
    elif regime.regime in [VolatilityRegime.STRESSED, VolatilityRegime.EXTREME]:
        alloc_pct = 0.20
        status = "TRADE"
        status_reason = "Maximum hedge allocation in stressed conditions"
    else:
        alloc_pct = 0.10
        status = "TRADE"
        status_reason = "Standard hedge allocation"
    
    return VariantParams(
        variant_id=variant_id,
        role=VariantRole.HEDGE,
        structure="long_only",    # Pure long for hedge
        entry_percentile=0.50,    # Enter at mid-range
        entry_lookback_weeks=52,
        long_strike_offset=10.0,  # Moderate OTM
        short_strike_offset=0.0,  # No short leg for pure hedge
        long_dte_weeks=26,        # Longer dated for stability
        short_dte_weeks=0,
        alloc_pct=alloc_pct,
        max_contracts=10,
        target_mult=2.0,          # Let winners run
        exit_mult=0.40,           # Wide stop
        max_hold_weeks=12,
        sigma_mult=1.2,
        tp_pct=1.00,              # 100% (very wide)
        sl_pct=0.60,
        active_in_regimes=[r for r in VolatilityRegime],  # Always active
        suppressed_in_regimes=[],
        status=status,
        status_reason=status_reason,
        generated_at=datetime.utcnow().isoformat(),
        signal_batch_id=batch_id,
    )


def generate_convex_variant(
    regime: RegimeState,
    base_config: Dict[str, Any],
    batch_id: str,
) -> VariantParams:
    """
    V4: Convex Tail Hunter (MAXIMUM GAIN VARIANT)
    
    Objective: Maximum payoff during rare volatility spikes.
    - Very small short size
    - Disproportionately large long convexity
    - NO early exit on longs
    - Activate BEFORE stress (when convexity is cheap)
    
    Expect: 70-90% of cycles may lose, but survivors are explosive.
    """
    variant_id = _generate_variant_id(VariantRole.CONVEX, batch_id)
    
    # Key: activate when convexity is CHEAP (before panic)
    if regime.regime == VolatilityRegime.CALM:
        alloc_pct = 0.08
        status = "TRADE"
        status_reason = "Convexity cheap - building position"
    elif regime.regime == VolatilityRegime.RISING:
        alloc_pct = 0.15
        status = "TRADE"
        status_reason = "Rising vol - maximum convex allocation"
    elif regime.regime == VolatilityRegime.STRESSED:
        alloc_pct = 0.05
        status = "TRADE"
        status_reason = "Holding existing convex positions"
    elif regime.regime == VolatilityRegime.EXTREME:
        alloc_pct = 0.0
        status = "NO_TRADE"
        status_reason = "Convexity too expensive - hold existing only"
    else:
        alloc_pct = 0.03
        status = "TRADE"
        status_reason = "Minimal convex exposure in declining vol"
    
    return VariantParams(
        variant_id=variant_id,
        role=VariantRole.CONVEX,
        structure="diagonal",     # Unbalanced diagonal
        entry_percentile=0.25,    # Enter at low vol when cheap
        entry_lookback_weeks=52,
        long_strike_offset=20.0,  # Far OTM for convexity
        short_strike_offset=5.0,  # Minimal short (just for financing)
        long_dte_weeks=26,        # Longer dated
        short_dte_weeks=2,        # Slightly longer short
        alloc_pct=alloc_pct,
        max_contracts=25,
        target_mult=3.0,          # HIGH target - let it run
        exit_mult=0.30,           # Accept large drawdowns
        max_hold_weeks=20,        # Long holding period
        sigma_mult=1.5,
        tp_pct=2.00,              # 200% - explosive upside
        sl_pct=0.70,              # Accept 70% loss on entry
        active_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.RISING],
        suppressed_in_regimes=[VolatilityRegime.EXTREME],  # Too expensive
        status=status,
        status_reason=status_reason,
        generated_at=datetime.utcnow().isoformat(),
        signal_batch_id=batch_id,
    )


def generate_adaptive_variant(
    regime: RegimeState,
    base_config: Dict[str, Any],
    batch_id: str,
    other_variants: List[VariantParams],
) -> VariantParams:
    """
    V5: Regime-Aware Allocator
    
    This variant doesn't trade directly - it adjusts other variants.
    Outputs: sizing multipliers, activation overrides, exit biases.
    """
    variant_id = _generate_variant_id(VariantRole.ADAPTIVE, batch_id)
    
    # Calculate recommended adjustments
    total_alloc = sum(v.alloc_pct for v in other_variants if v.status == "TRADE")
    
    # This variant provides allocation guidance
    if regime.regime == VolatilityRegime.CALM:
        status_reason = f"Normal allocation. Total: {total_alloc:.0%}"
    elif regime.regime == VolatilityRegime.RISING:
        status_reason = f"Shift to V3/V4. Reduce V1. Total: {total_alloc:.0%}"
    elif regime.regime == VolatilityRegime.STRESSED:
        status_reason = f"Maximum V3, hold V4. Suppress V1/V2. Total: {total_alloc:.0%}"
    elif regime.regime == VolatilityRegime.DECLINING:
        status_reason = f"Activate V2, restore V1. Total: {total_alloc:.0%}"
    else:
        status_reason = f"Emergency mode - V3 only. Total: {total_alloc:.0%}"
    
    return VariantParams(
        variant_id=variant_id,
        role=VariantRole.ADAPTIVE,
        structure="meta",         # Not a real trade
        entry_percentile=0.0,
        entry_lookback_weeks=52,
        long_strike_offset=0.0,
        short_strike_offset=0.0,
        long_dte_weeks=0,
        short_dte_weeks=0,
        alloc_pct=0.0,            # Doesn't trade directly
        max_contracts=0,
        target_mult=0.0,
        exit_mult=0.0,
        max_hold_weeks=0,
        sigma_mult=1.0,
        tp_pct=0.0,
        sl_pct=0.0,
        active_in_regimes=[r for r in VolatilityRegime],
        suppressed_in_regimes=[],
        status="GUIDANCE",
        status_reason=status_reason,
        generated_at=datetime.utcnow().isoformat(),
        signal_batch_id=batch_id,
    )


def generate_all_variants(
    regime: RegimeState,
    base_config: Dict[str, Any],
    signal_time: Optional[datetime] = None,
    validity_hours: int = 96,  # Valid for 4 days (Thu -> Mon)
) -> SignalBatch:
    """
    Generate all 5 strategy variants for current regime.
    
    Parameters
    ----------
    regime : RegimeState
        Current volatility regime
    base_config : Dict[str, Any]
        Base configuration (capital, fees, etc.)
    signal_time : Optional[datetime]
        Signal generation time (defaults to now)
    validity_hours : int
        Hours until signal expires
    
    Returns
    -------
    SignalBatch
        Complete batch with all variants
    """
    if signal_time is None:
        signal_time = datetime.utcnow()
    
    batch_id = _generate_batch_id(signal_time)
    from datetime import timedelta

    # Replace the manual hour assignment with this:
    base_time = signal_time.replace(minute=0, second=0, microsecond=0)
    valid_until = signal_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Add this temporary line before line 511:
    v1 = generate_income_variant(regime, base_config, batch_id)

    # Generate V1-V4 first
    v1 = generate_income_variant(regime, base_config, batch_id)
    v2 = generate_decay_variant(regime, base_config, batch_id)
    v3 = generate_hedge_variant(regime, base_config, batch_id)
    v4 = generate_convex_variant(regime, base_config, batch_id)
    
    # V5 needs awareness of other variants
    v5 = generate_adaptive_variant(regime, base_config, batch_id, [v1, v2, v3, v4])
    
    variants = [v1, v2, v3, v4, v5]
    
    # Set validity timestamps
    for v in variants:
        v.valid_until = valid_until.isoformat()
    
    return SignalBatch(
        batch_id=batch_id,
        generated_at=signal_time,
        valid_until=valid_until,
        regime_state=regime,
        variants=variants,
        frozen=False,
    )


def get_variant_display_name(role: VariantRole) -> str:
    """Get display name for variant role."""
    names = {
        VariantRole.INCOME: "V1: Income Harvester",
        VariantRole.DECAY: "V2: Decay Accelerator",
        VariantRole.HEDGE: "V3: Shock Absorber",
        VariantRole.CONVEX: "V4: Convex Hunter",
        VariantRole.ADAPTIVE: "V5: Adaptive Controller",
    }
    return names.get(role, role.value)


def get_variant_color(role: VariantRole) -> str:
    """Get display color for variant."""
    colors = {
        VariantRole.INCOME: "#28a745",   # Green
        VariantRole.DECAY: "#17a2b8",    # Cyan
        VariantRole.HEDGE: "#6c757d",    # Gray
        VariantRole.CONVEX: "#dc3545",   # Red
        VariantRole.ADAPTIVE: "#6f42c1", # Purple
    }
    return colors.get(role, "#6c757d")
