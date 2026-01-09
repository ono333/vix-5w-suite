"""
Variant Generator for VIX 5% Weekly Suite

Exports:
    - generate_all_variants
    - SignalBatch
    - VariantParams
    - VariantRole (re-exported from enums)
    - get_variant_display_name
    - get_variant_color
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd

from enums import VolatilityRegime, VariantRole


# ============================================================
# Variant Parameters
# ============================================================

@dataclass
class VariantParams:
    """Parameters for a single strategy variant."""
    variant_id: str
    name: str
    role: VariantRole
    
    # Entry conditions
    entry_percentile: float = 0.25
    entry_lookback_weeks: int = 52
    
    # Position structure
    position_type: str = "diagonal"
    long_dte_weeks: int = 26
    short_dte_weeks: int = 1
    long_strike_offset: float = 5.0
    short_strike_offset: float = 2.0
    
    # Volatility / pricing
    sigma_mult: float = 1.0
    
    # Risk management
    alloc_pct: float = 0.01
    tp_pct: float = 0.20
    sl_pct: float = 0.50
    max_hold_weeks: int = 8
    
    # Regime activation
    active_in_regimes: List[VolatilityRegime] = field(default_factory=list)
    suppressed_in_regimes: List[VolatilityRegime] = field(default_factory=list)
    
    # Calculated values (set during generation)
    long_strike: float = 0.0
    short_strike: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "role": self.role.value,
            "entry_percentile": self.entry_percentile,
            "entry_lookback_weeks": self.entry_lookback_weeks,
            "position_type": self.position_type,
            "long_dte_weeks": self.long_dte_weeks,
            "short_dte_weeks": self.short_dte_weeks,
            "long_strike_offset": self.long_strike_offset,
            "short_strike_offset": self.short_strike_offset,
            "sigma_mult": self.sigma_mult,
            "alloc_pct": self.alloc_pct,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "max_hold_weeks": self.max_hold_weeks,
            "active_in_regimes": [r.value for r in self.active_in_regimes],
            "suppressed_in_regimes": [r.value for r in self.suppressed_in_regimes],
            "long_strike": self.long_strike,
            "short_strike": self.short_strike,
        }


# ============================================================
# Signal Batch
# ============================================================

@dataclass
class SignalBatch:
    """A batch of variant signals generated together."""
    batch_id: str
    generated_at: datetime
    valid_until: datetime
    regime_state: Any  # RegimeState
    variants: List[VariantParams] = field(default_factory=list)
    frozen: bool = False
    
    # Alias for compatibility
    @property
    def signals(self) -> List[VariantParams]:
        return self.variants
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        regime_dict = {
            "regime": self.regime_state.regime.value,
            "confidence": getattr(self.regime_state, 'confidence', 0.8),
            "vix_level": self.regime_state.vix_level,
            "vix_percentile": self.regime_state.vix_percentile,
            "vix_slope": getattr(self.regime_state, 'vix_slope', 0.0),
            "term_structure": getattr(self.regime_state, 'term_structure', "normal"),
            "regime_age_days": getattr(self.regime_state, 'regime_age_days', 0),
        }
        
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "regime_state": regime_dict,
            "variants": [v.to_dict() for v in self.variants],
            "frozen": self.frozen,
        }


# ============================================================
# Variant Generators (V1-V5)
# ============================================================

def _generate_v1_income_harvester(vix_level: float, vix_percentile: float) -> VariantParams:
    """V1: Income Harvester - Stable income in CALM/DECLINING regimes."""
    return VariantParams(
        variant_id=f"V1-{uuid.uuid4().hex[:8]}",
        name="V1 Income Harvester",
        role=VariantRole.V1_INCOME_HARVESTER,
        entry_percentile=0.25,
        entry_lookback_weeks=52,
        position_type="diagonal",
        long_dte_weeks=26,
        short_dte_weeks=1,
        long_strike_offset=5.0,
        short_strike_offset=2.0,
        sigma_mult=0.8,
        alloc_pct=0.02,
        tp_pct=0.15,
        sl_pct=0.40,
        max_hold_weeks=12,
        active_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        long_strike=vix_level + 5.0,
        short_strike=vix_level + 2.0,
    )


def _generate_v2_mean_reversion(vix_level: float, vix_percentile: float) -> VariantParams:
    """V2: Mean Reversion Accelerator - Post-spike decay capture."""
    return VariantParams(
        variant_id=f"V2-{uuid.uuid4().hex[:8]}",
        name="V2 Mean Reversion",
        role=VariantRole.V2_MEAN_REVERSION,
        entry_percentile=0.60,
        entry_lookback_weeks=26,
        position_type="diagonal",
        long_dte_weeks=13,
        short_dte_weeks=1,
        long_strike_offset=8.0,
        short_strike_offset=3.0,
        sigma_mult=1.0,
        alloc_pct=0.015,
        tp_pct=0.25,
        sl_pct=0.35,
        max_hold_weeks=8,
        active_in_regimes=[VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.RISING],
        long_strike=vix_level + 8.0,
        short_strike=vix_level + 3.0,
    )


def _generate_v3_shock_absorber(vix_level: float, vix_percentile: float) -> VariantParams:
    """V3: Shock Absorber - Crisis hedge."""
    return VariantParams(
        variant_id=f"V3-{uuid.uuid4().hex[:8]}",
        name="V3 Shock Absorber",
        role=VariantRole.V3_SHOCK_ABSORBER,
        entry_percentile=0.75,
        entry_lookback_weeks=52,
        position_type="long_call",
        long_dte_weeks=8,
        short_dte_weeks=0,
        long_strike_offset=15.0,
        short_strike_offset=0.0,
        sigma_mult=1.2,
        alloc_pct=0.01,
        tp_pct=0.50,
        sl_pct=0.60,
        max_hold_weeks=6,
        active_in_regimes=[VolatilityRegime.STRESSED, VolatilityRegime.EXTREME, VolatilityRegime.RISING],
        suppressed_in_regimes=[VolatilityRegime.CALM],
        long_strike=vix_level + 15.0,
        short_strike=0.0,
    )


def _generate_v4_tail_hunter(vix_level: float, vix_percentile: float) -> VariantParams:
    """V4: Convex Tail Hunter - Rare spike capture."""
    return VariantParams(
        variant_id=f"V4-{uuid.uuid4().hex[:8]}",
        name="V4 Convex Tail Hunter",
        role=VariantRole.V4_TAIL_HUNTER,
        entry_percentile=0.90,
        entry_lookback_weeks=104,
        position_type="long_call",
        long_dte_weeks=4,
        short_dte_weeks=0,
        long_strike_offset=20.0,
        short_strike_offset=0.0,
        sigma_mult=1.5,
        alloc_pct=0.005,
        tp_pct=1.00,
        sl_pct=0.80,
        max_hold_weeks=4,
        active_in_regimes=[VolatilityRegime.EXTREME],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        long_strike=vix_level + 20.0,
        short_strike=0.0,
    )


def _generate_v5_regime_allocator(vix_level: float, vix_percentile: float, regime: VolatilityRegime) -> VariantParams:
    """V5: Regime-Aware Allocator - Meta-controller."""
    # Adjust parameters based on regime
    if regime == VolatilityRegime.CALM:
        alloc = 0.025
        tp = 0.12
        sl = 0.35
    elif regime == VolatilityRegime.DECLINING:
        alloc = 0.02
        tp = 0.18
        sl = 0.40
    elif regime == VolatilityRegime.RISING:
        alloc = 0.01
        tp = 0.25
        sl = 0.45
    elif regime == VolatilityRegime.STRESSED:
        alloc = 0.008
        tp = 0.35
        sl = 0.50
    else:  # EXTREME
        alloc = 0.005
        tp = 0.50
        sl = 0.60
    
    return VariantParams(
        variant_id=f"V5-{uuid.uuid4().hex[:8]}",
        name="V5 Regime-Aware Allocator",
        role=VariantRole.V5_REGIME_ALLOCATOR,
        entry_percentile=0.35,
        entry_lookback_weeks=52,
        position_type="adaptive",
        long_dte_weeks=13,
        short_dte_weeks=1,
        long_strike_offset=10.0,
        short_strike_offset=3.0,
        sigma_mult=1.0,
        alloc_pct=alloc,
        tp_pct=tp,
        sl_pct=sl,
        max_hold_weeks=10,
        active_in_regimes=list(VolatilityRegime),  # Active in all
        suppressed_in_regimes=[],
        long_strike=vix_level + 10.0,
        short_strike=vix_level + 3.0,
    )


# ============================================================
# Helper Functions
# ============================================================

def _extract_regime(regime_input: Any) -> VolatilityRegime:
    """Extract VolatilityRegime from various input types."""
    if isinstance(regime_input, VolatilityRegime):
        return regime_input
    
    # RegimeState object
    if hasattr(regime_input, 'regime'):
        return regime_input.regime
    
    # Dict
    if isinstance(regime_input, dict):
        regime_val = regime_input.get('regime') or regime_input.get('current_regime', 'CALM')
        if isinstance(regime_val, VolatilityRegime):
            return regime_val
        return VolatilityRegime(regime_val.upper())
    
    # String
    if isinstance(regime_input, str):
        return VolatilityRegime(regime_input.upper())
    
    return VolatilityRegime.CALM


def _calculate_percentile(data: Union[pd.Series, float], lookback: int = 52) -> float:
    """Calculate percentile from data."""
    if isinstance(data, pd.Series) and len(data) >= 2:
        current = float(data.iloc[-1])
        window = data.iloc[-lookback:] if len(data) >= lookback else data
        return float((window < current).mean())
    elif isinstance(data, (int, float)):
        return 0.5  # Default if just a scalar
    return 0.5


def _get_vix_level(data: Union[pd.Series, float]) -> float:
    """Extract VIX level from data."""
    if isinstance(data, pd.Series) and len(data) > 0:
        return float(data.iloc[-1])
    elif isinstance(data, (int, float)):
        return float(data)
    return 20.0


# ============================================================
# Main Generator Function
# ============================================================

def generate_all_variants(
    data: Union[pd.Series, "VolatilityRegime", float],
    regime_or_percentile: Optional[Any] = None,
    vix_percentile: Optional[float] = None,
    lookback: int = 52,
) -> SignalBatch:
    """
    Generate all variant signals for current market conditions.
    
    Flexible calling patterns:
    1. generate_all_variants(uvxy_series, regime_state) - app.py pattern
    2. generate_all_variants(regime, vix_level, vix_percentile) - explicit
    3. generate_all_variants(vix_level, regime) - simpler
    """
    from regime_detector import RegimeState
    
    # Parse arguments flexibly
    if isinstance(data, pd.Series):
        # Pattern 1: Series + RegimeState
        vix_level = _get_vix_level(data)
        pct = _calculate_percentile(data, lookback)
        
        if regime_or_percentile is not None:
            regime = _extract_regime(regime_or_percentile)
            # If regime_or_percentile is a RegimeState, use its values
            if hasattr(regime_or_percentile, 'vix_percentile'):
                pct = regime_or_percentile.vix_percentile
            if hasattr(regime_or_percentile, 'vix_level'):
                vix_level = regime_or_percentile.vix_level
            regime_state = regime_or_percentile if isinstance(regime_or_percentile, RegimeState) else None
        else:
            regime = VolatilityRegime.CALM
            regime_state = None
            
    elif isinstance(data, VolatilityRegime):
        # Pattern 2: regime enum first
        regime = data
        vix_level = float(regime_or_percentile) if regime_or_percentile is not None else 20.0
        pct = vix_percentile if vix_percentile is not None else 0.5
        regime_state = None
        
    elif isinstance(data, (int, float)):
        # Pattern 3: vix_level first
        vix_level = float(data)
        regime = _extract_regime(regime_or_percentile) if regime_or_percentile else VolatilityRegime.CALM
        pct = vix_percentile if vix_percentile is not None else 0.5
        regime_state = None
    else:
        # Fallback
        vix_level = 20.0
        pct = 0.5
        regime = VolatilityRegime.CALM
        regime_state = None
    
    # Create RegimeState if we don't have one
    if regime_state is None:
        regime_state = RegimeState(
            regime=regime,
            vix_level=vix_level,
            vix_percentile=pct,
            confidence=0.7,
            vix_slope=0.0,
            term_structure="normal",
            regime_age_days=0,
        )
    
    # Generate batch ID and timing
    batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now()
    
    # Valid until next Thursday 4:30 PM (or Monday if generated Thursday+)
    days_until_thursday = (3 - now.weekday()) % 7
    if days_until_thursday == 0 and now.hour >= 16:
        days_until_thursday = 7
    valid_until = (now + timedelta(days=days_until_thursday)).replace(hour=16, minute=30, second=0, microsecond=0)
    
    # Generate variants based on regime
    variants: List[VariantParams] = []
    
    # V1: Income Harvester - active in CALM, DECLINING
    if regime in [VolatilityRegime.CALM, VolatilityRegime.DECLINING]:
        variants.append(_generate_v1_income_harvester(vix_level, pct))
    
    # V2: Mean Reversion - active in DECLINING
    if regime == VolatilityRegime.DECLINING:
        variants.append(_generate_v2_mean_reversion(vix_level, pct))
    
    # V3: Shock Absorber - active in RISING, STRESSED, EXTREME
    if regime in [VolatilityRegime.RISING, VolatilityRegime.STRESSED, VolatilityRegime.EXTREME]:
        variants.append(_generate_v3_shock_absorber(vix_level, pct))
    
    # V4: Tail Hunter - active in EXTREME only
    if regime == VolatilityRegime.EXTREME:
        variants.append(_generate_v4_tail_hunter(vix_level, pct))
    
    # V5: Regime Allocator - always active
    variants.append(_generate_v5_regime_allocator(vix_level, pct, regime))
    
    return SignalBatch(
        batch_id=batch_id,
        generated_at=now,
        valid_until=valid_until,
        regime_state=regime_state,
        variants=variants,
        frozen=False,
    )


# ============================================================
# Display Helpers
# ============================================================

def get_variant_display_name(role: VariantRole) -> str:
    """Get display name for variant role."""
    names = {
        VariantRole.V1_INCOME_HARVESTER: "V1 Income Harvester",
        VariantRole.V2_MEAN_REVERSION: "V2 Mean Reversion",
        VariantRole.V3_SHOCK_ABSORBER: "V3 Shock Absorber",
        VariantRole.V4_TAIL_HUNTER: "V4 Tail Hunter",
        VariantRole.V5_REGIME_ALLOCATOR: "V5 Regime Allocator",
    }
    return names.get(role, str(role))


def get_variant_color(role: VariantRole) -> str:
    """Get color for variant role."""
    colors = {
        VariantRole.V1_INCOME_HARVESTER: "#4CAF50",  # Green
        VariantRole.V2_MEAN_REVERSION: "#2196F3",    # Blue
        VariantRole.V3_SHOCK_ABSORBER: "#FF9800",    # Orange
        VariantRole.V4_TAIL_HUNTER: "#F44336",       # Red
        VariantRole.V5_REGIME_ALLOCATOR: "#9C27B0",  # Purple
    }
    return colors.get(role, "#757575")
