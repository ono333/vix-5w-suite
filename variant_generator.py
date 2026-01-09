"""
Variant Generator for VIX 5% Weekly Suite

FIXED: Always generates ALL 5 variants regardless of regime.
Regime suitability is stored in active_in_regimes field, not used for filtering.

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
    roll_dte_days: int = 3
    
    # Volatility / pricing
    sigma_mult: float = 1.0
    
    # Risk management
    alloc_pct: float = 0.01
    tp_pct: float = 0.20
    sl_pct: float = 0.50
    max_hold_weeks: int = 8
    
    # Regime activation - defines WHERE this variant SHOULD trade
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
            "roll_dte_days": self.roll_dte_days,
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
# Variant Generators (V1-V5) - ALWAYS return a variant
# ============================================================

def _generate_v1_income_harvester(vix_level: float, vix_percentile: float) -> VariantParams:
    """
    V1: Income Harvester - Stable income via diagonal spreads.
    Active in: CALM, DECLINING
    """
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
        roll_dte_days=3,
        sigma_mult=0.8,
        alloc_pct=0.02,
        tp_pct=0.15,
        sl_pct=0.40,
        max_hold_weeks=12,
        active_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.RISING, VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        long_strike=vix_level + 5.0,
        short_strike=vix_level + 2.0,
    )


def _generate_v2_mean_reversion(vix_level: float, vix_percentile: float) -> VariantParams:
    """
    V2: Mean Reversion Accelerator - Post-spike decay capture.
    Active in: DECLINING only
    """
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
        roll_dte_days=3,
        sigma_mult=1.0,
        alloc_pct=0.015,
        tp_pct=0.25,
        sl_pct=0.35,
        max_hold_weeks=8,
        active_in_regimes=[VolatilityRegime.DECLINING],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.RISING, VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        long_strike=vix_level + 8.0,
        short_strike=vix_level + 3.0,
    )


def _generate_v3_shock_absorber(vix_level: float, vix_percentile: float) -> VariantParams:
    """
    V3: Shock Absorber - Crisis hedge, long calls only.
    Active in: RISING, STRESSED, EXTREME
    """
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
        roll_dte_days=0,
        sigma_mult=1.2,
        alloc_pct=0.01,
        tp_pct=0.50,
        sl_pct=0.60,
        max_hold_weeks=6,
        active_in_regimes=[VolatilityRegime.RISING, VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        long_strike=vix_level + 15.0,
        short_strike=0.0,
    )


def _generate_v4_tail_hunter(vix_level: float, vix_percentile: float) -> VariantParams:
    """
    V4: Tail Hunter - Extreme volatility plays.
    Active in: EXTREME only
    """
    return VariantParams(
        variant_id=f"V4-{uuid.uuid4().hex[:8]}",
        name="V4 Tail Hunter",
        role=VariantRole.V4_TAIL_HUNTER,
        entry_percentile=0.90,
        entry_lookback_weeks=52,
        position_type="long_call",
        long_dte_weeks=4,
        short_dte_weeks=0,
        long_strike_offset=20.0,
        short_strike_offset=0.0,
        roll_dte_days=0,
        sigma_mult=1.5,
        alloc_pct=0.005,
        tp_pct=1.00,
        sl_pct=0.80,
        max_hold_weeks=4,
        active_in_regimes=[VolatilityRegime.EXTREME],
        suppressed_in_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING, VolatilityRegime.RISING, VolatilityRegime.STRESSED],
        long_strike=vix_level + 20.0,
        short_strike=0.0,
    )


def _generate_v5_regime_allocator(vix_level: float, vix_percentile: float, regime: VolatilityRegime) -> VariantParams:
    """
    V5: Regime Allocator - Adapts parameters based on regime.
    Active in: ALL regimes (always)
    """
    # Regime-adaptive parameters
    if regime == VolatilityRegime.CALM:
        entry_pct = 0.35
        long_dte = 13
        otm_offset = 10.0
        tp = 0.12
        sl = 0.35
        alloc = 0.025
    elif regime == VolatilityRegime.DECLINING:
        entry_pct = 0.50
        long_dte = 8
        otm_offset = 8.0
        tp = 0.20
        sl = 0.30
        alloc = 0.02
    elif regime == VolatilityRegime.RISING:
        entry_pct = 0.65
        long_dte = 6
        otm_offset = 12.0
        tp = 0.30
        sl = 0.40
        alloc = 0.015
    elif regime == VolatilityRegime.STRESSED:
        entry_pct = 0.80
        long_dte = 4
        otm_offset = 15.0
        tp = 0.40
        sl = 0.50
        alloc = 0.01
    else:  # EXTREME
        entry_pct = 0.90
        long_dte = 4
        otm_offset = 20.0
        tp = 0.60
        sl = 0.60
        alloc = 0.005
    
    return VariantParams(
        variant_id=f"V5-{uuid.uuid4().hex[:8]}",
        name="V5 Regime Allocator",
        role=VariantRole.V5_REGIME_ALLOCATOR,
        entry_percentile=entry_pct,
        entry_lookback_weeks=52,
        position_type="diagonal",
        long_dte_weeks=long_dte,
        short_dte_weeks=1,
        long_strike_offset=otm_offset,
        short_strike_offset=otm_offset * 0.4,
        roll_dte_days=3,
        sigma_mult=1.0,
        alloc_pct=alloc,
        tp_pct=tp,
        sl_pct=sl,
        max_hold_weeks=long_dte,
        # V5 is ALWAYS active - it adapts instead of being suppressed
        active_in_regimes=[
            VolatilityRegime.CALM,
            VolatilityRegime.DECLINING,
            VolatilityRegime.RISING,
            VolatilityRegime.STRESSED,
            VolatilityRegime.EXTREME,
        ],
        suppressed_in_regimes=[],
        long_strike=vix_level + otm_offset,
        short_strike=vix_level + (otm_offset * 0.4),
    )


# ============================================================
# Helper Functions
# ============================================================

def _extract_regime(obj: Any) -> VolatilityRegime:
    """Extract VolatilityRegime from various input types."""
    if isinstance(obj, VolatilityRegime):
        return obj
    if hasattr(obj, 'regime'):
        return obj.regime
    if isinstance(obj, str):
        try:
            return VolatilityRegime(obj.lower())
        except ValueError:
            pass
    return VolatilityRegime.CALM


def _calculate_percentile(series: pd.Series, lookback: int = 52) -> float:
    """Calculate percentile of current value in lookback window."""
    if series is None or len(series) < 2:
        return 0.5
    
    lookback_adj = min(lookback, len(series))
    window = series.tail(lookback_adj)
    current = series.iloc[-1]
    
    return float((window < current).mean())


def _get_vix_level(data: Union[pd.Series, float]) -> float:
    """Extract VIX level from data."""
    if isinstance(data, pd.Series) and len(data) > 0:
        return float(data.iloc[-1])
    elif isinstance(data, (int, float)):
        return float(data)
    return 20.0


# ============================================================
# Main Generator Function - ALWAYS GENERATES ALL 5 VARIANTS
# ============================================================

def generate_all_variants(
    data: Union[pd.Series, "VolatilityRegime", float, Any],
    regime_or_percentile: Optional[Any] = None,
    vix_percentile: Optional[float] = None,
    lookback: int = 52,
) -> SignalBatch:
    """
    Generate ALL 5 variant signals for current market conditions.
    
    IMPORTANT: This function ALWAYS generates all 5 variants.
    Regime suitability is stored in active_in_regimes, NOT used for filtering.
    Filtering happens in the UI/selection layer, not here.
    
    Flexible calling patterns:
    1. generate_all_variants(regime_state) - RegimeState object
    2. generate_all_variants(uvxy_series, regime_state) - app.py pattern
    3. generate_all_variants(regime, vix_level, vix_percentile) - explicit
    4. generate_all_variants(vix_level, regime) - simpler
    """
    from regime_detector import RegimeState
    
    # Handle RegimeState passed directly as first argument
    if hasattr(data, 'regime') and hasattr(data, 'vix_level'):
        regime_state = data
        regime = regime_state.regime
        vix_level = regime_state.vix_level
        pct = regime_state.vix_percentile
    
    # Parse arguments flexibly
    elif isinstance(data, pd.Series):
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
    if regime_state is None or not isinstance(regime_state, RegimeState):
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
    
    # ================================================================
    # ALWAYS GENERATE ALL 5 VARIANTS - NO FILTERING HERE
    # ================================================================
    variants: List[VariantParams] = [
        _generate_v1_income_harvester(vix_level, pct),
        _generate_v2_mean_reversion(vix_level, pct),
        _generate_v3_shock_absorber(vix_level, pct),
        _generate_v4_tail_hunter(vix_level, pct),
        _generate_v5_regime_allocator(vix_level, pct, regime),
    ]
    # ================================================================
    
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
