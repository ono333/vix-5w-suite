"""
Enums for VIX 5% Weekly Suite
"""
from enum import Enum


class VolatilityRegime(Enum):
    """Market volatility regime classification."""
    CALM = "calm"
    RISING = "rising"
    STRESSED = "stressed"
    DECLINING = "declining"
    EXTREME = "extreme"


class VariantRole(Enum):
    """Strategy variant roles."""
    V1_INCOME_HARVESTER = "v1_income_harvester"
    V2_MEAN_REVERSION = "v2_mean_reversion"
    V3_SHOCK_ABSORBER = "v3_shock_absorber"
    V4_TAIL_HUNTER = "v4_tail_hunter"
    V5_REGIME_ALLOCATOR = "v5_regime_allocator"


class TradeStatus(Enum):
    """Trade/position status."""
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    ROLLED = "rolled"


class LegSide(Enum):
    """Option leg side."""
    LONG = "long"
    SHORT = "short"


class LegStatus(Enum):
    """Option leg status."""
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    ROLLED = "rolled"
