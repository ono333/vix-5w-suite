"""
Enums for VIX 5% Weekly Suite
"""

from enum import Enum


class VolatilityRegime(Enum):
    """VIX percentile-based regime classification."""
    CALM = "CALM"
    RISING = "RISING"
    STRESSED = "STRESSED"
    DECLINING = "DECLINING"
    EXTREME = "EXTREME"


class VariantRole(Enum):
    """Strategy variant roles."""
    INCOME = "V1_INCOME"
    DECAY = "V2_DECAY"
    HEDGE = "V3_HEDGE"
    CONVEX = "V4_CONVEX"
    ADAPTIVE = "V5_ADAPTIVE"


class TradeStatus(Enum):
    """Trade lifecycle status."""
    SIGNAL = "signal"
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class LegStatus(Enum):
    """Individual leg status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    ROLLED = "rolled"


class LegSide(Enum):
    """Leg side (long/short)."""
    LONG = "long"
    SHORT = "short"


class ExitType(Enum):
    """Types of exit triggers."""
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    REGIME_EXIT = "regime_exit"
    TIME_STOP = "time_stop"
    EXPIRATION = "expiration"
    MANUAL = "manual"
    ROLL = "roll"


class ExitUrgency(Enum):
    """Exit signal urgency levels."""
    IMMEDIATE = "immediate"
    SOON = "soon"
    OPTIONAL = "optional"


class ExitStatus(Enum):
    """Exit event status."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    EXECUTED = "executed"
    DISMISSED = "dismissed"
