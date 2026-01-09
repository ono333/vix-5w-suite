"""
Core enums for VIX 5% Weekly Suite
"""
from enum import Enum

class VolatilityRegime(Enum):
    CALM = "calm"
    RISING = "rising"
    STRESSED = "stressed"
    DECLINING = "declining"
    EXTREME = "extreme"

class VariantRole(Enum):
    V1_INCOME_HARVESTER = "v1_income_harvester"
    V2_MEAN_REVERSION = "v2_mean_reversion"
    V3_SHOCK_ABSORBER = "v3_shock_absorber"
    V4_TAIL_HUNTER = "v4_tail_hunter"
    V5_REGIME_ALLOCATOR = "v5_regime_allocator"

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class LegStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    ASSIGNED = "assigned"

class LegSide(Enum):
    LONG = "long"
    SHORT = "short"

class ExitType(Enum):
    TARGET = "target"
    STOP = "stop"
    EXPIRY = "expiry"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"
    TIME_STOP = "time_stop"

class ExitUrgency(Enum):
    IMMEDIATE = "immediate"
    END_OF_DAY = "end_of_day"
    END_OF_WEEK = "end_of_week"
    MONITOR = "monitor"

class ExitStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
