"""
Enums for VIX 5% Weekly Suite
"""

from enum import Enum


class VolatilityRegime(Enum):
    """
    VIX percentile-based regime classification.
    
    Percentile ranges (52-week rolling):
        CALM:      0-25%   - Low vol, ideal for income harvesting
        RISING:   25-50%   - Vol picking up, caution
        STRESSED: 50-75%   - Elevated vol, hedge activation
        DECLINING: 75-90%  - Post-spike decay, mean reversion
        EXTREME:  90-100%  - Crisis/spike, tail strategies
    """
    CALM = "CALM"
    RISING = "RISING"
    STRESSED = "STRESSED"
    DECLINING = "DECLINING"
    EXTREME = "EXTREME"


class VariantRole(Enum):
    """
    Strategy variant roles in the portfolio.
    """
    INCOME = "V1_INCOME"      # Income Harvester - stability anchor
    DECAY = "V2_DECAY"        # Mean Reversion Accelerator - post-spike
    HEDGE = "V3_HEDGE"        # Shock Absorber - crisis hedge
    CONVEX = "V4_CONVEX"      # Convex Tail Hunter - rare explosive payoffs
    ADAPTIVE = "V5_ADAPTIVE"  # Regime-Aware Allocator - meta-controller


class TradeStatus(Enum):
    """
    Trade lifecycle status.
    """
    SIGNAL = "signal"         # Generated, not yet executed
    PENDING = "pending"       # Awaiting execution window
    OPEN = "open"             # Position is live
    CLOSING = "closing"       # Exit triggered, awaiting fill
    CLOSED = "closed"         # Fully closed
    EXPIRED = "expired"       # Signal expired without execution
    CANCELLED = "cancelled"   # Manually cancelled


class ExitReason(Enum):
    """
    Reasons for closing a trade.
    """
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    REGIME_EXIT = "regime_exit"
    TIME_STOP = "time_stop"
    MANUAL = "manual"
    EXPIRATION = "expiration"
    ROLL = "roll"
