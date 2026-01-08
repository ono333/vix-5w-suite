"""
Variant Generator for VIX 5% Weekly Suite

Generates trading signals for 5 strategy variants based on current
volatility regime and market conditions.

Variants:
    V1 (INCOME):   Income Harvester - diagonal spreads in calm markets
    V2 (DECAY):    Mean Reversion Accelerator - post-spike decay capture
    V3 (HEDGE):    Shock Absorber - crisis hedge positions
    V4 (CONVEX):   Convex Tail Hunter - rare explosive payoffs
    V5 (ADAPTIVE): Regime-Aware Allocator - dynamic meta-controller
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# Local imports - NOT from numbered folder
from enums import VolatilityRegime, VariantRole


# =============================================================================
# VARIANT PARAMETERS (for UI/execution)
# =============================================================================

@dataclass
class VariantParams:
    """Parameters for executing a variant signal."""
    variant_role: VariantRole
    variant_name: str
    
    # Position structure
    position_type: str = "diagonal"
    long_dte_weeks: int = 26
    short_dte_weeks: int = 1
    otm_points: float = 5.0
    
    # Sizing
    allocation_pct: float = 2.0
    max_contracts: int = 10
    
    # Risk management
    target_mult: float = 1.20
    stop_mult: float = 0.50
    max_hold_weeks: int = 12
    
    # Entry conditions
    entry_percentile_low: float = 0.0
    entry_percentile_high: float = 0.25
    active_regimes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_role": self.variant_role.value,
            "variant_name": self.variant_name,
            "position_type": self.position_type,
            "long_dte_weeks": self.long_dte_weeks,
            "short_dte_weeks": self.short_dte_weeks,
            "otm_points": self.otm_points,
            "allocation_pct": self.allocation_pct,
            "max_contracts": self.max_contracts,
            "target_mult": self.target_mult,
            "stop_mult": self.stop_mult,
            "max_hold_weeks": self.max_hold_weeks,
            "entry_percentile_low": self.entry_percentile_low,
            "entry_percentile_high": self.entry_percentile_high,
            "active_regimes": self.active_regimes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantParams":
        return cls(
            variant_role=VariantRole(data["variant_role"]),
            variant_name=data["variant_name"],
            position_type=data.get("position_type", "diagonal"),
            long_dte_weeks=data.get("long_dte_weeks", 26),
            short_dte_weeks=data.get("short_dte_weeks", 1),
            otm_points=data.get("otm_points", 5.0),
            allocation_pct=data.get("allocation_pct", 2.0),
            max_contracts=data.get("max_contracts", 10),
            target_mult=data.get("target_mult", 1.20),
            stop_mult=data.get("stop_mult", 0.50),
            max_hold_weeks=data.get("max_hold_weeks", 12),
            entry_percentile_low=data.get("entry_percentile_low", 0.0),
            entry_percentile_high=data.get("entry_percentile_high", 0.25),
            active_regimes=data.get("active_regimes", []),
        )
    
    @classmethod
    def from_config(cls, config: "VariantConfig") -> "VariantParams":
        """Create VariantParams from a VariantConfig."""
        return cls(
            variant_role=config.role,
            variant_name=config.name,
            position_type=config.position_type,
            long_dte_weeks=config.long_dte_weeks,
            short_dte_weeks=config.short_dte_weeks,
            otm_points=config.otm_points,
            allocation_pct=config.max_allocation_pct,
            max_contracts=config.max_contracts,
            target_mult=config.target_mult,
            stop_mult=config.stop_mult,
            max_hold_weeks=config.max_hold_weeks,
            entry_percentile_low=config.entry_percentile_low,
            entry_percentile_high=config.entry_percentile_high,
            active_regimes=[r.value for r in config.active_regimes],
        )


def get_variant_params(role: VariantRole) -> Optional[VariantParams]:
    """Get default parameters for a variant role."""
    config = VARIANT_CONFIGS.get(role)
    if config:
        return VariantParams.from_config(config)
    return None


def get_all_variant_params() -> List[VariantParams]:
    """Get parameters for all variants."""
    return [VariantParams.from_config(c) for c in VARIANT_CONFIGS.values()]


# =============================================================================
# VARIANT CONFIGURATION
# =============================================================================

@dataclass
class VariantConfig:
    """Configuration for a single strategy variant."""
    role: VariantRole
    name: str
    description: str
    active_regimes: List[VolatilityRegime]
    
    # Position parameters
    position_type: str  # "diagonal", "long_call", "put_spread", etc.
    long_dte_weeks: int
    short_dte_weeks: int
    otm_points: float
    
    # Risk parameters
    max_allocation_pct: float  # Max % of equity per trade
    target_mult: float         # Take profit multiple
    stop_mult: float           # Stop loss multiple
    max_contracts: int         # Position size cap
    
    # Entry parameters
    entry_percentile_low: float   # Enter when VIX pct >= this
    entry_percentile_high: float  # Enter when VIX pct <= this
    
    # Time parameters
    max_hold_weeks: int
    signal_valid_hours: int  # How long signal stays valid
    
    # Priority (lower = higher priority when multiple variants active)
    priority: int = 5


# Default variant configurations
VARIANT_CONFIGS: Dict[VariantRole, VariantConfig] = {
    VariantRole.INCOME: VariantConfig(
        role=VariantRole.INCOME,
        name="V1 Income Harvester",
        description="Stability anchor - diagonal spreads harvesting theta in calm markets",
        active_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        position_type="diagonal",
        long_dte_weeks=26,
        short_dte_weeks=1,
        otm_points=5.0,
        max_allocation_pct=2.0,
        target_mult=1.20,
        stop_mult=0.50,
        max_contracts=10,
        entry_percentile_low=0.0,
        entry_percentile_high=0.25,
        max_hold_weeks=12,
        signal_valid_hours=72,
        priority=1,
    ),
    
    VariantRole.DECAY: VariantConfig(
        role=VariantRole.DECAY,
        name="V2 Mean Reversion Accelerator",
        description="Post-spike decay capture - profits from VIX mean reversion",
        active_regimes=[VolatilityRegime.DECLINING],
        position_type="diagonal",
        long_dte_weeks=13,
        short_dte_weeks=1,
        otm_points=3.0,
        max_allocation_pct=3.0,
        target_mult=1.50,
        stop_mult=0.40,
        max_contracts=15,
        entry_percentile_low=0.75,
        entry_percentile_high=0.90,
        max_hold_weeks=8,
        signal_valid_hours=48,
        priority=2,
    ),
    
    VariantRole.HEDGE: VariantConfig(
        role=VariantRole.HEDGE,
        name="V3 Shock Absorber",
        description="Crisis hedge - protection during stressed/extreme conditions",
        active_regimes=[VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        position_type="long_call",
        long_dte_weeks=8,
        short_dte_weeks=0,  # No short leg
        otm_points=10.0,
        max_allocation_pct=1.5,
        target_mult=2.00,
        stop_mult=0.30,
        max_contracts=5,
        entry_percentile_low=0.50,
        entry_percentile_high=1.00,
        max_hold_weeks=6,
        signal_valid_hours=24,
        priority=3,
    ),
    
    VariantRole.CONVEX: VariantConfig(
        role=VariantRole.CONVEX,
        name="V4 Convex Tail Hunter",
        description="Rare explosive payoffs - lottery tickets for extreme moves",
        active_regimes=[VolatilityRegime.EXTREME],
        position_type="long_call",
        long_dte_weeks=4,
        short_dte_weeks=0,
        otm_points=15.0,
        max_allocation_pct=0.5,
        target_mult=5.00,
        stop_mult=0.20,
        max_contracts=3,
        entry_percentile_low=0.90,
        entry_percentile_high=1.00,
        max_hold_weeks=4,
        signal_valid_hours=12,
        priority=4,
    ),
    
    VariantRole.ADAPTIVE: VariantConfig(
        role=VariantRole.ADAPTIVE,
        name="V5 Regime-Aware Allocator",
        description="Meta-controller - dynamic allocation based on regime transitions",
        active_regimes=[
            VolatilityRegime.CALM,
            VolatilityRegime.RISING,
            VolatilityRegime.STRESSED,
            VolatilityRegime.DECLINING,
            VolatilityRegime.EXTREME,
        ],
        position_type="adaptive",
        long_dte_weeks=13,
        short_dte_weeks=1,
        otm_points=5.0,
        max_allocation_pct=1.0,
        target_mult=1.30,
        stop_mult=0.50,
        max_contracts=5,
        entry_percentile_low=0.0,
        entry_percentile_high=1.00,
        max_hold_weeks=8,
        signal_valid_hours=48,
        priority=5,
    ),
}


# =============================================================================
# SIGNAL DATA STRUCTURES
# =============================================================================

@dataclass
class VariantSignal:
    """A trading signal generated by a variant."""
    signal_id: str
    variant_role: VariantRole
    variant_name: str
    
    # Timing
    generated_at: dt.datetime
    valid_until: dt.datetime
    
    # Market context
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    underlying_symbol: str
    
    # Position details
    position_type: str
    direction: str  # "long", "short", "spread"
    
    # Long leg
    long_strike: float
    long_dte_days: int
    long_expiration: dt.date
    long_estimated_price: float
    
    # Short leg (if applicable)
    short_strike: Optional[float] = None
    short_dte_days: Optional[int] = None
    short_expiration: Optional[dt.date] = None
    short_estimated_price: Optional[float] = None
    
    # Sizing
    suggested_contracts: int = 1
    max_contracts: int = 10
    estimated_debit: float = 0.0
    max_loss: float = 0.0
    
    # Risk parameters
    target_mult: float = 1.20
    stop_mult: float = 0.50
    max_hold_weeks: int = 12
    
    # Scoring
    robustness_score: float = 0.0
    confidence: str = "medium"  # "low", "medium", "high"
    
    # Status
    status: str = "pending"  # "pending", "executed", "expired", "cancelled"
    
    # Notes
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.signal_id,
            "variant_role": self.variant_role.value,
            "variant_name": self.variant_name,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "regime": self.regime.value,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "underlying_symbol": self.underlying_symbol,
            "position_type": self.position_type,
            "direction": self.direction,
            "long_strike": self.long_strike,
            "long_dte_days": self.long_dte_days,
            "long_expiration": self.long_expiration.isoformat() if self.long_expiration else None,
            "long_estimated_price": self.long_estimated_price,
            "short_strike": self.short_strike,
            "short_dte_days": self.short_dte_days,
            "short_expiration": self.short_expiration.isoformat() if self.short_expiration else None,
            "short_estimated_price": self.short_estimated_price,
            "suggested_contracts": self.suggested_contracts,
            "max_contracts": self.max_contracts,
            "estimated_debit": self.estimated_debit,
            "max_loss": self.max_loss,
            "target_mult": self.target_mult,
            "stop_mult": self.stop_mult,
            "max_hold_weeks": self.max_hold_weeks,
            "robustness_score": self.robustness_score,
            "confidence": self.confidence,
            "status": self.status,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantSignal":
        """Create from dictionary."""
        return cls(
            signal_id=data["signal_id"],
            variant_role=VariantRole(data["variant_role"]),
            variant_name=data["variant_name"],
            generated_at=dt.datetime.fromisoformat(data["generated_at"]),
            valid_until=dt.datetime.fromisoformat(data["valid_until"]),
            regime=VolatilityRegime(data["regime"]),
            vix_level=data["vix_level"],
            vix_percentile=data["vix_percentile"],
            underlying_symbol=data["underlying_symbol"],
            position_type=data["position_type"],
            direction=data["direction"],
            long_strike=data["long_strike"],
            long_dte_days=data["long_dte_days"],
            long_expiration=dt.date.fromisoformat(data["long_expiration"]) if data.get("long_expiration") else None,
            long_estimated_price=data["long_estimated_price"],
            short_strike=data.get("short_strike"),
            short_dte_days=data.get("short_dte_days"),
            short_expiration=dt.date.fromisoformat(data["short_expiration"]) if data.get("short_expiration") else None,
            short_estimated_price=data.get("short_estimated_price"),
            suggested_contracts=data.get("suggested_contracts", 1),
            max_contracts=data.get("max_contracts", 10),
            estimated_debit=data.get("estimated_debit", 0.0),
            max_loss=data.get("max_loss", 0.0),
            target_mult=data.get("target_mult", 1.20),
            stop_mult=data.get("stop_mult", 0.50),
            max_hold_weeks=data.get("max_hold_weeks", 12),
            robustness_score=data.get("robustness_score", 0.0),
            confidence=data.get("confidence", "medium"),
            status=data.get("status", "pending"),
            notes=data.get("notes", ""),
        )


@dataclass
class SignalBatch:
    """A batch of signals generated together."""
    batch_id: str
    generated_at: dt.datetime
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    underlying_symbol: str
    signals: List[VariantSignal] = field(default_factory=list)
    status: str = "active"  # "active", "frozen", "expired"
    frozen_at: Optional[dt.datetime] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat(),
            "regime": self.regime.value,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "underlying_symbol": self.underlying_symbol,
            "signals": [s.to_dict() for s in self.signals],
            "status": self.status,
            "frozen_at": self.frozen_at.isoformat() if self.frozen_at else None,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalBatch":
        return cls(
            batch_id=data["batch_id"],
            generated_at=dt.datetime.fromisoformat(data["generated_at"]),
            regime=VolatilityRegime(data["regime"]),
            vix_level=data["vix_level"],
            vix_percentile=data["vix_percentile"],
            underlying_symbol=data.get("underlying_symbol", "^VIX"),
            signals=[VariantSignal.from_dict(s) for s in data.get("signals", [])],
            status=data.get("status", "active"),
            frozen_at=dt.datetime.fromisoformat(data["frozen_at"]) if data.get("frozen_at") else None,
            notes=data.get("notes", ""),
        )
    
    def freeze(self) -> None:
        """Freeze the batch for execution."""
        self.status = "frozen"
        self.frozen_at = dt.datetime.now()
    
    def is_valid(self) -> bool:
        """Check if batch is still valid."""
        if self.status == "expired":
            return False
        # Check if any signals are still valid
        now = dt.datetime.now()
        return any(s.valid_until > now for s in self.signals)
    
    def get_active_signals(self) -> List[VariantSignal]:
        """Get signals that are still valid."""
        now = dt.datetime.now()
        return [s for s in self.signals if s.valid_until > now and s.status == "pending"]


# =============================================================================
# VARIANT GENERATOR
# =============================================================================

def generate_signal_id(role: VariantRole, timestamp: dt.datetime) -> str:
    """Generate unique signal ID."""
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{role.value}_{ts_str}"


def calculate_strike(
    spot: float,
    otm_points: float,
    direction: str = "call",
    round_to: float = 0.5,
) -> float:
    """Calculate strike price with rounding."""
    if direction == "call":
        raw_strike = spot + otm_points
    else:
        raw_strike = spot - otm_points
    
    # Round to nearest increment
    return round(raw_strike / round_to) * round_to


def calculate_expiration(
    trade_date: dt.date,
    dte_weeks: int,
) -> dt.date:
    """Calculate expiration date (next Friday after target DTE)."""
    target_date = trade_date + dt.timedelta(weeks=dte_weeks)
    
    # Find next Friday
    days_until_friday = (4 - target_date.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    
    return target_date + dt.timedelta(days=days_until_friday)


def estimate_option_price(
    spot: float,
    strike: float,
    dte_days: int,
    is_call: bool = True,
    vol_estimate: float = 0.80,
) -> float:
    """
    Simple Black-Scholes-ish estimate for option price.
    This is a rough approximation for signal generation only.
    """
    from math import log, sqrt, exp
    from scipy.stats import norm
    
    if dte_days <= 0:
        # At expiration - intrinsic only
        if is_call:
            return max(spot - strike, 0.0)
        else:
            return max(strike - spot, 0.0)
    
    T = dte_days / 365.0
    r = 0.03  # Risk-free rate assumption
    
    try:
        d1 = (log(spot / strike) + (r + 0.5 * vol_estimate ** 2) * T) / (vol_estimate * sqrt(T))
        d2 = d1 - vol_estimate * sqrt(T)
        
        if is_call:
            price = spot * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum price floor
    except Exception:
        # Fallback to simple intrinsic + time value estimate
        if is_call:
            intrinsic = max(spot - strike, 0.0)
        else:
            intrinsic = max(strike - spot, 0.0)
        
        time_value = spot * 0.02 * sqrt(T)  # Rough time value
        return intrinsic + time_value


def generate_variant_signal(
    config: VariantConfig,
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying_symbol: str = "^VIX",
    equity: float = 250000.0,
    signal_time: Optional[dt.datetime] = None,
) -> Optional[VariantSignal]:
    """
    Generate a trading signal for a specific variant.
    
    Returns None if the variant should not be active in the current regime.
    """
    # Check if variant is active in current regime
    if regime not in config.active_regimes:
        return None
    
    # Check if VIX percentile is in entry range
    if not (config.entry_percentile_low <= vix_percentile <= config.entry_percentile_high):
        return None
    
    # Use current time if not provided
    if signal_time is None:
        signal_time = dt.datetime.now()
    
    trade_date = signal_time.date()
    
    # Calculate long leg
    long_strike = calculate_strike(vix_level, config.otm_points, "call")
    long_expiration = calculate_expiration(trade_date, config.long_dte_weeks)
    long_dte_days = (long_expiration - trade_date).days
    long_price = estimate_option_price(vix_level, long_strike, long_dte_days, is_call=True)
    
    # Calculate short leg if applicable
    short_strike = None
    short_expiration = None
    short_dte_days = None
    short_price = None
    
    if config.short_dte_weeks > 0 and config.position_type == "diagonal":
        short_strike = calculate_strike(vix_level, config.otm_points, "call")
        short_expiration = calculate_expiration(trade_date, config.short_dte_weeks)
        short_dte_days = (short_expiration - trade_date).days
        short_price = estimate_option_price(vix_level, short_strike, short_dte_days, is_call=True)
    
    # Calculate position sizing
    allocation = equity * (config.max_allocation_pct / 100.0)
    
    if config.position_type == "diagonal" and short_price:
        # Net debit for diagonal
        net_debit = (long_price - short_price) * 100
    else:
        # Full debit for long only
        net_debit = long_price * 100
    
    if net_debit > 0:
        suggested_contracts = min(
            int(allocation / net_debit),
            config.max_contracts
        )
    else:
        suggested_contracts = 1
    
    suggested_contracts = max(1, suggested_contracts)
    
    # Calculate estimated costs
    estimated_debit = net_debit * suggested_contracts
    max_loss = estimated_debit  # For debit spreads, max loss is debit paid
    
    # Determine direction
    if config.position_type == "diagonal":
        direction = "spread"
    elif config.position_type == "long_call":
        direction = "long"
    else:
        direction = "long"
    
    # Calculate validity window - FIX: ensure valid hour values
    valid_hours = min(config.signal_valid_hours, 168)  # Cap at 1 week
    valid_until = signal_time + dt.timedelta(hours=valid_hours)
    
    # Generate signal
    signal = VariantSignal(
        signal_id=generate_signal_id(config.role, signal_time),
        variant_role=config.role,
        variant_name=config.name,
        generated_at=signal_time,
        valid_until=valid_until,
        regime=regime,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        underlying_symbol=underlying_symbol,
        position_type=config.position_type,
        direction=direction,
        long_strike=long_strike,
        long_dte_days=long_dte_days,
        long_expiration=long_expiration,
        long_estimated_price=long_price,
        short_strike=short_strike,
        short_dte_days=short_dte_days,
        short_expiration=short_expiration,
        short_estimated_price=short_price,
        suggested_contracts=suggested_contracts,
        max_contracts=config.max_contracts,
        estimated_debit=estimated_debit,
        max_loss=max_loss,
        target_mult=config.target_mult,
        stop_mult=config.stop_mult,
        max_hold_weeks=config.max_hold_weeks,
        robustness_score=0.0,  # Will be calculated separately
        confidence="medium",
        status="pending",
        notes=f"Auto-generated for {regime.value} regime",
    )
    
    return signal


def generate_all_variants(
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying_symbol: str = "^VIX",
    equity: float = 250000.0,
    signal_time: Optional[dt.datetime] = None,
    active_roles: Optional[List[VariantRole]] = None,
) -> List[VariantSignal]:
    """
    Generate signals for all active variants in the current regime.
    
    Args:
        regime: Current volatility regime
        vix_level: Current VIX level
        vix_percentile: Current VIX percentile (0-1)
        underlying_symbol: Symbol being traded
        equity: Current account equity
        signal_time: Time of signal generation
        active_roles: Optional filter for specific roles (default: all)
    
    Returns:
        List of VariantSignal objects, sorted by priority
    """
    signals = []
    
    roles_to_check = active_roles if active_roles else list(VARIANT_CONFIGS.keys())
    
    for role in roles_to_check:
        config = VARIANT_CONFIGS.get(role)
        if config is None:
            continue
        
        signal = generate_variant_signal(
            config=config,
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            underlying_symbol=underlying_symbol,
            equity=equity,
            signal_time=signal_time,
        )
        
        if signal is not None:
            signals.append(signal)
    
    # Sort by priority (lower = higher priority)
    signals.sort(key=lambda s: VARIANT_CONFIGS[s.variant_role].priority)
    
    return signals


def get_active_variants_for_regime(regime: VolatilityRegime) -> List[VariantConfig]:
    """Get all variant configurations active in a given regime."""
    return [
        config for config in VARIANT_CONFIGS.values()
        if regime in config.active_regimes
    ]


def get_variant_config(role: VariantRole) -> Optional[VariantConfig]:
    """Get configuration for a specific variant role."""
    return VARIANT_CONFIGS.get(role)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def format_signal_summary(signal: VariantSignal) -> str:
    """Format a signal for display."""
    lines = [
        f"📊 {signal.variant_name}",
        f"   Regime: {signal.regime.value} | VIX: {signal.vix_level:.2f} ({signal.vix_percentile:.0%})",
        f"   Type: {signal.position_type.upper()} | Direction: {signal.direction}",
        f"   Long: {signal.long_strike:.1f} strike, {signal.long_dte_days}d DTE @ ${signal.long_estimated_price:.2f}",
    ]
    
    if signal.short_strike:
        lines.append(
            f"   Short: {signal.short_strike:.1f} strike, {signal.short_dte_days}d DTE @ ${signal.short_estimated_price:.2f}"
        )
    
    lines.extend([
        f"   Contracts: {signal.suggested_contracts} (max {signal.max_contracts})",
        f"   Est. Debit: ${signal.estimated_debit:,.2f} | Max Loss: ${signal.max_loss:,.2f}",
        f"   Target: {signal.target_mult:.0%} | Stop: {signal.stop_mult:.0%}",
        f"   Valid until: {signal.valid_until.strftime('%Y-%m-%d %H:%M')}",
    ])
    
    return "\n".join(lines)


# =============================================================================
# SIGNAL BATCH MANAGEMENT
# =============================================================================

import json
from pathlib import Path

SIGNAL_BATCH_PATH = Path.home() / ".vix_suite" / "current_signal_batch.json"


def generate_signal_batch(
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying_symbol: str = "^VIX",
    equity: float = 250000.0,
    signal_time: Optional[dt.datetime] = None,
) -> SignalBatch:
    """Generate a batch of signals for current market conditions."""
    if signal_time is None:
        signal_time = dt.datetime.now()
    
    # Generate all variant signals
    signals = generate_all_variants(
        regime=regime,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        underlying_symbol=underlying_symbol,
        equity=equity,
        signal_time=signal_time,
    )
    
    batch_id = f"BATCH_{signal_time.strftime('%Y%m%d_%H%M%S')}"
    
    return SignalBatch(
        batch_id=batch_id,
        generated_at=signal_time,
        regime=regime,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        underlying_symbol=underlying_symbol,
        signals=signals,
        status="active",
    )


def save_signal_batch(batch: SignalBatch, path: Optional[Path] = None) -> None:
    """Save signal batch to disk."""
    save_path = path or SIGNAL_BATCH_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(batch.to_dict(), f, indent=2)


def load_signal_batch(path: Optional[Path] = None) -> Optional[SignalBatch]:
    """Load signal batch from disk."""
    load_path = path or SIGNAL_BATCH_PATH
    
    if not load_path.exists():
        return None
    
    try:
        with open(load_path, "r") as f:
            data = json.load(f)
        return SignalBatch.from_dict(data)
    except Exception as e:
        print(f"Error loading signal batch: {e}")
        return None


def get_current_batch() -> Optional[SignalBatch]:
    """Get the current active signal batch."""
    batch = load_signal_batch()
    if batch and batch.is_valid():
        return batch
    return None


def clear_signal_batch(path: Optional[Path] = None) -> None:
    """Clear the current signal batch."""
    clear_path = path or SIGNAL_BATCH_PATH
    if clear_path.exists():
        clear_path.unlink()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing variant generator...")
    
    # Simulate a CALM regime
    signals = generate_all_variants(
        regime=VolatilityRegime.CALM,
        vix_level=14.5,
        vix_percentile=0.15,
        equity=250000.0,
    )
    
    print(f"\nGenerated {len(signals)} signals for CALM regime:")
    for sig in signals:
        print(format_signal_summary(sig))
        print()
    
    # Simulate an EXTREME regime
    signals = generate_all_variants(
        regime=VolatilityRegime.EXTREME,
        vix_level=45.0,
        vix_percentile=0.95,
        equity=250000.0,
    )
    
    print(f"\nGenerated {len(signals)} signals for EXTREME regime:")
    for sig in signals:
        print(format_signal_summary(sig))
        print()
