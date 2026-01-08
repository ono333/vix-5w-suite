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

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Re-export VariantRole
from enums import VolatilityRegime, VariantRole


# =============================================================================
# VARIANT DISPLAY HELPERS
# =============================================================================

def get_variant_display_name(role: VariantRole) -> str:
    """Get human-readable name for a variant."""
    names = {
        VariantRole.INCOME: "V1 Income Harvester",
        VariantRole.DECAY: "V2 Mean Reversion Accelerator",
        VariantRole.HEDGE: "V3 Shock Absorber",
        VariantRole.CONVEX: "V4 Convex Tail Hunter",
        VariantRole.ADAPTIVE: "V5 Regime-Aware Allocator",
    }
    return names.get(role, role.value)


def get_variant_color(role: VariantRole) -> str:
    """Get display color for a variant."""
    colors = {
        VariantRole.INCOME: "#27AE60",    # Green
        VariantRole.DECAY: "#3498DB",     # Blue
        VariantRole.HEDGE: "#E67E22",     # Orange
        VariantRole.CONVEX: "#9B59B6",    # Purple
        VariantRole.ADAPTIVE: "#1ABC9C",  # Teal
    }
    return colors.get(role, "#95A5A6")


def get_variant_emoji(role: VariantRole) -> str:
    """Get emoji for a variant."""
    emojis = {
        VariantRole.INCOME: "💰",
        VariantRole.DECAY: "📉",
        VariantRole.HEDGE: "🛡️",
        VariantRole.CONVEX: "🎯",
        VariantRole.ADAPTIVE: "🔄",
    }
    return emojis.get(role, "📊")


# =============================================================================
# VARIANT PARAMETERS
# =============================================================================

@dataclass
class VariantParams:
    """Parameters for a variant strategy."""
    role: VariantRole
    name: str
    description: str = ""
    
    # Active regimes
    active_regimes: List[VolatilityRegime] = field(default_factory=list)
    
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
    entry_pct_low: float = 0.0
    entry_pct_high: float = 0.25
    
    # Priority (lower = higher priority)
    priority: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "name": self.name,
            "description": self.description,
            "active_regimes": [r.value for r in self.active_regimes],
            "position_type": self.position_type,
            "long_dte_weeks": self.long_dte_weeks,
            "short_dte_weeks": self.short_dte_weeks,
            "otm_points": self.otm_points,
            "allocation_pct": self.allocation_pct,
            "max_contracts": self.max_contracts,
            "target_mult": self.target_mult,
            "stop_mult": self.stop_mult,
            "max_hold_weeks": self.max_hold_weeks,
            "entry_pct_low": self.entry_pct_low,
            "entry_pct_high": self.entry_pct_high,
            "priority": self.priority,
        }


# Default variant configurations
VARIANT_CONFIGS: Dict[VariantRole, VariantParams] = {
    VariantRole.INCOME: VariantParams(
        role=VariantRole.INCOME,
        name="V1 Income Harvester",
        description="Stability anchor - diagonal spreads in calm markets",
        active_regimes=[VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        position_type="diagonal",
        long_dte_weeks=26,
        short_dte_weeks=1,
        otm_points=5.0,
        allocation_pct=2.0,
        target_mult=1.20,
        stop_mult=0.50,
        max_contracts=10,
        entry_pct_low=0.0,
        entry_pct_high=0.25,
        priority=1,
    ),
    VariantRole.DECAY: VariantParams(
        role=VariantRole.DECAY,
        name="V2 Mean Reversion Accelerator",
        description="Post-spike decay capture",
        active_regimes=[VolatilityRegime.DECLINING],
        position_type="diagonal",
        long_dte_weeks=13,
        short_dte_weeks=1,
        otm_points=3.0,
        allocation_pct=3.0,
        target_mult=1.50,
        stop_mult=0.40,
        max_contracts=15,
        entry_pct_low=0.75,
        entry_pct_high=0.90,
        priority=2,
    ),
    VariantRole.HEDGE: VariantParams(
        role=VariantRole.HEDGE,
        name="V3 Shock Absorber",
        description="Crisis hedge positions",
        active_regimes=[VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        position_type="long_call",
        long_dte_weeks=8,
        short_dte_weeks=0,
        otm_points=10.0,
        allocation_pct=1.5,
        target_mult=2.00,
        stop_mult=0.30,
        max_contracts=5,
        entry_pct_low=0.50,
        entry_pct_high=1.00,
        priority=3,
    ),
    VariantRole.CONVEX: VariantParams(
        role=VariantRole.CONVEX,
        name="V4 Convex Tail Hunter",
        description="Rare explosive payoffs",
        active_regimes=[VolatilityRegime.EXTREME],
        position_type="long_call",
        long_dte_weeks=4,
        short_dte_weeks=0,
        otm_points=15.0,
        allocation_pct=0.5,
        target_mult=5.00,
        stop_mult=0.20,
        max_contracts=3,
        entry_pct_low=0.90,
        entry_pct_high=1.00,
        priority=4,
    ),
    VariantRole.ADAPTIVE: VariantParams(
        role=VariantRole.ADAPTIVE,
        name="V5 Regime-Aware Allocator",
        description="Dynamic meta-controller",
        active_regimes=list(VolatilityRegime),
        position_type="adaptive",
        long_dte_weeks=13,
        short_dte_weeks=1,
        otm_points=5.0,
        allocation_pct=1.0,
        target_mult=1.30,
        stop_mult=0.50,
        max_contracts=5,
        entry_pct_low=0.0,
        entry_pct_high=1.00,
        priority=5,
    ),
}


# =============================================================================
# VARIANT SIGNAL
# =============================================================================

@dataclass
class VariantSignal:
    """A trading signal from a variant."""
    signal_id: str
    role: VariantRole
    name: str
    generated_at: dt.datetime
    valid_until: dt.datetime
    
    # Market context
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    underlying: str = "^VIX"
    
    # Position details
    position_type: str = "diagonal"
    long_strike: float = 0.0
    long_dte_days: int = 0
    long_expiration: Optional[dt.date] = None
    long_price: float = 0.0
    
    short_strike: Optional[float] = None
    short_dte_days: Optional[int] = None
    short_expiration: Optional[dt.date] = None
    short_price: Optional[float] = None
    
    # Sizing
    contracts: int = 1
    max_contracts: int = 10
    estimated_debit: float = 0.0
    max_loss: float = 0.0
    
    # Risk
    target_mult: float = 1.20
    stop_mult: float = 0.50
    max_hold_weeks: int = 12
    
    # Scoring
    robustness_score: float = 0.0
    confidence: str = "medium"
    
    # Status
    status: str = "pending"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "role": self.role.value,
            "name": self.name,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "regime": self.regime.value,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "underlying": self.underlying,
            "position_type": self.position_type,
            "long_strike": self.long_strike,
            "long_dte_days": self.long_dte_days,
            "long_expiration": self.long_expiration.isoformat() if self.long_expiration else None,
            "long_price": self.long_price,
            "short_strike": self.short_strike,
            "short_dte_days": self.short_dte_days,
            "short_expiration": self.short_expiration.isoformat() if self.short_expiration else None,
            "short_price": self.short_price,
            "contracts": self.contracts,
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
    def from_dict(cls, d: Dict[str, Any]) -> "VariantSignal":
        return cls(
            signal_id=d["signal_id"],
            role=VariantRole(d["role"]),
            name=d["name"],
            generated_at=dt.datetime.fromisoformat(d["generated_at"]),
            valid_until=dt.datetime.fromisoformat(d["valid_until"]),
            regime=VolatilityRegime(d["regime"]),
            vix_level=d["vix_level"],
            vix_percentile=d["vix_percentile"],
            underlying=d.get("underlying", "^VIX"),
            position_type=d.get("position_type", "diagonal"),
            long_strike=d.get("long_strike", 0.0),
            long_dte_days=d.get("long_dte_days", 0),
            long_expiration=dt.date.fromisoformat(d["long_expiration"]) if d.get("long_expiration") else None,
            long_price=d.get("long_price", 0.0),
            short_strike=d.get("short_strike"),
            short_dte_days=d.get("short_dte_days"),
            short_expiration=dt.date.fromisoformat(d["short_expiration"]) if d.get("short_expiration") else None,
            short_price=d.get("short_price"),
            contracts=d.get("contracts", 1),
            max_contracts=d.get("max_contracts", 10),
            estimated_debit=d.get("estimated_debit", 0.0),
            max_loss=d.get("max_loss", 0.0),
            target_mult=d.get("target_mult", 1.20),
            stop_mult=d.get("stop_mult", 0.50),
            max_hold_weeks=d.get("max_hold_weeks", 12),
            robustness_score=d.get("robustness_score", 0.0),
            confidence=d.get("confidence", "medium"),
            status=d.get("status", "pending"),
            notes=d.get("notes", ""),
        )


# =============================================================================
# SIGNAL BATCH
# =============================================================================

@dataclass
class SignalBatch:
    """A batch of signals generated together."""
    batch_id: str
    generated_at: dt.datetime
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    underlying: str = "^VIX"
    signals: List[VariantSignal] = field(default_factory=list)
    status: str = "active"
    frozen_at: Optional[dt.datetime] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat(),
            "regime": self.regime.value,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "underlying": self.underlying,
            "signals": [s.to_dict() for s in self.signals],
            "status": self.status,
            "frozen_at": self.frozen_at.isoformat() if self.frozen_at else None,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalBatch":
        return cls(
            batch_id=d["batch_id"],
            generated_at=dt.datetime.fromisoformat(d["generated_at"]),
            regime=VolatilityRegime(d["regime"]),
            vix_level=d["vix_level"],
            vix_percentile=d["vix_percentile"],
            underlying=d.get("underlying", "^VIX"),
            signals=[VariantSignal.from_dict(s) for s in d.get("signals", [])],
            status=d.get("status", "active"),
            frozen_at=dt.datetime.fromisoformat(d["frozen_at"]) if d.get("frozen_at") else None,
            notes=d.get("notes", ""),
        )
    
    def freeze(self) -> None:
        self.status = "frozen"
        self.frozen_at = dt.datetime.now()
    
    def is_valid(self) -> bool:
        if self.status == "expired":
            return False
        now = dt.datetime.now()
        return any(s.valid_until > now for s in self.signals)
    
    def get_active_signals(self) -> List[VariantSignal]:
        now = dt.datetime.now()
        return [s for s in self.signals if s.valid_until > now and s.status == "pending"]


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def _estimate_option_price(spot: float, strike: float, dte_days: int, vol: float = 0.80) -> float:
    """Simple option price estimate."""
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
    except ImportError:
        # Fallback without scipy
        return max(spot - strike, 0.0) + spot * 0.02 * sqrt(dte_days / 365.0)
    
    if dte_days <= 0:
        return max(spot - strike, 0.0)
    
    T = dte_days / 365.0
    r = 0.03
    
    try:
        d1 = (log(spot / strike) + (r + 0.5 * vol ** 2) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        return spot * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)
    except:
        return max(spot - strike, 0.0) + spot * 0.02 * sqrt(T)


def _next_friday(from_date: dt.date, weeks: int) -> dt.date:
    """Calculate next Friday after N weeks."""
    target = from_date + dt.timedelta(weeks=weeks)
    days_to_friday = (4 - target.weekday()) % 7
    return target + dt.timedelta(days=days_to_friday or 7)


def generate_variant_signal(
    params: VariantParams,
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying: str = "^VIX",
    equity: float = 250000.0,
    signal_time: Optional[dt.datetime] = None,
) -> Optional[VariantSignal]:
    """Generate a signal for a single variant."""
    
    # Check if variant is active in current regime
    if regime not in params.active_regimes:
        return None
    
    # Check percentile entry conditions
    if not (params.entry_pct_low <= vix_percentile <= params.entry_pct_high):
        return None
    
    if signal_time is None:
        signal_time = dt.datetime.now()
    
    trade_date = signal_time.date()
    
    # Calculate strikes and expirations
    long_strike = round((vix_level + params.otm_points) * 2) / 2  # Round to 0.5
    long_expiration = _next_friday(trade_date, params.long_dte_weeks)
    long_dte_days = (long_expiration - trade_date).days
    long_price = _estimate_option_price(vix_level, long_strike, long_dte_days)
    
    short_strike = None
    short_expiration = None
    short_dte_days = None
    short_price = None
    
    if params.short_dte_weeks > 0 and params.position_type == "diagonal":
        short_strike = long_strike
        short_expiration = _next_friday(trade_date, params.short_dte_weeks)
        short_dte_days = (short_expiration - trade_date).days
        short_price = _estimate_option_price(vix_level, short_strike, short_dte_days)
    
    # Calculate sizing
    allocation = equity * (params.allocation_pct / 100.0)
    if params.position_type == "diagonal" and short_price:
        net_debit = (long_price - short_price) * 100
    else:
        net_debit = long_price * 100
    
    contracts = min(int(allocation / net_debit), params.max_contracts) if net_debit > 0 else 1
    contracts = max(1, contracts)
    
    estimated_debit = net_debit * contracts
    
    # Generate signal
    signal_id = f"{params.role.value}_{signal_time.strftime('%Y%m%d_%H%M%S')}"
    valid_hours = 72 if regime == VolatilityRegime.CALM else 24
    valid_until = signal_time + dt.timedelta(hours=valid_hours)
    
    return VariantSignal(
        signal_id=signal_id,
        role=params.role,
        name=params.name,
        generated_at=signal_time,
        valid_until=valid_until,
        regime=regime,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        underlying=underlying,
        position_type=params.position_type,
        long_strike=long_strike,
        long_dte_days=long_dte_days,
        long_expiration=long_expiration,
        long_price=long_price,
        short_strike=short_strike,
        short_dte_days=short_dte_days,
        short_expiration=short_expiration,
        short_price=short_price,
        contracts=contracts,
        max_contracts=params.max_contracts,
        estimated_debit=estimated_debit,
        max_loss=estimated_debit,
        target_mult=params.target_mult,
        stop_mult=params.stop_mult,
        max_hold_weeks=params.max_hold_weeks,
        robustness_score=0.0,
        confidence="medium",
        status="pending",
        notes=f"Auto-generated for {regime.value} regime",
    )


def generate_all_variants(
    regime: VolatilityRegime,
    vix_level: float,
    vix_percentile: float,
    underlying: str = "^VIX",
    equity: float = 250000.0,
    signal_time: Optional[dt.datetime] = None,
) -> List[VariantSignal]:
    """Generate signals for all active variants."""
    signals = []
    
    for params in VARIANT_CONFIGS.values():
        signal = generate_variant_signal(
            params=params,
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            underlying=underlying,
            equity=equity,
            signal_time=signal_time,
        )
        if signal:
            signals.append(signal)
    
    # Sort by priority
    signals.sort(key=lambda s: VARIANT_CONFIGS[s.role].priority)
    
    return signals


# =============================================================================
# BATCH STORAGE
# =============================================================================

BATCH_STORAGE_PATH = Path.home() / ".vix_suite" / "current_signal_batch.json"


def save_signal_batch(batch: SignalBatch, path: Optional[Path] = None) -> None:
    """Save batch to disk."""
    p = path or BATCH_STORAGE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(batch.to_dict(), f, indent=2)


def load_signal_batch(path: Optional[Path] = None) -> Optional[SignalBatch]:
    """Load batch from disk."""
    p = path or BATCH_STORAGE_PATH
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            return SignalBatch.from_dict(json.load(f))
    except:
        return None


def get_current_batch() -> Optional[SignalBatch]:
    """Get current active batch."""
    batch = load_signal_batch()
    if batch and batch.is_valid():
        return batch
    return None
