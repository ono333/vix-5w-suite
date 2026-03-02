#!/usr/bin/env python3
"""
Trade Log + Position Manager for VIX 5% Weekly Suite

This module tracks:
- All paper trades by variant
- Open positions (long legs)
- Entry/exit prices and P&L
- DTE remaining
- Regime at entry vs current

Key Design:
- Email reads from this to determine MANAGEMENT vs ENTRY mode
- Each variant can have at most ONE open position
- Positions are keyed by variant_id
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Try to import VariantRole, fallback to string if not available
try:
    from enums import VariantRole, VolatilityRegime
except ImportError:
    VariantRole = str
    VolatilityRegime = str


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    ROLLED = "rolled"


@dataclass
class Position:
    """
    Represents a single position (long leg) for a variant.
    """
    position_id: str
    variant_id: str  # e.g., "V1_INCOME_HARVESTER"
    variant_name: str  # e.g., "V1 Income Harvester"
    
    # Entry details
    entry_date: str  # ISO format
    entry_price: float  # Credit received (for short premium) or debit paid (for long)
    entry_regime: str  # Regime at entry
    entry_vix_level: float = 0.0
    entry_percentile: float = 0.0
    
    # Position structure
    underlying: str = "UVXY"
    strike: float = 0.0
    expiration_date: str = ""  # ISO format
    contracts: int = 1
    position_type: str = "diagonal"  # diagonal, long_call, etc.
    
    # Targets (computed at entry)
    target_price: float = 0.0  # Price to close at for profit
    stop_price: float = 0.0  # Price to close at for loss
    target_pct: float = 0.40  # 40% gain target
    stop_pct: float = 0.60  # 60% loss stop
    
    # Current state
    status: str = "open"  # open, closed, expired, rolled
    current_price: float = 0.0
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    
    # Exit details (filled when closed)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # target_hit, stop_hit, expired, manual, rolled
    final_pnl: Optional[float] = None
    
    # Allocation
    allocation_pct: float = 2.0  # % of portfolio
    allocation_dollars: float = 5000.0
    
    # Metadata
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def days_to_expiry(self) -> int:
        """Calculate DTE from today."""
        if not self.expiration_date:
            return 0
        try:
            exp = datetime.fromisoformat(self.expiration_date).date()
            today = date.today()
            return max(0, (exp - today).days)
        except:
            return 0
    
    def is_open(self) -> bool:
        return self.status == "open"
    
    def compute_targets(self) -> None:
        """Compute target and stop prices based on entry price."""
        if self.entry_price > 0:
            # For short premium (credit received)
            # Target: buy back cheaper (price goes down)
            # Stop: buy back more expensive (price goes up)
            self.target_price = self.entry_price * (1 - self.target_pct)
            self.stop_price = self.entry_price * (1 + self.stop_pct)
        else:
            # For long positions (debit paid)
            # Target: sell higher
            # Stop: sell lower
            abs_entry = abs(self.entry_price)
            self.target_price = abs_entry * (1 + self.target_pct)
            self.stop_price = abs_entry * (1 - self.stop_pct)
    
    def update_pnl(self, current_price: float) -> None:
        """Update current P&L based on current price."""
        self.current_price = current_price
        if self.entry_price > 0:
            # Short premium: profit when price drops
            self.current_pnl = (self.entry_price - current_price) * 100 * self.contracts
            self.current_pnl_pct = (self.entry_price - current_price) / self.entry_price
        else:
            # Long position: profit when price rises
            self.current_pnl = (current_price - abs(self.entry_price)) * 100 * self.contracts
            self.current_pnl_pct = (current_price - abs(self.entry_price)) / abs(self.entry_price)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(**data)


@dataclass 
class TradeRecord:
    """Record of a completed trade (for history)."""
    trade_id: str = ""
    variant_id: str = ""
    variant_name: str = ""
    entry_date: str = ""
    exit_date: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    duration_days: int = 0
    exit_reason: str = ""
    entry_regime: str = ""
    exit_regime: str = ""
    contracts: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        return cls(**data)




# ============================================================
# Roll Tracking Classes
# ============================================================

@dataclass
class ShortLeg:
    """Represents a short leg (weekly call) that can be rolled."""
    leg_id: str = ""
    position_id: str = ""  # Parent position ID
    
    # Entry
    entry_date: str = ""
    strike: float = 0.0
    expiration_date: str = ""
    entry_credit: float = 0.0  # Credit received when sold
    contracts: int = 1
    
    # Current state
    status: str = "open"  # open, closed, expired, rolled
    current_price: float = 0.0
    
    # Exit
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None  # Price paid to buy back (or 0 if expired worthless)
    exit_reason: Optional[str] = None  # rolled, expired_worthless, expired_itm, closed_manual
    
    # P&L for this leg
    @property
    def pnl(self) -> float:
        """P&L for this short leg (positive = profit)."""
        if self.status == "open":
            return (self.entry_credit - self.current_price) * 100 * self.contracts
        elif self.exit_price is not None:
            return (self.entry_credit - self.exit_price) * 100 * self.contracts
        else:
            # Expired worthless = full credit kept
            return self.entry_credit * 100 * self.contracts
    
    def days_to_expiry(self) -> int:
        if not self.expiration_date:
            return 0
        try:
            exp = datetime.fromisoformat(self.expiration_date).date()
            return max(0, (exp - date.today()).days)
        except:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pnl"] = self.pnl
        d["dte"] = self.days_to_expiry()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortLeg":
        # Remove computed fields if present
        data.pop("pnl", None)
        data.pop("dte", None)
        return cls(**data)


@dataclass
class RollRecord:
    """Record of a single roll transaction."""
    roll_id: str
    position_id: str
    roll_date: str
    
    # Old leg (closed)
    old_strike: float
    old_expiration: str
    old_exit_price: float  # What we paid to buy back
    
    # New leg (opened)
    new_strike: float
    new_expiration: str
    new_credit: float  # Credit received for new short
    
    # Net
    roll_credit: float  # new_credit - old_exit_price (positive = credit, negative = debit)
    
    # Market context
    underlying_price: float
    contracts: int = 1
    roll_type: str = "short"  # "short" or "long"
    regime:          str   = ""
    vix_level:       float = 0.0
    vix_percentile:  float = 0.0
    notes:           str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollRecord":
        return cls(**data)


@dataclass
class DiagonalPosition:
    """
    A complete diagonal spread position with roll tracking.
    
    Structure:
    - One long leg (LEAP call) - stays open
    - Multiple short legs (weekly calls) - rolled over time
    - Roll history for analysis
    """
    position_id: str
    variant_id: str
    variant_name: str
    
    # Entry context
    entry_date: str = ""
    entry_regime: str = ""
    entry_vix_level: float = 0.0
    entry_percentile: float = 0.0
    contracts: int = 1
    
    # Long leg (LEAP)
    long_strike: float = 0.0
    long_expiration: str = ""
    long_entry_price: float = 0.0  # Debit paid
    long_current_price: float = 0.0
    long_status: str = "open"
    
    # Current short leg
    short_legs: List[ShortLeg] = field(default_factory=list)
    
    # Roll history
    roll_history: List[RollRecord] = field(default_factory=list)
    
    # Aggregated stats
    total_short_credits: float = 0.0  # Sum of all credits received from shorts
    total_roll_credits: float = 0.0   # Sum of all roll net credits
    total_rolls: int = 0
    
    # Commission tracking
    fee_per_contract: float = 0.65  # Default broker fee per contract
    total_commissions: float = 0.0  # Running total of all commissions paid
    
    # Overall position status
    status: str = "open"  # open, closed
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # Targets for the overall position
    target_pct: float = 0.40
    stop_pct: float = 0.60
    
    # Commission tracking
    fee_per_contract: float = 0.65
    total_commissions: float = 0.0
    
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def current_short_leg(self) -> Optional[ShortLeg]:
        """Get the currently active short leg."""
        for leg in self.short_legs:
            if leg.status == "open":
                return leg
        return None
    
    @property
    def net_entry_cost(self) -> float:
        """Net cost to enter (long debit - first short credit)."""
        if self.short_legs:
            return self.long_entry_price - self.short_legs[0].entry_credit
        return self.long_entry_price
    
    @property
    def target_price(self) -> float:
        """Target UVXY price - at or below short strike, short expires worthless."""
        short = self.current_short_leg
        return short.strike if short else self.long_strike
    
    @property
    def stop_price(self) -> float:
        """Stop UVXY price - above this level, position is in danger."""
        short = self.current_short_leg
        base = short.strike if short else self.long_strike
        return base * (1 + self.stop_pct)
    
    @property
    def long_pnl(self) -> float:
        """P&L on long leg."""
        return (self.long_current_price - self.long_entry_price) * 100 * self.contracts
    
    @property
    def short_pnl(self) -> float:
        """Total P&L from all short legs."""
        return sum(leg.pnl for leg in self.short_legs)
    
    @property
    def short_coverage_pct(self) -> float:
        """Percentage of position duration with active short earning income."""
        if not self.entry_date:
            return 0.0
        
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d")
            today = datetime.now()
            total_days = max(1, (today - entry).days)
            
            # Calculate days covered by short legs
            covered_days = 0
            for leg in self.short_legs:
                if leg.entry_date and leg.expiration_date:
                    leg_start = datetime.strptime(leg.entry_date, "%Y-%m-%d")
                    leg_end = datetime.strptime(leg.expiration_date, "%Y-%m-%d")
                    # Don't count beyond today for open legs
                    if leg.status == "open":
                        leg_end = min(leg_end, today)
                    covered_days += max(0, (leg_end - leg_start).days)
            
            return min(100.0, (covered_days / total_days) * 100)
        except:
            return 0.0
    
    @property
    def total_pnl(self) -> float:
        """Combined P&L (long + all shorts - commissions)."""
        return self.long_pnl + self.short_pnl - self.total_commissions
    
    @property
    def long_dte(self) -> int:
        """Days to expiry for long leg."""
        try:
            exp = datetime.strptime(self.long_expiration, "%Y-%m-%d")
            return (exp - datetime.now()).days
        except:
            return 999
    
    @property
    def short_dte(self) -> int:
        """Days to expiry for current short leg."""
        short = self.current_short_leg
        if short:
            return short.days_to_expiry()
        return -1  # No short leg
    
    def get_health_status(self) -> dict:
        """
        Assess position health and return recommended actions.
        
        Returns dict with:
            - status: "healthy", "attention", "critical"
            - long_status: "ok", "roll_soon", "roll_now"
            - short_status: "ok", "roll_soon", "expired", "none"
            - actions: list of recommended actions
            - alerts: list of alert messages
        """
        actions = []
        alerts = []
        
        long_dte = self.long_dte
        short_dte = self.short_dte
        short = self.current_short_leg
        
        # Long leg assessment
        if long_dte <= 30:
            long_status = "roll_now"
            alerts.append(f"⚠️ LONG expiring in {long_dte} days - ROLL IMMEDIATELY")
            actions.append("roll_long")
        elif long_dte <= 60:
            long_status = "roll_soon"
            alerts.append(f"🟡 Long DTE {long_dte} days - plan to roll within 30 days")
            actions.append("plan_roll_long")
        elif long_dte <= 90:
            long_status = "ok"
            alerts.append(f"🟢 Long DTE {long_dte} days - monitor")
        else:
            long_status = "ok"
        
        # Short leg assessment
        if not short or short.status != "open":
            short_status = "none"
            alerts.append("📭 No active short leg - consider selling new short")
            actions.append("sell_new_short")
        elif short_dte <= 0:
            short_status = "expired"
            alerts.append("🎉 Short expired - lock in profit!")
            actions.append("expire_short")
        elif short_dte <= 3:
            short_status = "roll_soon"
            alerts.append(f"🔴 Short expiring in {short_dte} days - roll or expire")
            actions.append("roll_short")
        elif short_dte <= 7:
            short_status = "ok"
            alerts.append(f"🟡 Short DTE {short_dte} days - prepare to roll")
        else:
            short_status = "ok"
        
        # Overall status
        if "roll_now" in [long_status] or short_status == "expired":
            status = "critical"
        elif "roll_soon" in [long_status, short_status] or short_status == "none":
            status = "attention"
        else:
            status = "healthy"
        
        # P&L based actions
        if self.total_pnl > 0 and self.long_entry_price > 0:
            pnl_pct = self.total_pnl / (self.long_entry_price * self.contracts * 100)
            if pnl_pct >= 0.40:  # 40%+ profit
                alerts.append(f"💰 Position up {pnl_pct:.0%} - consider taking profits")
                actions.append("consider_close")
        
        return {
            "status": status,
            "long_status": long_status,
            "short_status": short_status,
            "long_dte": long_dte,
            "short_dte": short_dte,
            "actions": actions,
            "alerts": alerts,
        }
    
    @property
    def total_credits_received(self) -> float:
        """All credits received from shorts + rolls."""
        return self.total_short_credits + self.total_roll_credits
    
    def days_to_long_expiry(self) -> int:
        if not self.long_expiration:
            return 0
        try:
            exp = datetime.fromisoformat(self.long_expiration).date()
            return max(0, (exp - date.today()).days)
        except:
            return 0
    
    def days_to_short_expiry(self) -> int:
        short = self.current_short_leg
        return short.days_to_expiry() if short else 0
    
    def should_roll(self, roll_dte_threshold: int = 3) -> bool:
        """Check if short leg should be rolled."""
        short = self.current_short_leg
        if not short:
            return False  # No short to roll
        return short.days_to_expiry() <= roll_dte_threshold
    
    def add_short_leg(self, strike: float, expiration: str, credit: float, contracts: Optional[int] = None) -> ShortLeg:
        """Add a new short leg. If contracts not specified, uses position's contracts."""
        num_contracts = contracts if contracts else self.contracts
        leg_num = len(self.short_legs) + 1
        leg = ShortLeg(
            leg_id=f"{self.position_id}-S{leg_num}",
            position_id=self.position_id,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            strike=strike,
            expiration_date=expiration,
            entry_credit=credit,
            contracts=num_contracts,
        )
        self.short_legs.append(leg)
        self.total_short_credits += credit * num_contracts
        # Add commission for selling short
        self.total_commissions += self.fee_per_contract * num_contracts
        self.updated_at = datetime.now().isoformat()
        return leg
    
    def roll_short(
        self,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_credit: float,
        underlying_price: float,
        regime: str,
        notes: str = "",
        contracts: Optional[int] = None,
    ) -> Tuple[ShortLeg, RollRecord]:
        """
        Roll the current short leg to a new one.
        Supports partial rolls - if contracts < current short contracts,
        only that many are rolled and the rest stay open.
        
        Returns (new_short_leg, roll_record)
        """
        old_short = self.current_short_leg
        if not old_short:
            raise ValueError("No open short leg to roll")
        
        # Determine contracts to roll
        contracts_to_roll = contracts if contracts else old_short.contracts
        contracts_to_roll = min(contracts_to_roll, old_short.contracts)
        remaining_contracts = old_short.contracts - contracts_to_roll
        
        if remaining_contracts > 0:
            # Partial roll - reduce old short's contracts, keep it open
            old_short.contracts = remaining_contracts
            roll_notes = f"Partial roll: {contracts_to_roll} contracts. {notes}"
        else:
            # Full roll - close old short
            old_short.status = "rolled"
            old_short.exit_date = datetime.now().strftime("%Y-%m-%d")
            old_short.exit_price = exit_price
            old_short.exit_reason = "rolled"
            roll_notes = notes
        
        # Create roll record
        roll_credit = new_credit - exit_price
        roll = RollRecord(
            roll_id=f"{self.position_id}-R{len(self.roll_history) + 1}",
            position_id=self.position_id,
            roll_date=datetime.now().strftime("%Y-%m-%d"),
            old_strike=old_short.strike,
            old_expiration=old_short.expiration_date,
            old_exit_price=exit_price,
            new_strike=new_strike,
            new_expiration=new_expiration,
            new_credit=new_credit,
            roll_credit=roll_credit,
            underlying_price=underlying_price,
            contracts=contracts_to_roll,
            regime=regime,
            notes=roll_notes if 'roll_notes' in dir() else notes,
        )
        self.roll_history.append(roll)
        self.total_rolls += 1
        self.total_roll_credits += roll_credit * contracts_to_roll
        
        # Commission for buying back old short (only for contracts rolled)
        self.total_commissions += self.fee_per_contract * contracts_to_roll
        
        # Add new short leg with the rolled contracts
        new_leg = self.add_short_leg(new_strike, new_expiration, new_credit, contracts=contracts_to_roll)
        
        self.updated_at = datetime.now().isoformat()
        return new_leg, roll
    
    def roll_long(
        self,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_entry_price: float,
        underlying_price: float,
        regime: str,
        notes: str = "",
    ) -> dict:
        """
        Roll the long leg to a new strike/expiration.
        
        This is typically done when:
        - Long leg is approaching expiration (< 60 DTE)
        - Want to adjust strike based on underlying movement
        - Rolling out to capture more time value
        
        Returns dict with roll details and P&L impact.
        """
        # Calculate P&L on old long leg
        old_long_pnl = (exit_price - self.long_entry_price) * 100 * self.contracts
        
        # Net debit/credit for the roll
        roll_debit = new_entry_price - exit_price  # Positive = paid more
        
        roll_info = {
            "roll_date": datetime.now().isoformat(),
            "roll_type": "long",
            "old_strike": self.long_strike,
            "old_expiration": self.long_expiration,
            "old_exit_price": exit_price,
            "new_strike": new_strike,
            "new_expiration": new_expiration,
            "new_entry_price": new_entry_price,
            "roll_debit": roll_debit,
            "old_long_pnl": old_long_pnl,
            "underlying_price": underlying_price,
            "regime": regime,
            "notes": notes,
        }
        
        # Add to roll history
        roll_record = RollRecord(
            roll_id=f"{self.position_id}-RL{len(self.roll_history) + 1}",
            position_id=self.position_id,
            roll_date=datetime.now().strftime("%Y-%m-%d"),
            roll_type="long",
            old_strike=self.long_strike,
            old_expiration=self.long_expiration,
            old_exit_price=exit_price,
            new_strike=new_strike,
            new_expiration=new_expiration,
            new_credit=-roll_debit,  # Negative if we paid more
            roll_credit=-roll_debit,
            underlying_price=underlying_price,
            contracts=self.contracts,
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            notes=notes,
        )
        self.roll_history.append(roll_record)
        
        # Commission for selling old long and buying new long
        self.total_commissions += self.fee_per_contract * self.contracts * 2
        
        # Update position with new long leg
        self.long_strike = new_strike
        self.long_expiration = new_expiration
        self.long_entry_price = new_entry_price
        self.long_current_price = new_entry_price
        
        self.updated_at = datetime.now().isoformat()
        return roll_info

    def recalc_roll_totals(self) -> None:
        """Recalculate roll totals from roll history."""
        total_rolls = 0
        total_roll_credits = 0.0
        for roll in self.roll_history:
            roll_type = getattr(roll, "roll_type", "short") or "short"
            if roll_type == "short":
                total_rolls += 1
                roll_contracts = getattr(roll, "contracts", None)
                if roll_contracts is None:
                    roll_contracts = self.contracts or 1
                total_roll_credits += roll.roll_credit * roll_contracts
        self.total_rolls = total_rolls
        self.total_roll_credits = total_roll_credits
    
    def close_short_expired(self, expired_itm: bool = False, exit_price: float = 0.0) -> None:
        """Mark short as expired (worthless or ITM assignment)."""
        short = self.current_short_leg
        if short:
            short.status = "expired"
            short.exit_date = datetime.now().strftime("%Y-%m-%d")
            short.exit_price = exit_price
            short.exit_reason = "expired_itm" if expired_itm else "expired_worthless"
            self.updated_at = datetime.now().isoformat()
    
    def expire_short_worthless(self):
        """
        Mark current short leg as expired worthless.
        The full credit received becomes realized profit.
        Does NOT roll into a new short - use roll_short() for that.
        """
        short = self.current_short_leg
        if not short:
            return None
        
        short.status = "expired"
        short.exit_date = datetime.now().strftime("%Y-%m-%d")
        short.exit_price = 0.0  # Expired worthless
        short.exit_reason = "expired_worthless"
        
        # No commission for expiration (no action taken)
        self.updated_at = datetime.now().isoformat()
        return short
    
    def close_position(self, long_exit_price: float, short_exit_price: float, reason: str) -> None:
        """Close the entire position."""
        self.long_current_price = long_exit_price
        self.status = "closed"
        self.exit_date = datetime.now().strftime("%Y-%m-%d")
        self.exit_reason = reason
        
        # Close any open short
        short = self.current_short_leg
        if short and short.status == "open":
            short.status = "closed"
            short.exit_date = self.exit_date
            short.exit_price = short_exit_price
            short.exit_reason = "position_closed"
        
        self.updated_at = datetime.now().isoformat()
    
    def update_prices(self, long_price: float, short_price: float) -> None:
        """Update current market prices."""
        self.long_current_price = long_price
        short = self.current_short_leg
        if short:
            short.current_price = short_price
        self.updated_at = datetime.now().isoformat()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get position summary for display."""
        short = self.current_short_leg
        return {
            "position_id": self.position_id,
            "variant": self.variant_name,
            "status": self.status,
            "contracts": self.contracts,
            "long_strike": self.long_strike,
            "long_dte": self.days_to_long_expiry(),
            "long_entry": self.long_entry_price,
            "long_current": self.long_current_price,
            "long_pnl": self.long_pnl,
            "short_strike": short.strike if short else None,
            "short_dte": self.days_to_short_expiry(),
            "short_credit": short.entry_credit if short else 0,
            "short_current": short.current_price if short else 0,
            "short_pnl": self.short_pnl,
            "total_pnl": self.total_pnl,
            "total_rolls": self.total_rolls,
            "total_credits": self.total_credits_received,
            "should_roll": self.should_roll(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested objects
        d["short_legs"] = [leg.to_dict() for leg in self.short_legs]
        d["roll_history"] = [r.to_dict() for r in self.roll_history]
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagonalPosition":
        short_legs_data = data.pop("short_legs", [])
        roll_history_data = data.pop("roll_history", [])
        pos = cls(**data)
        pos.short_legs = [ShortLeg.from_dict(l) for l in short_legs_data]
        pos.roll_history = [RollRecord.from_dict(r) for r in roll_history_data]
        return pos



class TradeLog:
    """
    Manages positions and trade history for all variants.
    
    Key methods for email integration:
    - has_open_position(variant_id) -> bool
    - get_open_position(variant_id) -> Optional[Position]
    - get_all_open_positions() -> List[Position]
    - get_variants_needing_entry() -> List[str]
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize trade log with optional file persistence."""
        if storage_path is None:
            storage_path = os.path.expanduser("~/.vix_suite/trade_log.json")
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Active positions by variant_id
        self.positions: Dict[str, Position] = {}
        
        # Diagonal positions with roll tracking
        self.diagonal_positions: Dict[str, DiagonalPosition] = {}
        
        # Completed trade history
        self.history: List[TradeRecord] = []
        
        # Load from disk
        self._load()
    
    def _load(self) -> None:
        """Load positions and history from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load positions
            positions_data = data.get("positions", {})
            for variant_id, pos_data in positions_data.items():
                self.positions[variant_id] = Position.from_dict(pos_data)
            
            # Load diagonal positions
            diagonal_data = data.get("diagonal_positions", {})
            for pos_id, pos_data in diagonal_data.items():
                self.diagonal_positions[pos_id] = DiagonalPosition.from_dict(pos_data)
            
            # Load history
            history_data = data.get("history", [])
            for record_data in history_data:
                self.history.append(TradeRecord.from_dict(record_data))
                
        except Exception as e:
            print(f"Warning: Could not load trade log: {e}")
    
    def _save(self, skip_backup: bool = False) -> None:
        """
        Persist positions and history to disk.
        
        Auto-backup is performed before each save to prevent data loss.
        Set skip_backup=True only for rapid successive saves.
        """
        # AUTO-BACKUP before saving (prevents data loss)
        if not skip_backup:
            try:
                from backup_manager import get_backup_manager
                backup_mgr = get_backup_manager()
                backup_mgr.backup_before_save()
            except ImportError:
                pass  # Backup manager not available
            except Exception as e:
                print(f"Warning: Backup failed (continuing with save): {e}")
        
        try:
            data = {
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "diagonal_positions": {k: v.to_dict() for k, v in self.diagonal_positions.items()},
                "history": [r.to_dict() for r in self.history],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save trade log: {e}")
    

    # ================================================================
    # Diagonal Position Management (with Roll Tracking)
    # ================================================================
    
    def open_diagonal(
        self,
        variant_id: str,
        variant_name: str,
        contracts: int,
        long_strike: float,
        long_expiration: str,
        long_price: float,
        short_strike: float,
        short_expiration: str,
        short_credit: float,
        entry_regime: str = "CALM",
        entry_vix_level: float = 20.0,
        entry_percentile: float = 0.5,
        target_pct: float = 0.40,
        stop_pct: float = 0.60,
        fee_per_contract: float = 0.65,
        notes: str = "",
    ) -> DiagonalPosition:
        """Open a new diagonal spread position with roll tracking and commission tracking."""
        position_id = f"{variant_id[:2]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        pos = DiagonalPosition(
            position_id=position_id,
            variant_id=variant_id,
            variant_name=variant_name,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            entry_regime=entry_regime,
            entry_vix_level=entry_vix_level,
            entry_percentile=entry_percentile,
            contracts=contracts,
            long_strike=long_strike,
            long_expiration=long_expiration,
            long_entry_price=long_price,
            target_pct=target_pct,
            stop_pct=stop_pct,
            fee_per_contract=fee_per_contract,
            notes=notes,
        )
        
        # Commission for buying long leg
        pos.total_commissions = fee_per_contract * contracts
        
        # Add initial short leg (this also adds commission for selling)
        pos.add_short_leg(short_strike, short_expiration, short_credit)
        
        self.diagonal_positions[position_id] = pos
        self._save()
        return pos
    
    def get_diagonal(self, position_id: str) -> Optional[DiagonalPosition]:
        """Get a diagonal position by ID."""
        return self.diagonal_positions.get(position_id)
    
    def get_all_diagonals(self) -> List[DiagonalPosition]:
        """Get all diagonal positions."""
        return list(self.diagonal_positions.values())
    
    def get_open_diagonals(self) -> List[DiagonalPosition]:
        """Get all open diagonal positions."""
        return [p for p in self.diagonal_positions.values() if p.status == "open"]
    
    def get_diagonals_needing_roll(self, dte_threshold: int = 3) -> List[DiagonalPosition]:
        """Get positions that need rolling (short DTE below threshold)."""
        return [p for p in self.get_open_diagonals() if p.should_roll(dte_threshold)]
    
    def get_diagonals_by_health(self, status: str = None) -> List[DiagonalPosition]:
        """
        Get diagonal positions filtered by health status.
        
        status: "healthy", "attention", "critical", or None for all
        """
        result = []
        for pos in self.get_open_diagonals():
            health = pos.get_health_status()
            if status is None or health["status"] == status:
                result.append(pos)
        return result
    
    def get_diagonals_needing_long_roll(self, dte_threshold: int = 60) -> List[DiagonalPosition]:
        """Get positions where LONG leg needs rolling soon."""
        return [p for p in self.get_open_diagonals() if p.long_dte <= dte_threshold]
    
    def get_diagonals_without_short(self) -> List[DiagonalPosition]:
        """Get positions that have no active short leg (need new short)."""
        result = []
        for pos in self.get_open_diagonals():
            short = pos.current_short_leg
            if not short or short.status != "open":
                result.append(pos)
        return result
    
    def get_position_health_summary(self) -> dict:
        """Get summary of all positions by health status."""
        open_positions = self.get_open_diagonals()
        
        summary = {
            "total": len(open_positions),
            "healthy": 0,
            "attention": 0,
            "critical": 0,
            "need_short_roll": 0,
            "need_long_roll": 0,
            "need_new_short": 0,
            "positions": []
        }
        
        for pos in open_positions:
            health = pos.get_health_status()
            summary[health["status"]] += 1
            
            if "roll_short" in health["actions"] or "expire_short" in health["actions"]:
                summary["need_short_roll"] += 1
            if "roll_long" in health["actions"] or "plan_roll_long" in health["actions"]:
                summary["need_long_roll"] += 1
            if "sell_new_short" in health["actions"]:
                summary["need_new_short"] += 1
            
            summary["positions"].append({
                "position_id": pos.position_id,
                "variant_name": pos.variant_name,
                "health": health
            })
        
        return summary
    
    def roll_diagonal_short(
        self,
        position_id: str,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_credit: float,
        underlying_price: float,
        regime: str,
        vix_level: float = 0.0,
        vix_percentile: float = 0.0,
        notes: str = "",
        contracts: Optional[int] = None,
    ) -> Tuple[Optional[ShortLeg], Optional[RollRecord]]:
        """Roll the short leg of a diagonal position. Supports partial rolls."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None, None
        
        new_leg, roll_record = pos.roll_short(
            exit_price=exit_price,
            new_strike=new_strike,
            new_expiration=new_expiration,
            new_credit=new_credit,
            underlying_price=underlying_price,
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            notes=notes,
            contracts=contracts,
        )
        
        self._save()
        return new_leg, roll_record
    
    def close_short_leg(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "closed_manual",
    ) -> bool:
        """Close the current short leg without opening a new one."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return False
        
        short = pos.current_short_leg
        if not short or short.status != "open":
            return False
        
        short.status = "closed"
        short.exit_date = datetime.now().strftime("%Y-%m-%d")
        short.exit_price = exit_price
        short.exit_reason = exit_reason
        
        self._save()
        return True
    
    def roll_diagonal_long(
        self,
        position_id: str,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_entry_price: float,
        underlying_price: float,
        regime: str,
        notes: str = "",
    ) -> Optional[dict]:
        """Roll the long leg of a diagonal position."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        roll_info = pos.roll_long(
            exit_price=exit_price,
            new_strike=new_strike,
            new_expiration=new_expiration,
            new_entry_price=new_entry_price,
            underlying_price=underlying_price,
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            notes=notes,
        )
        
        self._save()
        return roll_info
    
    def expire_diagonal_short(self, position_id: str) -> Optional[DiagonalPosition]:
        """
        Mark the current short leg as expired worthless.
        Locks in full credit as realized profit without rolling.
        Use this when short expires OTM and you want to keep position open.
        """
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        pos.expire_short_worthless()
        self._save()
        return pos
    
    def close_diagonal(
        self,
        position_id: str,
        long_exit_price: float,
        short_exit_price: float,
        reason: str,
    ) -> Optional[DiagonalPosition]:
        """Close a diagonal position completely."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        pos.close_position(long_exit_price, short_exit_price, reason)
        self._save()
        return pos
    
    def update_diagonal_prices(
        self,
        position_id: str,
        long_price: float,
        short_price: float,
    ) -> Optional[DiagonalPosition]:
        """Update current prices for a diagonal position."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        pos.update_prices(long_price, short_price)
        self._save()
        return pos
    
    
    def delete_diagonal(self, position_id: str) -> bool:
        """Delete a diagonal position completely."""
        if position_id in self.diagonal_positions:
            del self.diagonal_positions[position_id]
            self._save()
            return True
        return False
    
    def update_diagonal(
        self,
        position_id: str,
        **kwargs
    ) -> Optional[DiagonalPosition]:
        """
        Update diagonal position fields.
        
        Allowed fields: variant_name, contracts, long_strike, long_expiration,
        long_entry_price, entry_regime, entry_vix_level, target_pct, stop_pct, notes
        """
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        allowed_fields = {
            'variant_name', 'contracts', 'long_strike', 'long_expiration',
            'long_entry_price', 'entry_regime', 'entry_vix_level', 
            'entry_percentile', 'target_pct', 'stop_pct', 'notes',
            'total_commissions', 'fee_per_contract'
        }
        
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(pos, key):
                setattr(pos, key, value)
        
        pos.updated_at = datetime.now().isoformat()
        self._save()
        return pos
    
    def update_diagonal_short_leg(
        self,
        position_id: str,
        strike: float = None,
        expiration_date: str = None,
        entry_credit: float = None,
    ) -> Optional[ShortLeg]:
        """Update the current short leg details."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        short = pos.current_short_leg
        if not short:
            return None
        
        if strike is not None:
            short.strike = strike
        if expiration_date is not None:
            short.expiration_date = expiration_date
        if entry_credit is not None:
            short.entry_credit = entry_credit
        
        pos.updated_at = datetime.now().isoformat()
        self._save()
        return short

    def update_roll_record(
        self,
        position_id: str,
        roll_id: str,
        **kwargs,
    ) -> Optional[RollRecord]:
        """Update a roll history record and recalc totals."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None
        
        roll = next((r for r in pos.roll_history if r.roll_id == roll_id), None)
        if not roll:
            return None
        
        for key, value in kwargs.items():
            if hasattr(roll, key) and value is not None:
                setattr(roll, key, value)
        
        # Keep roll_credit in sync with old/new prices
        if hasattr(roll, "new_credit") and hasattr(roll, "old_exit_price"):
            roll.roll_credit = roll.new_credit - roll.old_exit_price
        
        pos.recalc_roll_totals()
        pos.updated_at = datetime.now().isoformat()
        self._save()
        return roll

    def delete_roll_record(self, position_id: str, roll_id: str) -> bool:
        """Delete a roll record and recalc totals."""
        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return False
        
        original_count = len(pos.roll_history)
        pos.roll_history = [r for r in pos.roll_history if r.roll_id != roll_id]
        if len(pos.roll_history) == original_count:
            return False
        
        pos.recalc_roll_totals()
        pos.updated_at = datetime.now().isoformat()
        self._save()
        return True


    def get_roll_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all rolls."""
        total_rolls = 0
        total_roll_credits = 0.0
        positions_with_rolls = 0
        
        for pos in self.diagonal_positions.values():
            if pos.total_rolls > 0:
                positions_with_rolls += 1
                total_rolls += pos.total_rolls
                total_roll_credits += pos.total_roll_credits
        
        return {
            "total_rolls": total_rolls,
            "total_roll_credits": total_roll_credits,
            "positions_with_rolls": positions_with_rolls,
            "avg_roll_credit": total_roll_credits / total_rolls if total_rolls > 0 else 0,
        }

    # ================================================================
    # Position Query Methods (for email integration)
    # ================================================================
    
    def has_open_position(self, variant_id: str) -> bool:
        """Check if variant has an open position."""
        pos = self.positions.get(variant_id)
        return pos is not None and pos.is_open()
    
    def get_open_position(self, variant_id: str) -> Optional[Position]:
        """Get open position for variant, or None. Checks both positions and diagonal_positions."""
        # First check regular positions
        pos = self.positions.get(variant_id)
        if pos and pos.is_open():
            return pos
        
        # Then check diagonal positions (case-insensitive match)
        variant_id_upper = variant_id.upper()
        for pid, diag in self.diagonal_positions.items():
            if diag.variant_id.upper() == variant_id_upper and diag.status == "open":
                return self._diagonal_to_position(diag)
        
        return None
    
    def _diagonal_to_position(self, diag) -> Position:
        """Convert DiagonalPosition to Position for signal generator compatibility."""
        # Calculate net debit per spread
        # Use the property that correctly calculates net entry cost
        net_debit = diag.net_entry_cost
        
        # Get current short leg info
        current_short = diag.current_short_leg
        current_short_price = current_short.current_price if current_short else 0
        
        # Use LONG expiration for DTE (that's the main position timeframe)
        # Short leg DTE is for roll decisions, handled separately
        expiration = diag.long_expiration
        
        # Current spread value
        current_spread_value = diag.long_current_price - current_short_price
        
        # Generate a position_id from diagonal
        position_id = f"diag_{diag.position_id}"
        
        # Get P&L directly from diagonal's calculations
        total_pnl = diag.total_pnl
        pnl_pct = (total_pnl / (net_debit * 100 * diag.contracts)) if (net_debit and diag.contracts) else 0
        
        pos = Position(
            position_id=position_id,
            variant_id=diag.variant_id,
            variant_name=diag.variant_name,
            entry_date=diag.entry_date,
            entry_price=net_debit,
            entry_regime=diag.entry_regime,
            entry_vix_level=diag.entry_vix_level,
            entry_percentile=diag.entry_percentile,
            strike=diag.long_strike,
            expiration_date=expiration,
            contracts=diag.contracts,
            target_price=diag.target_price,
            stop_price=diag.stop_price,
            target_pct=diag.target_pct,
            stop_pct=diag.stop_pct,
            status=diag.status,
            current_price=current_spread_value,
            current_pnl=total_pnl,
            current_pnl_pct=pnl_pct,
        )
        
        # Store reference to short leg DTE for roll warnings
        pos._short_dte = diag.short_dte
        pos._diag_ref = diag  # Keep reference for detailed info
        
        return pos
    
    def get_all_open_positions(self) -> List[Position]:
        """Get all currently open positions."""
        return [p for p in self.positions.values() if p.is_open()]
    
    def get_variants_with_open_positions(self) -> List[str]:
        """Get list of variant_ids that have open positions."""
        return [vid for vid, pos in self.positions.items() if pos.is_open()]
    
    def get_variants_needing_entry(self, all_variant_ids: List[str]) -> List[str]:
        """
        Given a list of all variant IDs, return those without open positions.
        These are candidates for new entries.
        """
        open_variants = set(self.get_variants_with_open_positions())
        return [vid for vid in all_variant_ids if vid not in open_variants]
    
    # ================================================================
    # Position Management
    # ================================================================
    
    def open_position(
        self,
        variant_id: str,
        variant_name: str,
        entry_price: float,
        entry_regime: str,
        entry_vix_level: float,
        entry_percentile: float,
        strike: float = 0.0,
        expiration_date: str = "",
        contracts: int = 1,
        allocation_pct: float = 2.0,
        allocation_dollars: float = 5000.0,
        target_pct: float = 0.40,
        stop_pct: float = 0.60,
        position_type: str = "diagonal",
        notes: str = "",
    ) -> Position:
        """
        Open a new position for a variant.
        Raises error if position already exists.
        """
        if self.has_open_position(variant_id):
            raise ValueError(f"Position already exists for {variant_id}")
        
        position_id = f"POS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{variant_id[:3]}"
        
        pos = Position(
            position_id=position_id,
            variant_id=variant_id,
            variant_name=variant_name,
            entry_date=datetime.now().isoformat(),
            entry_price=entry_price,
            entry_regime=entry_regime,
            entry_vix_level=entry_vix_level,
            entry_percentile=entry_percentile,
            strike=strike,
            expiration_date=expiration_date,
            contracts=contracts,
            position_type=position_type,
            allocation_pct=allocation_pct,
            allocation_dollars=allocation_dollars,
            target_pct=target_pct,
            stop_pct=stop_pct,
            notes=notes,
        )
        
        # Compute target/stop prices
        pos.compute_targets()
        
        self.positions[variant_id] = pos
        self._save()
        
        return pos
    
    def close_position(
        self,
        variant_id: str,
        exit_price: float,
        exit_reason: str,
        exit_regime: str = "",
    ) -> Optional[TradeRecord]:
        """
        Close an open position and record to history.
        Returns the trade record, or None if no position existed.
        """
        pos = self.get_open_position(variant_id)
        if pos is None:
            return None
        
        # Calculate final P&L
        if pos.entry_price > 0:
            # Short premium
            final_pnl = (pos.entry_price - exit_price) * 100 * pos.contracts
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        else:
            # Long position
            final_pnl = (exit_price - abs(pos.entry_price)) * 100 * pos.contracts
            pnl_pct = (exit_price - abs(pos.entry_price)) / abs(pos.entry_price)
        
        # Calculate duration
        entry_dt = datetime.fromisoformat(pos.entry_date)
        exit_dt = datetime.now()
        duration_days = (exit_dt - entry_dt).days
        
        # Update position
        pos.status = "closed"
        pos.exit_date = exit_dt.isoformat()
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        pos.final_pnl = final_pnl
        
        # Create trade record
        record = TradeRecord(
            trade_id=f"TRADE-{exit_dt.strftime('%Y%m%d%H%M%S')}",
            variant_id=variant_id,
            variant_name=pos.variant_name,
            entry_date=pos.entry_date,
            exit_date=pos.exit_date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl_dollars=final_pnl,
            pnl_pct=pnl_pct,
            duration_days=duration_days,
            exit_reason=exit_reason,
            entry_regime=pos.entry_regime,
            exit_regime=exit_regime,
            contracts=pos.contracts,
        )
        
        self.history.append(record)
        
        # Remove from active positions
        del self.positions[variant_id]
        
        self._save()
        return record
    
    def update_position_price(self, variant_id: str, current_price: float) -> None:
        """Update current price and P&L for a position."""
        pos = self.get_open_position(variant_id)
        if pos:
            pos.update_pnl(current_price)
            self._save()
    
    # ================================================================
    # Summary & Analytics
    # ================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard display."""
        open_positions = self.get_all_open_positions()
        
        total_pnl = sum(p.current_pnl for p in open_positions)
        
        history_pnl = sum(r.pnl_dollars for r in self.history)
        wins = sum(1 for r in self.history if r.pnl_dollars > 0)
        losses = sum(1 for r in self.history if r.pnl_dollars <= 0)
        
        return {
            # New keys
            "open_positions": len(open_positions),
            "open_pnl": total_pnl,
            "total_trades": len(self.history),
            "total_realized_pnl": history_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(1, wins + losses),
            # Compatibility keys for app.py
            "open_trades": len(open_positions),
            "closed_trades": len(self.history),
            "combined_pnl": history_pnl + total_pnl,
            # Compatibility keys for app.py
            "open_trades": len(open_positions),
            "closed_trades": len(self.history),
            "total_pnl": history_pnl + total_pnl,
            "realized_pnl": history_pnl,
            "unrealized_pnl": total_pnl,
        }
    
    def get_variant_history(self, variant_id: str) -> List[TradeRecord]:
        """Get trade history for a specific variant."""
        return [r for r in self.history if r.variant_id == variant_id]



    # ============================================================
    # Compatibility methods for app.py
    # ============================================================
    
    def get_all_trades(self):
        """Get all trades (open positions + closed history)."""
        trades = []
        for pos in self.positions.values():
            trades.append({
                "trade_id": pos.position_id,
                "variant_id": pos.variant_id,
                "variant_name": pos.variant_name,
                "status": pos.status,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "contracts": pos.contracts,
                "pnl": pos.current_pnl,
                "exit_date": pos.exit_date,
                "exit_price": pos.exit_price,
            })
        for record in self.history:
            trades.append({
                "trade_id": record.trade_id,
                "variant_id": record.variant_id,
                "variant_name": record.variant_name,
                "status": "closed",
                "entry_date": record.entry_date,
                "entry_price": record.entry_price,
                "current_price": record.exit_price,
                "contracts": record.contracts,
                "pnl": record.pnl_dollars,
                "exit_date": record.exit_date,
                "exit_price": record.exit_price,
            })
        return trades
    
    def get_open_trades(self):
        """Get only open trades."""
        return [t for t in self.get_all_trades() if t["status"] == "open"]
    
    def get_closed_trades(self):
        """Get only closed trades."""
        return [t for t in self.get_all_trades() if t["status"] == "closed"]
    
    def get_trades_by_variant(self, variant_id):
        """Get trades for a specific variant."""
        return [t for t in self.get_all_trades() if t["variant_id"] == variant_id]
    
    def create_trade(self, variant_id, variant_name, entry_price, contracts=1, 
                     entry_regime="CALM", entry_vix_level=20.0, entry_percentile=0.5, **kwargs):
        """Create a new trade (alias for open_position)."""
        return self.open_position(
            variant_id=variant_id, variant_name=variant_name,
            entry_price=entry_price, contracts=contracts,
            entry_regime=entry_regime, entry_vix_level=entry_vix_level,
            entry_percentile=entry_percentile, **kwargs
        )
    
    def add_leg(self, trade_id, leg):
        """Add a leg to an existing trade (placeholder)."""
        pass
    
    def save(self):
        """Explicit save."""
        self._save()


# ================================================================
# Singleton instance for app-wide use
# ================================================================

_trade_log_instance: Optional[TradeLog] = None

def get_trade_log() -> TradeLog:
    """Get the global trade log instance."""
    global _trade_log_instance
    if _trade_log_instance is None:
        _trade_log_instance = TradeLog()
    return _trade_log_instance

# ============================================================
# Compatibility aliases for app.py imports
# ============================================================

# Trade is an alias for Position
Trade = Position

# TradeStatus is an alias for PositionStatus
TradeStatus = PositionStatus

# Placeholder classes for leg tracking (not fully implemented yet)
class LegSide:
    LONG = "long"
    SHORT = "short"

class LegStatus:
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"

@dataclass
class TradeLeg:
    """Placeholder for individual leg tracking."""
    leg_id: str = ""
    side: str = "long"
    strike: float = 0.0
    expiration: str = ""
    entry_price: float = 0.0
    current_price: float = 0.0
    status: str = "open"


    # ============================================================
# Singleton instance for app-wide use
# ================================================================

_trade_log_instance: Optional[TradeLog] = None

def get_trade_log() -> TradeLog:
    """Get the global trade log instance."""
    global _trade_log_instance
    if _trade_log_instance is None:
        _trade_log_instance = TradeLog()
    return _trade_log_instance

# ============================================================
# Compatibility aliases for app.py imports
# ============================================================

Trade = Position
TradeStatus = PositionStatus

class LegSide:
    LONG = "long"
    SHORT = "short"

class LegStatus:
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"

@dataclass
class TradeLeg:
    """Placeholder for individual leg tracking."""
    leg_id: str = ""
    side: str = "long"
    strike: float = 0.0
    expiration: str = ""
    entry_price: float = 0.0
    current_price: float = 0.0
    status: str = "open"
