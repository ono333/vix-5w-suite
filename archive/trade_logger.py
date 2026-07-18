#!/usr/bin/env python3
"""
Trade Logging Module for VIX 5% Weekly Suite

Features:
    - Leg-level tracking (not position-level)
    - Suggested TP/SL levels
    - Partial close support
    - Status tracking (OPEN, CLOSED, PARTIAL)
    - PnL computation (realized and unrealized)
    - Execution metadata

Trade logs are persisted to JSON for durability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from regime_detector import VolatilityRegime


class LegSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class LegStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    EXPIRED = "EXPIRED"


class ExitReason(Enum):
    """
    Exit reasons for trade legs.
    
    NOTE: These are simplified reasons for the trade log.
    For detailed post-mortem analysis, use ExitType from post_mortem.py
    
    Priority Order (regime > structure > planned > loss):
    """
    # REGIME (Priority 1) - Closed due to regime change
    REGIME_CHANGE = "REGIME_CHANGE"
    
    # STRUCTURE (Priority 2) - Closed due to structure invalidation
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    
    # PLANNED (Priority 3) - Expected/designed exits
    TP_CONDITION = "TP_CONDITION"      # Profit condition met (not price)
    TIME_STOP = "TIME_STOP"            # Max hold weeks reached
    EXPIRATION = "EXPIRATION"          # DTE hit
    EXPIRE_BY_DESIGN = "EXPIRE_BY_DESIGN"  # Let option expire worthless
    MANUAL = "MANUAL"                  # Operator discretion
    
    # LOSS (Priority 4, soft/tertiary)
    SL_CONDITION = "SL_CONDITION"      # Loss condition met (not price)
    CATASTROPHIC = "CATASTROPHIC"      # Emergency stop
    
    @property
    def priority(self) -> int:
        """Get exit priority (lower = higher priority)."""
        priority_map = {
            "REGIME_CHANGE": 1,
            "STRUCTURE_BREAK": 2,
            "TP_CONDITION": 3,
            "TIME_STOP": 3,
            "EXPIRATION": 3,
            "EXPIRE_BY_DESIGN": 3,
            "MANUAL": 3,
            "SL_CONDITION": 4,
            "CATASTROPHIC": 4,
        }
        return priority_map.get(self.value, 99)
    
    @property
    def is_soft_stop(self) -> bool:
        """Is this a soft (tertiary) stop that should be marked clearly?"""
        return self.value in ["SL_CONDITION", "CATASTROPHIC"]
    
    @property
    def display_label(self) -> str:
        """Human-readable label for UI."""
        labels = {
            "REGIME_CHANGE": "🔴 Regime Change (Priority 1)",
            "STRUCTURE_BREAK": "🟠 Structure Break (Priority 2)",
            "TP_CONDITION": "🟢 Take Profit Condition",
            "TIME_STOP": "⏰ Time Stop",
            "EXPIRATION": "📅 Expiration",
            "EXPIRE_BY_DESIGN": "✓ Expired by Design",
            "MANUAL": "👤 Manual Discretion",
            "SL_CONDITION": "⚠️ Loss Condition (Soft)",
            "CATASTROPHIC": "🚨 Catastrophic Stop (Soft)",
        }
        return labels.get(self.value, self.value)


@dataclass
class TradeLeg:
    """
    Individual leg of a trade (long or short).
    
    IMPORTANT: TP/SL Design Philosophy
    ----------------------------------
    The tp_price and sl_price fields are DIAGNOSTIC REFERENCES ONLY.
    They are NOT price triggers and should NOT be used to place broker orders.
    
    Exit decisions are condition-based (regime > structure > loss), not price-based:
    - Regime exits (Priority 1): Closed due to regime change
    - Structure exits (Priority 2): Closed due to structure invalidation
    - Planned exits (Priority 3): Time stop, expiration, profit condition
    - Loss exits (Priority 4, soft): Should be rare, clearly marked
    
    Price levels are shown for:
    - Visual reference on charts
    - Post-mortem analysis
    - Robustness score calculation
    """
    leg_id: str
    trade_id: str  # parent trade grouping
    variant_id: str
    
    side: LegSide
    instrument: str  # e.g., "UVXY_250221C25"
    
    # Entry details
    entry_datetime: datetime
    entry_price: float
    entry_size: int  # positive for long, negative for short
    
    # Exit details (nullable until closed)
    exit_datetime: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_size: Optional[int] = None
    exit_reason: Optional[ExitReason] = None  # Classify the exit
    
    # DIAGNOSTIC REFERENCE LEVELS (NOT TRIGGERS)
    # These are for visual reference and post-mortem analysis only
    tp_price: Optional[float] = None  # Reference only - not a broker order
    sl_price: Optional[float] = None  # Reference only - not a broker order
    
    # Condition-based exit rules (what actually triggers exits)
    exit_conditions: List[str] = field(default_factory=list)  # e.g., ["regime_change", "dte_warning"]
    
    # Execution tracking (for post-mortem)
    signal_mid_price: Optional[float] = None  # What generator suggested
    signal_bid: Optional[float] = None
    signal_ask: Optional[float] = None
    intended_fill_price: Optional[float] = None  # What operator planned
    
    # Status
    status: LegStatus = LegStatus.OPEN
    size_remaining: Optional[int] = None  # for partial closes
    
    # Metadata
    strike: Optional[float] = None
    expiration: Optional[datetime] = None
    dte_at_entry: Optional[int] = None
    regime_at_entry: Optional[VolatilityRegime] = None
    
    notes: str = ""
    
    def __post_init__(self):
        if self.size_remaining is None:
            self.size_remaining = abs(self.entry_size)
    
    @property
    def is_long(self) -> bool:
        return self.side == LegSide.LONG
    
    @property
    def is_open(self) -> bool:
        return self.status in [LegStatus.OPEN, LegStatus.PARTIAL]
    
    @property
    def unrealized_pnl(self) -> float:
        """Compute unrealized PnL (requires current price to be set externally)."""
        # This would need current market price - placeholder
        return 0.0
    
    @property
    def realized_pnl(self) -> float:
        """Compute realized PnL from closed portion."""
        if self.exit_price is None or self.exit_size is None:
            return 0.0
        
        if self.is_long:
            return (self.exit_price - self.entry_price) * abs(self.exit_size) * 100
        else:
            return (self.entry_price - self.exit_price) * abs(self.exit_size) * 100
    
    def close(
        self,
        exit_price: float,
        exit_datetime: datetime,
        exit_size: Optional[int] = None,
        reason: ExitReason = ExitReason.MANUAL,
        notes: str = "",
    ) -> 'TradeLeg':
        """
        Close this leg (fully or partially).
        Returns self for chaining.
        """
        if exit_size is None:
            exit_size = self.size_remaining
        
        if exit_size >= self.size_remaining:
            # Full close
            self.exit_datetime = exit_datetime
            self.exit_price = exit_price
            self.exit_size = self.size_remaining
            self.size_remaining = 0
            self.status = LegStatus.CLOSED
        else:
            # Partial close
            self.exit_datetime = exit_datetime
            self.exit_price = exit_price
            self.exit_size = (self.exit_size or 0) + exit_size
            self.size_remaining -= exit_size
            self.status = LegStatus.PARTIAL
        
        if notes:
            self.notes = f"{self.notes} | {reason.value}: {notes}".strip(" |")
        else:
            self.notes = f"{self.notes} | {reason.value}".strip(" |")
        
        # Store the exit reason for post-mortem
        self.exit_reason = reason
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "trade_id": self.trade_id,
            "variant_id": self.variant_id,
            "side": self.side.value,
            "instrument": self.instrument,
            "entry_datetime": self.entry_datetime.isoformat(),
            "entry_price": self.entry_price,
            "entry_size": self.entry_size,
            "exit_datetime": self.exit_datetime.isoformat() if self.exit_datetime else None,
            "exit_price": self.exit_price,
            "exit_size": self.exit_size,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            # Diagnostic reference levels (NOT triggers)
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            # Condition-based exits
            "exit_conditions": self.exit_conditions,
            # Execution tracking
            "signal_mid_price": self.signal_mid_price,
            "signal_bid": self.signal_bid,
            "signal_ask": self.signal_ask,
            "intended_fill_price": self.intended_fill_price,
            # Status
            "status": self.status.value,
            "size_remaining": self.size_remaining,
            "strike": self.strike,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "dte_at_entry": self.dte_at_entry,
            "regime_at_entry": self.regime_at_entry.value if self.regime_at_entry else None,
            "notes": self.notes,
            "realized_pnl": self.realized_pnl,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TradeLeg':
        return cls(
            leg_id=d["leg_id"],
            trade_id=d["trade_id"],
            variant_id=d["variant_id"],
            side=LegSide(d["side"]),
            instrument=d["instrument"],
            entry_datetime=datetime.fromisoformat(d["entry_datetime"]),
            entry_price=d["entry_price"],
            entry_size=d["entry_size"],
            exit_datetime=datetime.fromisoformat(d["exit_datetime"]) if d.get("exit_datetime") else None,
            exit_price=d.get("exit_price"),
            exit_size=d.get("exit_size"),
            exit_reason=ExitReason(d["exit_reason"]) if d.get("exit_reason") else None,
            tp_price=d.get("tp_price"),
            sl_price=d.get("sl_price"),
            exit_conditions=d.get("exit_conditions", []),
            signal_mid_price=d.get("signal_mid_price"),
            signal_bid=d.get("signal_bid"),
            signal_ask=d.get("signal_ask"),
            intended_fill_price=d.get("intended_fill_price"),
            status=LegStatus(d.get("status", "OPEN")),
            size_remaining=d.get("size_remaining"),
            strike=d.get("strike"),
            expiration=datetime.fromisoformat(d["expiration"]) if d.get("expiration") else None,
            dte_at_entry=d.get("dte_at_entry"),
            regime_at_entry=VolatilityRegime(d["regime_at_entry"]) if d.get("regime_at_entry") else None,
            notes=d.get("notes", ""),
        )


@dataclass
class Trade:
    """
    A complete trade consisting of one or more legs.
    
    For diagonal spreads: long leg + short leg
    For long-only: single long leg
    """
    trade_id: str
    variant_id: str
    signal_batch_id: str
    
    legs: List[TradeLeg] = field(default_factory=list)
    
    # Trade-level metadata
    structure_type: str = "diagonal"
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    regime_at_open: Optional[VolatilityRegime] = None
    
    # Suggested management
    max_hold_weeks: int = 26
    
    # Execution metadata
    execution_delay_hours: Optional[float] = None
    slippage_estimate: Optional[float] = None
    
    notes: str = ""
    
    @property
    def is_open(self) -> bool:
        return any(leg.is_open for leg in self.legs)
    
    @property
    def long_leg(self) -> Optional[TradeLeg]:
        for leg in self.legs:
            if leg.side == LegSide.LONG:
                return leg
        return None
    
    @property
    def short_leg(self) -> Optional[TradeLeg]:
        for leg in self.legs:
            if leg.side == LegSide.SHORT:
                return leg
        return None
    
    @property
    def total_realized_pnl(self) -> float:
        return sum(leg.realized_pnl for leg in self.legs)
    
    @property
    def net_premium_paid(self) -> float:
        """Net debit (positive) or credit (negative) at entry."""
        total = 0.0
        for leg in self.legs:
            if leg.is_long:
                total += leg.entry_price * abs(leg.entry_size) * 100
            else:
                total -= leg.entry_price * abs(leg.entry_size) * 100
        return total
    
    @property
    def weeks_held(self) -> int:
        if not self.opened_at:
            return 0
        end = self.closed_at or datetime.utcnow()
        return (end - self.opened_at).days // 7
    
    def add_leg(self, leg: TradeLeg) -> None:
        self.legs.append(leg)
        if self.opened_at is None:
            self.opened_at = leg.entry_datetime
    
    def close_all(
        self,
        exit_prices: Dict[str, float],  # leg_id -> price
        exit_datetime: datetime,
        reason: ExitReason = ExitReason.MANUAL,
    ) -> None:
        """Close all open legs."""
        for leg in self.legs:
            if leg.is_open and leg.leg_id in exit_prices:
                leg.close(
                    exit_price=exit_prices[leg.leg_id],
                    exit_datetime=exit_datetime,
                    reason=reason,
                )
        
        if not self.is_open:
            self.closed_at = exit_datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "variant_id": self.variant_id,
            "signal_batch_id": self.signal_batch_id,
            "legs": [leg.to_dict() for leg in self.legs],
            "structure_type": self.structure_type,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "regime_at_open": self.regime_at_open.value if self.regime_at_open else None,
            "max_hold_weeks": self.max_hold_weeks,
            "execution_delay_hours": self.execution_delay_hours,
            "slippage_estimate": self.slippage_estimate,
            "notes": self.notes,
            "is_open": self.is_open,
            "total_realized_pnl": self.total_realized_pnl,
            "net_premium_paid": self.net_premium_paid,
            "weeks_held": self.weeks_held,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Trade':
        trade = cls(
            trade_id=d["trade_id"],
            variant_id=d["variant_id"],
            signal_batch_id=d["signal_batch_id"],
            structure_type=d.get("structure_type", "diagonal"),
            opened_at=datetime.fromisoformat(d["opened_at"]) if d.get("opened_at") else None,
            closed_at=datetime.fromisoformat(d["closed_at"]) if d.get("closed_at") else None,
            regime_at_open=VolatilityRegime(d["regime_at_open"]) if d.get("regime_at_open") else None,
            max_hold_weeks=d.get("max_hold_weeks", 26),
            execution_delay_hours=d.get("execution_delay_hours"),
            slippage_estimate=d.get("slippage_estimate"),
            notes=d.get("notes", ""),
        )
        trade.legs = [TradeLeg.from_dict(leg) for leg in d.get("legs", [])]
        return trade


def compute_suggested_tp_sl(
    entry_price: float,
    is_long: bool,
    tp_pct: float = 0.50,  # 50% profit target
    sl_pct: float = 0.30,  # 30% stop loss
) -> tuple[float, float]:
    """
    Compute DIAGNOSTIC REFERENCE levels for TP and SL.
    
    ⚠️ IMPORTANT: These are NOT triggers and should NOT be used to place broker orders.
    
    These levels are for:
    - Visual reference on charts
    - Post-mortem analysis
    - Robustness score calculation
    
    Exit decisions should be condition-based (regime > structure > loss), not price-based.
    See ExitReason enum for the proper exit taxonomy.
    
    For Long Positions:
        TP = Entry * (1 + tp_pct)
        SL = Entry * (1 - sl_pct)
    
    For Short Positions:
        TP = Entry * (1 - tp_pct)
        SL = Entry * (1 + sl_pct)
    
    Returns:
        tuple[float, float]: (tp_reference, sl_reference) - DIAGNOSTIC ONLY
    """
    if is_long:
        tp = entry_price * (1 + tp_pct)
        sl = entry_price * (1 - sl_pct)
    else:
        tp = entry_price * (1 - tp_pct)
        sl = entry_price * (1 + sl_pct)
    
    return round(tp, 2), round(sl, 2)


# --- Trade Log Manager ---

class TradeLogManager:
    """
    Manages trade log persistence and operations.
    """
    
    def __init__(self, log_path: Path = Path("trade_log.json")):
        self.log_path = log_path
        self.trades: Dict[str, Trade] = {}
        self._load()
    
    def _load(self) -> None:
        """Load trades from disk."""
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                for trade_dict in data.get("trades", []):
                    trade = Trade.from_dict(trade_dict)
                    self.trades[trade.trade_id] = trade
            except Exception as e:
                print(f"Warning: Could not load trade log: {e}")
    
    def _save(self) -> None:
        """Save trades to disk."""
        data = {
            "trades": [trade.to_dict() for trade in self.trades.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_trade(
        self,
        variant_id: str,
        signal_batch_id: str,
        structure_type: str = "diagonal",
        regime: Optional[VolatilityRegime] = None,
        max_hold_weeks: int = 26,
    ) -> Trade:
        """Create a new trade."""
        trade_id = f"TRD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        
        trade = Trade(
            trade_id=trade_id,
            variant_id=variant_id,
            signal_batch_id=signal_batch_id,
            structure_type=structure_type,
            regime_at_open=regime,
            max_hold_weeks=max_hold_weeks,
        )
        
        self.trades[trade_id] = trade
        self._save()
        return trade
    
    def add_leg_to_trade(
        self,
        trade_id: str,
        instrument: str,
        side: LegSide,
        entry_price: float,
        entry_size: int,
        entry_datetime: Optional[datetime] = None,
        strike: Optional[float] = None,
        expiration: Optional[datetime] = None,
        regime: Optional[VolatilityRegime] = None,
        tp_pct: float = 0.50,
        sl_pct: float = 0.30,
    ) -> TradeLeg:
        """Add a leg to an existing trade."""
        if trade_id not in self.trades:
            raise ValueError(f"Trade {trade_id} not found")
        
        trade = self.trades[trade_id]
        
        if entry_datetime is None:
            entry_datetime = datetime.utcnow()
        
        leg_id = f"LEG-{uuid.uuid4().hex[:8]}"
        
        # Compute suggested TP/SL
        is_long = side == LegSide.LONG
        tp_price, sl_price = compute_suggested_tp_sl(entry_price, is_long, tp_pct, sl_pct)
        
        # Compute DTE
        dte = None
        if expiration:
            dte = (expiration - entry_datetime).days
        
        leg = TradeLeg(
            leg_id=leg_id,
            trade_id=trade_id,
            variant_id=trade.variant_id,
            side=side,
            instrument=instrument,
            entry_datetime=entry_datetime,
            entry_price=entry_price,
            entry_size=entry_size if is_long else -abs(entry_size),
            tp_price=tp_price,
            sl_price=sl_price,
            strike=strike,
            expiration=expiration,
            dte_at_entry=dte,
            regime_at_entry=regime,
        )
        
        trade.add_leg(leg)
        self._save()
        return leg
    
    def close_leg(
        self,
        leg_id: str,
        exit_price: float,
        exit_datetime: Optional[datetime] = None,
        exit_size: Optional[int] = None,
        reason: ExitReason = ExitReason.MANUAL,
        notes: str = "",
    ) -> Optional[TradeLeg]:
        """Close a specific leg."""
        if exit_datetime is None:
            exit_datetime = datetime.utcnow()
        
        for trade in self.trades.values():
            for leg in trade.legs:
                if leg.leg_id == leg_id:
                    leg.close(exit_price, exit_datetime, exit_size, reason, notes)
                    
                    # Check if all legs closed
                    if not trade.is_open:
                        trade.closed_at = exit_datetime
                    
                    self._save()
                    return leg
        
        return None
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades.values() if t.is_open]
    
    def get_open_legs(self) -> List[TradeLeg]:
        """Get all open legs across all trades."""
        legs = []
        for trade in self.trades.values():
            for leg in trade.legs:
                if leg.is_open:
                    legs.append(leg)
        return legs
    
    def get_trades_by_variant(self, variant_id: str) -> List[Trade]:
        """Get all trades for a specific variant."""
        return [t for t in self.trades.values() if t.variant_id == variant_id]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio-level summary."""
        open_trades = self.get_open_trades()
        closed_trades = [t for t in self.trades.values() if not t.is_open]
        
        total_realized = sum(t.total_realized_pnl for t in self.trades.values())
        
        # By variant
        by_variant = {}
        for trade in self.trades.values():
            vid = trade.variant_id
            if vid not in by_variant:
                by_variant[vid] = {
                    "trade_count": 0,
                    "open_count": 0,
                    "realized_pnl": 0.0,
                }
            by_variant[vid]["trade_count"] += 1
            if trade.is_open:
                by_variant[vid]["open_count"] += 1
            by_variant[vid]["realized_pnl"] += trade.total_realized_pnl
        
        return {
            "total_trades": len(self.trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "total_realized_pnl": total_realized,
            "by_variant": by_variant,
        }
    
    def export_to_csv(self, output_path: Path) -> None:
        """Export trade log to CSV for external analysis."""
        import csv
        
        rows = []
        for trade in self.trades.values():
            for leg in trade.legs:
                rows.append({
                    "trade_id": trade.trade_id,
                    "variant_id": trade.variant_id,
                    "signal_batch_id": trade.signal_batch_id,
                    "structure_type": trade.structure_type,
                    "leg_id": leg.leg_id,
                    "side": leg.side.value,
                    "instrument": leg.instrument,
                    "entry_datetime": leg.entry_datetime.isoformat(),
                    "entry_price": leg.entry_price,
                    "entry_size": leg.entry_size,
                    "exit_datetime": leg.exit_datetime.isoformat() if leg.exit_datetime else "",
                    "exit_price": leg.exit_price or "",
                    "exit_size": leg.exit_size or "",
                    "tp_price": leg.tp_price,
                    "sl_price": leg.sl_price,
                    "status": leg.status.value,
                    "strike": leg.strike,
                    "expiration": leg.expiration.isoformat() if leg.expiration else "",
                    "dte_at_entry": leg.dte_at_entry,
                    "regime_at_entry": leg.regime_at_entry.value if leg.regime_at_entry else "",
                    "realized_pnl": leg.realized_pnl,
                    "notes": leg.notes,
                })
        
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def get_recently_closed_trades(self) -> List[Trade]:
        """
        Get trades that are closed but may not yet have post-mortems.
        
        Useful for finding trades that need post-mortem review.
        """
        return [t for t in self.trades.values() if not t.is_open and t.closed_at is not None]
    
    def get_closed_trades_for_variant(self, variant_id: str) -> List[Trade]:
        """Get all closed trades for a specific variant."""
        return [
            t for t in self.trades.values() 
            if t.variant_id == variant_id and not t.is_open
        ]
    
    def get_trade_for_post_mortem(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade data formatted for post-mortem creation.
        
        Returns a dict suitable for create_post_mortem_from_closed_trade().
        """
        trade = self.trades.get(trade_id)
        if not trade or trade.is_open:
            return None
        
        # Calculate weeks held
        weeks_held = 0
        if trade.opened_at and trade.closed_at:
            delta = trade.closed_at - trade.opened_at
            weeks_held = delta.days // 7
        
        return {
            "trade_id": trade.trade_id,
            "variant_id": trade.variant_id,
            "signal_batch_id": trade.signal_batch_id,
            "structure_type": trade.structure_type,
            "regime_at_open": trade.regime_at_open.value if trade.regime_at_open else None,
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            "max_hold_weeks": trade.max_hold_weeks,
            "weeks_held": weeks_held,
            "notes": trade.notes,
            "legs": [leg.to_dict() for leg in trade.legs],
            "total_realized_pnl": trade.total_realized_pnl,
        }
    
    def close_trade_with_post_mortem(
        self,
        trade_id: str,
        regime_at_exit: Optional[VolatilityRegime] = None,
        robustness_at_entry: float = 0.0,
        signal_context: Optional[Dict[str, Any]] = None,
        pm_manager: Optional['PostMortemManager'] = None,
    ) -> Optional['TradePostMortem']:
        """
        Mark a trade as fully closed and create its post-mortem.
        
        This is the recommended way to close trades for proper tracking.
        Call this after all legs are closed.
        
        Args:
            trade_id: ID of the trade to close
            regime_at_exit: Current regime at exit time
            robustness_at_entry: Robustness score when trade was opened
            signal_context: Optional signal data for execution tracking
            pm_manager: PostMortemManager instance (imports lazily to avoid circular)
        
        Returns:
            TradePostMortem if created, None otherwise
        """
        trade = self.trades.get(trade_id)
        if not trade:
            return None
        
        # Ensure trade is fully closed
        if trade.is_open:
            return None
        
        # Mark closed time if not already set
        if trade.closed_at is None:
            trade.closed_at = datetime.utcnow()
            self._save()
        
        # Create post-mortem if manager provided
        if pm_manager is not None:
            try:
                from post_mortem import create_post_mortem_from_closed_trade
                
                trade_dict = self.get_trade_for_post_mortem(trade_id)
                if trade_dict:
                    return create_post_mortem_from_closed_trade(
                        trade_dict=trade_dict,
                        regime_at_exit=regime_at_exit,
                        robustness_at_entry=robustness_at_entry,
                        pm_manager=pm_manager,
                        signal_context=signal_context,
                    )
            except ImportError:
                print("Warning: post_mortem module not available")
        
        return None
