#!/usr/bin/env python3
"""
Trade Log Module for VIX/UVXY Suite

Leg-level trade logging with:
- Independent leg tracking (short and long managed separately)
- Suggested TP/SL calculations
- Execution tracking
- Status management
- Audit trail

Data Model:
- Trade: Groups related legs into a single strategy position
- Leg: Individual option position (can be opened/closed independently)
- Execution: Record of actual fills
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid


class LegSide(Enum):
    """Option leg side."""
    LONG = "long"
    SHORT = "short"


class LegStatus(Enum):
    """Leg status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    EXPIRED = "expired"


class TradeStatus(Enum):
    """Overall trade status."""
    PENDING = "pending"       # Signal generated, not executed
    OPEN = "open"             # At least one leg open
    CLOSED = "closed"         # All legs closed
    EXPIRED = "expired"       # All legs expired
    CANCELLED = "cancelled"   # Never executed


@dataclass
class LegExecution:
    """Record of a leg execution (entry or exit)."""
    execution_id: str
    leg_id: str
    action: str  # "OPEN", "CLOSE", "PARTIAL_CLOSE"
    timestamp: datetime
    quantity: int
    price: float
    slippage: float = 0.0
    commission: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "leg_id": self.leg_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "quantity": self.quantity,
            "price": self.price,
            "slippage": self.slippage,
            "commission": self.commission,
            "notes": self.notes,
        }


@dataclass
class TradeLeg:
    """Single option leg within a trade."""
    leg_id: str
    trade_id: str
    
    # Leg details
    side: LegSide
    instrument: str  # e.g., "UVXY_20260220_C_30"
    underlying: str
    option_type: str  # "C" or "P"
    strike: float
    expiration: str  # ISO date
    
    # Position
    quantity: int  # Positive for long, negative for short
    status: LegStatus = LegStatus.OPEN
    
    # Entry
    entry_datetime: Optional[datetime] = None
    entry_price: float = 0.0
    entry_quantity: int = 0
    
    # Exit
    exit_datetime: Optional[datetime] = None
    exit_price: float = 0.0
    exit_quantity: int = 0
    
    # Suggested levels
    tp_price: float = 0.0  # Take profit
    sl_price: float = 0.0  # Stop loss
    
    # Current state
    current_price: float = 0.0
    current_pnl: float = 0.0
    
    # Metadata
    notes: str = ""
    executions: List[LegExecution] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "trade_id": self.trade_id,
            "side": self.side.value,
            "instrument": self.instrument,
            "underlying": self.underlying,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "quantity": self.quantity,
            "status": self.status.value,
            "entry_datetime": self.entry_datetime.isoformat() if self.entry_datetime else None,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "exit_datetime": self.exit_datetime.isoformat() if self.exit_datetime else None,
            "exit_price": self.exit_price,
            "exit_quantity": self.exit_quantity,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "current_price": self.current_price,
            "current_pnl": self.current_pnl,
            "notes": self.notes,
            "executions": [e.to_dict() for e in self.executions],
        }


@dataclass
class Trade:
    """Complete trade with multiple legs."""
    trade_id: str
    variant_id: str
    signal_batch_id: str
    
    # Trade info
    variant_role: str
    structure: str
    underlying: str
    
    # Status
    status: TradeStatus = TradeStatus.PENDING
    
    # Timing
    signal_datetime: Optional[datetime] = None
    entry_datetime: Optional[datetime] = None
    exit_datetime: Optional[datetime] = None
    
    # Legs
    legs: List[TradeLeg] = field(default_factory=list)
    
    # Suggested management
    suggested_tp_pct: float = 0.0
    suggested_sl_pct: float = 0.0
    
    # PnL tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    
    # Operational metrics
    intervention_count: int = 0
    attention_score: int = 1  # 1-5, how much attention required
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "variant_id": self.variant_id,
            "signal_batch_id": self.signal_batch_id,
            "variant_role": self.variant_role,
            "structure": self.structure,
            "underlying": self.underlying,
            "status": self.status.value,
            "signal_datetime": self.signal_datetime.isoformat() if self.signal_datetime else None,
            "entry_datetime": self.entry_datetime.isoformat() if self.entry_datetime else None,
            "exit_datetime": self.exit_datetime.isoformat() if self.exit_datetime else None,
            "legs": [leg.to_dict() for leg in self.legs],
            "suggested_tp_pct": self.suggested_tp_pct,
            "suggested_sl_pct": self.suggested_sl_pct,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_commission": self.total_commission,
            "intervention_count": self.intervention_count,
            "attention_score": self.attention_score,
            "notes": self.notes,
            "tags": self.tags,
        }


def calculate_tp_sl(
    entry_price: float,
    side: LegSide,
    tp_pct: float = 0.10,
    sl_pct: float = 0.06,
) -> tuple[float, float]:
    """
    Calculate suggested TP and SL levels.
    
    For Long positions:
        TP = Entry * (1 + tp_pct)
        SL = Entry * (1 - sl_pct)
    
    For Short positions:
        TP = Entry * (1 - tp_pct)  (want price to go down)
        SL = Entry * (1 + sl_pct)  (stop if price goes up)
    """
    if side == LegSide.LONG:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    
    return tp_price, sl_price


class TradeLog:
    """
    Trade log manager with persistence.
    
    Stores trades in JSON format for easy inspection and recovery.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.home() / ".vix_suite" / "trade_log.json"
        
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.trades: Dict[str, Trade] = {}
        self._load()
    
    def _load(self):
        """Load trades from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                
                for trade_data in data.get("trades", []):
                    trade = self._deserialize_trade(trade_data)
                    self.trades[trade.trade_id] = trade
            except Exception as e:
                print(f"Warning: Could not load trade log: {e}")
    
    def _save(self):
        """Save trades to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "trades": [t.to_dict() for t in self.trades.values()],
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save trade log: {e}")
    
    def _deserialize_trade(self, data: Dict[str, Any]) -> Trade:
        """Deserialize trade from dict."""
        legs = []
        for leg_data in data.get("legs", []):
            executions = []
            for exec_data in leg_data.get("executions", []):
                exec_data["timestamp"] = datetime.fromisoformat(exec_data["timestamp"])
                executions.append(LegExecution(**exec_data))
            
            leg_data["side"] = LegSide(leg_data["side"])
            leg_data["status"] = LegStatus(leg_data["status"])
            leg_data["executions"] = executions
            
            if leg_data.get("entry_datetime"):
                leg_data["entry_datetime"] = datetime.fromisoformat(leg_data["entry_datetime"])
            if leg_data.get("exit_datetime"):
                leg_data["exit_datetime"] = datetime.fromisoformat(leg_data["exit_datetime"])
            
            legs.append(TradeLeg(**leg_data))
        
        data["legs"] = legs
        data["status"] = TradeStatus(data["status"])
        
        if data.get("signal_datetime"):
            data["signal_datetime"] = datetime.fromisoformat(data["signal_datetime"])
        if data.get("entry_datetime"):
            data["entry_datetime"] = datetime.fromisoformat(data["entry_datetime"])
        if data.get("exit_datetime"):
            data["exit_datetime"] = datetime.fromisoformat(data["exit_datetime"])
        
        return Trade(**data)
    
    def create_trade(
        self,
        variant_id: str,
        signal_batch_id: str,
        variant_role: str,
        structure: str,
        underlying: str,
        suggested_tp_pct: float = 0.10,
        suggested_sl_pct: float = 0.06,
    ) -> Trade:
        """Create a new trade (pending execution)."""
        trade_id = f"T_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        trade = Trade(
            trade_id=trade_id,
            variant_id=variant_id,
            signal_batch_id=signal_batch_id,
            variant_role=variant_role,
            structure=structure,
            underlying=underlying,
            status=TradeStatus.PENDING,
            signal_datetime=datetime.utcnow(),
            suggested_tp_pct=suggested_tp_pct,
            suggested_sl_pct=suggested_sl_pct,
        )
        
        self.trades[trade_id] = trade
        self._save()
        return trade
    
    def add_leg(
        self,
        trade_id: str,
        side: LegSide,
        instrument: str,
        underlying: str,
        option_type: str,
        strike: float,
        expiration: str,
        quantity: int,
        entry_price: float,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
    ) -> TradeLeg:
        """Add a leg to an existing trade."""
        if trade_id not in self.trades:
            raise ValueError(f"Trade {trade_id} not found")
        
        trade = self.trades[trade_id]
        
        # Use trade-level TP/SL if not specified
        if tp_pct is None:
            tp_pct = trade.suggested_tp_pct
        if sl_pct is None:
            sl_pct = trade.suggested_sl_pct
        
        tp_price, sl_price = calculate_tp_sl(entry_price, side, tp_pct, sl_pct)
        
        leg_id = f"L_{trade_id}_{len(trade.legs) + 1}"
        
        leg = TradeLeg(
            leg_id=leg_id,
            trade_id=trade_id,
            side=side,
            instrument=instrument,
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            quantity=quantity if side == LegSide.LONG else -abs(quantity),
            status=LegStatus.OPEN,
            entry_datetime=datetime.utcnow(),
            entry_price=entry_price,
            entry_quantity=abs(quantity),
            tp_price=tp_price,
            sl_price=sl_price,
            current_price=entry_price,
        )
        
        trade.legs.append(leg)
        trade.status = TradeStatus.OPEN
        trade.entry_datetime = datetime.utcnow()
        
        self._save()
        return leg
    
    def close_leg(
        self,
        leg_id: str,
        exit_price: float,
        exit_quantity: Optional[int] = None,
        commission: float = 0.0,
        notes: str = "",
    ) -> TradeLeg:
        """Close a leg (fully or partially)."""
        # Find the leg
        leg = None
        trade = None
        for t in self.trades.values():
            for l in t.legs:
                if l.leg_id == leg_id:
                    leg = l
                    trade = t
                    break
            if leg:
                break
        
        if not leg:
            raise ValueError(f"Leg {leg_id} not found")
        
        if exit_quantity is None:
            exit_quantity = abs(leg.quantity)
        
        # Create execution record
        exec_id = f"E_{leg_id}_{len(leg.executions) + 1}"
        action = "CLOSE" if exit_quantity >= abs(leg.quantity) else "PARTIAL_CLOSE"
        
        execution = LegExecution(
            execution_id=exec_id,
            leg_id=leg_id,
            action=action,
            timestamp=datetime.utcnow(),
            quantity=exit_quantity,
            price=exit_price,
            commission=commission,
            notes=notes,
        )
        leg.executions.append(execution)
        
        # Update leg
        leg.exit_price = exit_price
        leg.exit_quantity += exit_quantity
        leg.exit_datetime = datetime.utcnow()
        
        # Calculate PnL
        if leg.side == LegSide.LONG:
            pnl = (exit_price - leg.entry_price) * exit_quantity * 100
        else:
            pnl = (leg.entry_price - exit_price) * exit_quantity * 100
        
        leg.current_pnl = pnl
        trade.realized_pnl += pnl
        trade.total_commission += commission
        
        # Update status
        if leg.exit_quantity >= abs(leg.entry_quantity):
            leg.status = LegStatus.CLOSED
        else:
            leg.status = LegStatus.PARTIAL
        
        # Check if trade is fully closed
        all_closed = all(
            l.status in [LegStatus.CLOSED, LegStatus.EXPIRED]
            for l in trade.legs
        )
        if all_closed:
            trade.status = TradeStatus.CLOSED
            trade.exit_datetime = datetime.utcnow()
        
        self._save()
        return leg
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all open legs.
        
        prices: Dict mapping instrument to current price
        """
        for trade in self.trades.values():
            if trade.status != TradeStatus.OPEN:
                continue
            
            trade.unrealized_pnl = 0.0
            
            for leg in trade.legs:
                if leg.status != LegStatus.OPEN:
                    continue
                
                if leg.instrument in prices:
                    leg.current_price = prices[leg.instrument]
                    
                    if leg.side == LegSide.LONG:
                        leg.current_pnl = (leg.current_price - leg.entry_price) * abs(leg.quantity) * 100
                    else:
                        leg.current_pnl = (leg.entry_price - leg.current_price) * abs(leg.quantity) * 100
                    
                    trade.unrealized_pnl += leg.current_pnl
        
        self._save()
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.OPEN]
    
    def get_trades_by_variant(self, variant_role: str) -> List[Trade]:
        """Get trades for a specific variant."""
        return [t for t in self.trades.values() if t.variant_role == variant_role]
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a specific trade."""
        return self.trades.get(trade_id)
    
    def get_leg(self, leg_id: str) -> Optional[TradeLeg]:
        """Get a specific leg."""
        for trade in self.trades.values():
            for leg in trade.legs:
                if leg.leg_id == leg_id:
                    return leg
        return None
    
    def add_note(self, trade_id: str, note: str):
        """Add a note to a trade."""
        if trade_id in self.trades:
            self.trades[trade_id].notes += f"\n[{datetime.utcnow().isoformat()}] {note}"
            self._save()
    
    def increment_intervention(self, trade_id: str):
        """Track that an intervention was made."""
        if trade_id in self.trades:
            self.trades[trade_id].intervention_count += 1
            self._save()
    
    def set_attention_score(self, trade_id: str, score: int):
        """Set attention score (1-5)."""
        if trade_id in self.trades:
            self.trades[trade_id].attention_score = max(1, min(5, score))
            self._save()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        open_trades = self.get_open_trades()
        closed_trades = [t for t in self.trades.values() if t.status == TradeStatus.CLOSED]
        
        total_realized = sum(t.realized_pnl for t in closed_trades)
        total_unrealized = sum(t.unrealized_pnl for t in open_trades)
        total_commission = sum(t.total_commission for t in self.trades.values())
        
        return {
            "total_trades": len(self.trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_commission": total_commission,
            "net_pnl": total_realized + total_unrealized - total_commission,
        }


# Global instance
_trade_log: Optional[TradeLog] = None


def get_trade_log(storage_path: Optional[Path] = None) -> TradeLog:
    """Get or create the global trade log instance."""
    global _trade_log
    if _trade_log is None:
        _trade_log = TradeLog(storage_path)
    return _trade_log
