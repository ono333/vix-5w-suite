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
from typing import Dict, List, Optional, Any
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
    entry_vix_level: float
    entry_percentile: float
    
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
    trade_id: str
    variant_id: str
    variant_name: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_pct: float
    duration_days: int
    exit_reason: str
    entry_regime: str
    exit_regime: str
    contracts: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        return cls(**data)


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
            
            # Load history
            history_data = data.get("history", [])
            for record_data in history_data:
                self.history.append(TradeRecord.from_dict(record_data))
                
        except Exception as e:
            print(f"Warning: Could not load trade log: {e}")
    
    def _save(self) -> None:
        """Persist positions and history to disk."""
        try:
            data = {
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "history": [r.to_dict() for r in self.history],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save trade log: {e}")
    
    # ================================================================
    # Position Query Methods (for email integration)
    # ================================================================
    
    def has_open_position(self, variant_id: str) -> bool:
        """Check if variant has an open position."""
        pos = self.positions.get(variant_id)
        return pos is not None and pos.is_open()
    
    def get_open_position(self, variant_id: str) -> Optional[Position]:
        """Get open position for variant, or None."""
        pos = self.positions.get(variant_id)
        if pos and pos.is_open():
            return pos
        return None
    
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
