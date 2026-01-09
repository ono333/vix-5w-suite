"""
Trade Log for VIX 5% Weekly Suite

Exports:
    - TradeLog
    - get_trade_log
    - Trade
    - TradeLeg
    - LegSide (re-exported from enums)
    - LegStatus (re-exported from enums)
    - TradeStatus (re-exported from enums)
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Re-export enums
from enums import (
    VolatilityRegime, 
    VariantRole, 
    TradeStatus, 
    LegStatus, 
    LegSide,
    ExitType,
)


TRADE_LOG_PATH = Path.home() / ".vix_suite" / "trade_log.json"


@dataclass
class TradeLeg:
    """Individual option leg."""
    leg_id: str
    side: LegSide
    leg_type: str  # "call" or "put"
    
    # Contract details
    strike: float
    expiration: dt.date
    contracts: int
    
    # Pricing
    entry_price: float
    current_price: float = 0.0
    exit_price: Optional[float] = None
    
    # Status
    status: LegStatus = LegStatus.OPEN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "side": self.side.value,
            "leg_type": self.leg_type,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "contracts": self.contracts,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeLeg":
        return cls(
            leg_id=d["leg_id"],
            side=LegSide(d["side"]),
            leg_type=d["leg_type"],
            strike=d["strike"],
            expiration=dt.date.fromisoformat(d["expiration"]),
            contracts=d["contracts"],
            entry_price=d["entry_price"],
            current_price=d.get("current_price", 0.0),
            exit_price=d.get("exit_price"),
            status=LegStatus(d.get("status", "open")),
        )


@dataclass
class Trade:
    """A complete trade."""
    trade_id: str
    signal_id: str
    variant_role: VariantRole
    variant_name: str
    
    # Timing
    entry_date: dt.datetime
    exit_date: Optional[dt.datetime] = None
    
    # Market context
    entry_regime: VolatilityRegime = VolatilityRegime.CALM
    entry_vix: float = 0.0
    entry_percentile: float = 0.0
    underlying: str = "^VIX"
    
    # Position
    position_type: str = "diagonal"
    legs: List[TradeLeg] = field(default_factory=list)
    
    # Sizing
    total_contracts: int = 0
    entry_debit: float = 0.0
    max_risk: float = 0.0
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    exit_reason: Optional[ExitType] = None
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk parameters
    target_mult: float = 1.20
    stop_mult: float = 0.50
    target_price: float = 0.0
    stop_price: float = 0.0
    
    # Notes
    entry_notes: str = ""
    exit_notes: str = ""
    lessons_learned: str = ""
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        if self.entry_debit > 0:
            return self.total_pnl / self.entry_debit
        return 0.0
    
    @property
    def days_held(self) -> int:
        end = self.exit_date or dt.datetime.now()
        return (end - self.entry_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "variant_role": self.variant_role.value,
            "variant_name": self.variant_name,
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "entry_regime": self.entry_regime.value,
            "entry_vix": self.entry_vix,
            "entry_percentile": self.entry_percentile,
            "underlying": self.underlying,
            "position_type": self.position_type,
            "legs": [leg.to_dict() for leg in self.legs],
            "total_contracts": self.total_contracts,
            "entry_debit": self.entry_debit,
            "max_risk": self.max_risk,
            "status": self.status.value,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "target_mult": self.target_mult,
            "stop_mult": self.stop_mult,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "entry_notes": self.entry_notes,
            "exit_notes": self.exit_notes,
            "lessons_learned": self.lessons_learned,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trade":
        return cls(
            trade_id=d["trade_id"],
            signal_id=d["signal_id"],
            variant_role=VariantRole(d["variant_role"]),
            variant_name=d["variant_name"],
            entry_date=dt.datetime.fromisoformat(d["entry_date"]),
            exit_date=dt.datetime.fromisoformat(d["exit_date"]) if d.get("exit_date") else None,
            entry_regime=VolatilityRegime(d["entry_regime"]),
            entry_vix=d["entry_vix"],
            entry_percentile=d["entry_percentile"],
            underlying=d.get("underlying", "^VIX"),
            position_type=d["position_type"],
            legs=[TradeLeg.from_dict(leg) for leg in d.get("legs", [])],
            total_contracts=d["total_contracts"],
            entry_debit=d["entry_debit"],
            max_risk=d["max_risk"],
            status=TradeStatus(d["status"]),
            exit_reason=ExitType(d["exit_reason"]) if d.get("exit_reason") else None,
            realized_pnl=d.get("realized_pnl", 0.0),
            unrealized_pnl=d.get("unrealized_pnl", 0.0),
            target_mult=d.get("target_mult", 1.20),
            stop_mult=d.get("stop_mult", 0.50),
            target_price=d.get("target_price", 0.0),
            stop_price=d.get("stop_price", 0.0),
            entry_notes=d.get("entry_notes", ""),
            exit_notes=d.get("exit_notes", ""),
            lessons_learned=d.get("lessons_learned", ""),
        )


class TradeLog:
    """Manages trade storage and retrieval."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or TRADE_LOG_PATH
        self.trades: Dict[str, Trade] = {}
        self._load()
    
    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.trades = {
                    tid: Trade.from_dict(tdata)
                    for tid, tdata in data.get("trades", {}).items()
                }
            except Exception as e:
                print(f"Warning: Could not load trade log: {e}")
                self.trades = {}
    
    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump(
                    {"trades": {tid: t.to_dict() for tid, t in self.trades.items()}},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: Could not save trade log: {e}")
    
    def add_trade(self, trade: Trade) -> str:
        self.trades[trade.trade_id] = trade
        self._save()
        return trade.trade_id
    
    def update_trade(self, trade: Trade) -> None:
        self.trades[trade.trade_id] = trade
        self._save()
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        return self.trades.get(trade_id)
    
    def get_open_trades(self) -> List[Trade]:
        return [t for t in self.trades.values() if t.status == TradeStatus.OPEN]
    
    def get_closed_trades(self) -> List[Trade]:
        return [t for t in self.trades.values() if t.status == TradeStatus.CLOSED]
    
    def get_all_trades(self) -> List[Trade]:
        return list(self.trades.values())
    
    def get_trades_by_variant(self, role: VariantRole) -> List[Trade]:
        return [t for t in self.trades.values() if t.variant_role == role]
    
    def get_trades_by_status(self, status: TradeStatus) -> List[Trade]:
        return [t for t in self.trades.values() if t.status == status]
    
    def close_trade(
        self,
        trade_id: str,
        exit_reason: ExitType,
        exit_notes: str = "",
    ) -> Optional[Trade]:
        trade = self.get_trade(trade_id)
        if not trade:
            return None
        
        trade.exit_date = dt.datetime.now()
        trade.exit_reason = exit_reason
        trade.status = TradeStatus.CLOSED
        trade.exit_notes = exit_notes
        trade.realized_pnl = trade.unrealized_pnl
        trade.unrealized_pnl = 0.0
        
        for leg in trade.legs:
            leg.status = LegStatus.CLOSED
        
        self._save()
        return trade
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        closed = self.get_closed_trades()
        open_trades = self.get_open_trades()
        
        if not closed:
            return {
                "total_trades": len(self.trades),
                "open_trades": len(open_trades),
                "closed_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }
        
        wins = [t for t in closed if t.realized_pnl > 0]
        total_pnl = sum(t.realized_pnl for t in closed)
        
        return {
            "total_trades": len(self.trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed),
            "win_rate": len(wins) / len(closed),
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for display (alias for get_statistics + extras)."""
        stats = self.get_statistics()
        
        # Add more summary details
        all_trades = self.get_all_trades()
        open_trades = self.get_open_trades()
        closed_trades = self.get_closed_trades()
        
        # Calculate by variant
        variant_stats = {}
        for role in VariantRole:
            variant_trades = self.get_trades_by_variant(role)
            closed_variant = [t for t in variant_trades if t.status == TradeStatus.CLOSED]
            if closed_variant:
                wins = [t for t in closed_variant if t.realized_pnl > 0]
                variant_stats[role.value] = {
                    "total": len(variant_trades),
                    "open": len([t for t in variant_trades if t.status == TradeStatus.OPEN]),
                    "closed": len(closed_variant),
                    "win_rate": len(wins) / len(closed_variant) if closed_variant else 0.0,
                    "total_pnl": sum(t.realized_pnl for t in closed_variant),
                }
            else:
                variant_stats[role.value] = {
                    "total": len(variant_trades),
                    "open": len([t for t in variant_trades if t.status == TradeStatus.OPEN]),
                    "closed": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                }
        
        # Calculate unrealized P&L
        total_unrealized = sum(t.unrealized_pnl for t in open_trades)
        total_realized = sum(t.realized_pnl for t in closed_trades)
        
        # Average hold time
        if closed_trades:
            avg_hold_days = sum(t.days_held for t in closed_trades) / len(closed_trades)
        else:
            avg_hold_days = 0.0
        
        return {
            **stats,
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "combined_pnl": total_unrealized + total_realized,
            "avg_hold_days": avg_hold_days,
            "variant_stats": variant_stats,
            "trades_this_week": len([t for t in all_trades 
                                     if t.entry_date > dt.datetime.now() - dt.timedelta(days=7)]),
            "trades_this_month": len([t for t in all_trades 
                                      if t.entry_date > dt.datetime.now() - dt.timedelta(days=30)]),
        }


# Singleton instance
_trade_log_instance: Optional[TradeLog] = None


def get_trade_log(storage_path: Optional[Path] = None) -> TradeLog:
    """Get or create TradeLog instance."""
    global _trade_log_instance
    if _trade_log_instance is None:
        _trade_log_instance = TradeLog(storage_path)
    return _trade_log_instance
