"""
Trade Log for VIX 5% Weekly Suite

Manages trade logging, storage, and retrieval for paper trading
and post-mortem analysis.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from enums import VolatilityRegime, VariantRole, TradeStatus, ExitReason


# =============================================================================
# TRADE DATA STRUCTURES
# =============================================================================

@dataclass
class TradeLeg:
    """Individual option leg in a trade."""
    leg_id: str
    leg_type: str  # "long_call", "short_call", "long_put", "short_put"
    
    # Contract details
    strike: float
    expiration: dt.date
    contracts: int
    
    # Pricing
    entry_price: float
    current_price: float = 0.0
    exit_price: Optional[float] = None
    
    # Status
    status: str = "open"  # "open", "closed", "expired", "rolled"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "leg_type": self.leg_type,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "contracts": self.contracts,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeLeg":
        return cls(
            leg_id=data["leg_id"],
            leg_type=data["leg_type"],
            strike=data["strike"],
            expiration=dt.date.fromisoformat(data["expiration"]),
            contracts=data["contracts"],
            entry_price=data["entry_price"],
            current_price=data.get("current_price", 0.0),
            exit_price=data.get("exit_price"),
            status=data.get("status", "open"),
        )


@dataclass
class Trade:
    """A complete trade with all legs."""
    trade_id: str
    signal_id: str
    variant_role: VariantRole
    variant_name: str
    
    # Timing
    entry_date: dt.datetime
    exit_date: Optional[dt.datetime] = None
    
    # Market context at entry
    entry_regime: VolatilityRegime = VolatilityRegime.CALM
    entry_vix: float = 0.0
    entry_percentile: float = 0.0
    underlying_symbol: str = "^VIX"
    
    # Position
    position_type: str = "diagonal"
    legs: List[TradeLeg] = field(default_factory=list)
    
    # Sizing
    total_contracts: int = 0
    entry_debit: float = 0.0
    max_risk: float = 0.0
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    exit_reason: Optional[ExitReason] = None
    
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
            "underlying_symbol": self.underlying_symbol,
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
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        return cls(
            trade_id=data["trade_id"],
            signal_id=data["signal_id"],
            variant_role=VariantRole(data["variant_role"]),
            variant_name=data["variant_name"],
            entry_date=dt.datetime.fromisoformat(data["entry_date"]),
            exit_date=dt.datetime.fromisoformat(data["exit_date"]) if data.get("exit_date") else None,
            entry_regime=VolatilityRegime(data["entry_regime"]),
            entry_vix=data["entry_vix"],
            entry_percentile=data["entry_percentile"],
            underlying_symbol=data.get("underlying_symbol", "^VIX"),
            position_type=data["position_type"],
            legs=[TradeLeg.from_dict(leg) for leg in data.get("legs", [])],
            total_contracts=data["total_contracts"],
            entry_debit=data["entry_debit"],
            max_risk=data["max_risk"],
            status=TradeStatus(data["status"]),
            exit_reason=ExitReason(data["exit_reason"]) if data.get("exit_reason") else None,
            realized_pnl=data.get("realized_pnl", 0.0),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            target_mult=data.get("target_mult", 1.20),
            stop_mult=data.get("stop_mult", 0.50),
            target_price=data.get("target_price", 0.0),
            stop_price=data.get("stop_price", 0.0),
            entry_notes=data.get("entry_notes", ""),
            exit_notes=data.get("exit_notes", ""),
            lessons_learned=data.get("lessons_learned", ""),
        )
    
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


# =============================================================================
# TRADE LOG MANAGER
# =============================================================================

class TradeLog:
    """Manages trade storage and retrieval."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".vix_suite" / "trade_log.json"
        self.trades: Dict[str, Trade] = {}
        self._load()
    
    def _load(self) -> None:
        """Load trades from disk."""
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
        """Save trades to disk."""
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
        """Add a new trade."""
        self.trades[trade.trade_id] = trade
        self._save()
        return trade.trade_id
    
    def update_trade(self, trade: Trade) -> None:
        """Update an existing trade."""
        self.trades[trade.trade_id] = trade
        self._save()
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID."""
        return self.trades.get(trade_id)
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.OPEN]
    
    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.CLOSED]
    
    def get_trades_by_variant(self, role: VariantRole) -> List[Trade]:
        """Get all trades for a specific variant."""
        return [t for t in self.trades.values() if t.variant_role == role]
    
    def get_trades_by_regime(self, regime: VolatilityRegime) -> List[Trade]:
        """Get all trades entered in a specific regime."""
        return [t for t in self.trades.values() if t.entry_regime == regime]
    
    def get_trades_in_period(
        self,
        start: dt.datetime,
        end: Optional[dt.datetime] = None,
    ) -> List[Trade]:
        """Get trades entered within a date range."""
        end = end or dt.datetime.now()
        return [
            t for t in self.trades.values()
            if start <= t.entry_date <= end
        ]
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        exit_notes: str = "",
    ) -> Optional[Trade]:
        """Close a trade and calculate final P&L."""
        trade = self.get_trade(trade_id)
        if trade is None:
            return None
        
        trade.exit_date = dt.datetime.now()
        trade.exit_reason = exit_reason
        trade.status = TradeStatus.CLOSED
        trade.exit_notes = exit_notes
        
        # Calculate realized P&L (simplified)
        # In reality, this would use actual exit prices per leg
        trade.realized_pnl = trade.unrealized_pnl
        trade.unrealized_pnl = 0.0
        
        # Update legs
        for leg in trade.legs:
            leg.status = "closed"
            leg.exit_price = exit_price  # Simplified
        
        self._save()
        return trade
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate aggregate statistics."""
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
                "avg_hold_days": 0.0,
            }
        
        wins = [t for t in closed if t.realized_pnl > 0]
        total_pnl = sum(t.realized_pnl for t in closed)
        
        return {
            "total_trades": len(self.trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed),
            "avg_hold_days": sum(t.days_held for t in closed) / len(closed),
            "best_trade": max(t.realized_pnl for t in closed),
            "worst_trade": min(t.realized_pnl for t in closed),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_trade_from_signal(
    signal,  # VariantSignal type from variant_generator
    actual_fill_price: Optional[float] = None,
    contracts_filled: Optional[int] = None,
    entry_notes: str = "",
) -> Trade:
    """Create a Trade from an executed signal."""
    
    # Use signal data or overrides
    fill_price = actual_fill_price if actual_fill_price else signal.estimated_debit
    contracts = contracts_filled if contracts_filled else signal.suggested_contracts
    
    # Create legs
    legs = []
    
    # Long leg
    long_leg = TradeLeg(
        leg_id=f"{signal.signal_id}_long",
        leg_type="long_call",
        strike=signal.long_strike,
        expiration=signal.long_expiration,
        contracts=contracts,
        entry_price=signal.long_estimated_price,
        current_price=signal.long_estimated_price,
        status="open",
    )
    legs.append(long_leg)
    
    # Short leg (if diagonal)
    if signal.short_strike is not None and signal.position_type == "diagonal":
        short_leg = TradeLeg(
            leg_id=f"{signal.signal_id}_short",
            leg_type="short_call",
            strike=signal.short_strike,
            expiration=signal.short_expiration,
            contracts=-contracts,  # Negative for short
            entry_price=signal.short_estimated_price or 0.0,
            current_price=signal.short_estimated_price or 0.0,
            status="open",
        )
        legs.append(short_leg)
    
    # Calculate targets and stops
    target_price = fill_price * signal.target_mult
    stop_price = fill_price * signal.stop_mult
    
    trade = Trade(
        trade_id=str(uuid.uuid4())[:8],
        signal_id=signal.signal_id,
        variant_role=signal.variant_role,
        variant_name=signal.variant_name,
        entry_date=dt.datetime.now(),
        entry_regime=signal.regime,
        entry_vix=signal.vix_level,
        entry_percentile=signal.vix_percentile,
        underlying_symbol=signal.underlying_symbol,
        position_type=signal.position_type,
        legs=legs,
        total_contracts=contracts,
        entry_debit=fill_price,
        max_risk=fill_price,  # For debit spreads
        status=TradeStatus.OPEN,
        target_mult=signal.target_mult,
        stop_mult=signal.stop_mult,
        target_price=target_price,
        stop_price=stop_price,
        entry_notes=entry_notes,
    )
    
    return trade


def format_trade_summary(trade: Trade) -> str:
    """Format trade for display."""
    status_emoji = {
        TradeStatus.OPEN: "🟢",
        TradeStatus.CLOSED: "⚫",
        TradeStatus.SIGNAL: "🔵",
        TradeStatus.PENDING: "🟡",
        TradeStatus.CLOSING: "🟠",
        TradeStatus.EXPIRED: "⚪",
        TradeStatus.CANCELLED: "🔴",
    }
    
    emoji = status_emoji.get(trade.status, "⚪")
    
    lines = [
        f"{emoji} **{trade.variant_name}** ({trade.trade_id})",
        f"   Status: {trade.status.value.upper()} | {trade.position_type}",
        f"   Entry: {trade.entry_date.strftime('%Y-%m-%d')} @ ${trade.entry_debit:,.2f}",
        f"   VIX: {trade.entry_vix:.2f} ({trade.entry_percentile:.0%}) - {trade.entry_regime.value}",
    ]
    
    # Legs summary
    for leg in trade.legs:
        leg_dir = "+" if leg.contracts > 0 else "-"
        lines.append(
            f"   {leg_dir}{abs(leg.contracts)} {leg.strike:.1f} {leg.leg_type} "
            f"exp {leg.expiration}"
        )
    
    # P&L
    if trade.status == TradeStatus.OPEN:
        lines.append(f"   Unrealized P&L: ${trade.unrealized_pnl:,.2f} ({trade.return_pct:.1%})")
    else:
        lines.append(f"   Realized P&L: ${trade.realized_pnl:,.2f} ({trade.return_pct:.1%})")
        if trade.exit_reason:
            lines.append(f"   Exit: {trade.exit_reason.value}")
    
    lines.append(f"   Days held: {trade.days_held}")
    
    return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS FOR APP IMPORTS
# =============================================================================

def get_trade_log(storage_path: Optional[Path] = None) -> TradeLog:
    """Get or create a TradeLog instance."""
    return TradeLog(storage_path)


def load_trade_log(storage_path: Optional[Path] = None) -> TradeLog:
    """Alias for get_trade_log."""
    return TradeLog(storage_path)


def get_open_trades(storage_path: Optional[Path] = None) -> List[Trade]:
    """Get all open trades."""
    log = TradeLog(storage_path)
    return log.get_open_trades()


def get_closed_trades(storage_path: Optional[Path] = None) -> List[Trade]:
    """Get all closed trades."""
    log = TradeLog(storage_path)
    return log.get_closed_trades()


def get_all_trades(storage_path: Optional[Path] = None) -> List[Trade]:
    """Get all trades."""
    log = TradeLog(storage_path)
    return list(log.trades.values())


def save_trade(trade: Trade, storage_path: Optional[Path] = None) -> str:
    """Save a trade to the log."""
    log = TradeLog(storage_path)
    return log.add_trade(trade)


def update_trade(trade: Trade, storage_path: Optional[Path] = None) -> None:
    """Update an existing trade."""
    log = TradeLog(storage_path)
    log.update_trade(trade)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing trade log...")
    
    # Create a sample trade
    trade = Trade(
        trade_id="TEST001",
        signal_id="V1_INCOME_20250108_150000",
        variant_role=VariantRole.INCOME,
        variant_name="V1 Income Harvester",
        entry_date=dt.datetime.now() - dt.timedelta(days=7),
        entry_regime=VolatilityRegime.CALM,
        entry_vix=14.5,
        entry_percentile=0.15,
        position_type="diagonal",
        legs=[
            TradeLeg(
                leg_id="TEST001_long",
                leg_type="long_call",
                strike=20.0,
                expiration=dt.date.today() + dt.timedelta(days=180),
                contracts=5,
                entry_price=3.50,
                current_price=3.75,
            ),
            TradeLeg(
                leg_id="TEST001_short",
                leg_type="short_call",
                strike=20.0,
                expiration=dt.date.today() + dt.timedelta(days=7),
                contracts=-5,
                entry_price=0.50,
                current_price=0.30,
            ),
        ],
        total_contracts=5,
        entry_debit=1500.0,
        max_risk=1500.0,
        status=TradeStatus.OPEN,
        unrealized_pnl=225.0,  # (3.75-3.50)*5*100 + (0.50-0.30)*5*100
    )
    
    print(format_trade_summary(trade))
