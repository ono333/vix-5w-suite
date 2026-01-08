"""
Exit Detector for VIX 5% Weekly Suite

Exports:
    - detect_all_exits
    - ExitEvent
    - ExitType (re-exported from enums)
    - ExitUrgency (re-exported from enums)
    - ExitStatus (re-exported from enums)
    - get_exit_store
    - get_exit_urgency_color
    - get_exit_type_icon
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
    ExitType,
    ExitUrgency,
    ExitStatus,
)


EXIT_STORE_PATH = Path.home() / ".vix_suite" / "exit_events.json"


def get_exit_urgency_color(urgency: ExitUrgency) -> str:
    """Get color for exit urgency."""
    colors = {
        ExitUrgency.IMMEDIATE: "#E74C3C",  # Red
        ExitUrgency.SOON: "#F1C40F",       # Yellow
        ExitUrgency.OPTIONAL: "#3498DB",   # Blue
    }
    return colors.get(urgency, "#95A5A6")


def get_exit_type_icon(exit_type: ExitType) -> str:
    """Get icon for exit type."""
    icons = {
        ExitType.TARGET_HIT: "🎯",
        ExitType.STOP_HIT: "🛑",
        ExitType.REGIME_EXIT: "🔄",
        ExitType.TIME_STOP: "⏰",
        ExitType.EXPIRATION: "📅",
        ExitType.MANUAL: "✋",
        ExitType.ROLL: "🔁",
    }
    return icons.get(exit_type, "📤")


@dataclass
class ExitEvent:
    """An exit signal for a trade."""
    event_id: str
    trade_id: str
    signal_id: str
    variant_role: VariantRole
    
    # Exit details
    exit_type: ExitType
    urgency: ExitUrgency
    status: ExitStatus = ExitStatus.PENDING
    
    # Timing
    detected_at: dt.datetime = field(default_factory=dt.datetime.now)
    acknowledged_at: Optional[dt.datetime] = None
    executed_at: Optional[dt.datetime] = None
    
    # Context
    current_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    pnl_at_detection: float = 0.0
    
    # Regime context
    current_regime: VolatilityRegime = VolatilityRegime.CALM
    entry_regime: VolatilityRegime = VolatilityRegime.CALM
    
    # Recommendation
    recommendation: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "variant_role": self.variant_role.value,
            "exit_type": self.exit_type.value,
            "urgency": self.urgency.value,
            "status": self.status.value,
            "detected_at": self.detected_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "pnl_at_detection": self.pnl_at_detection,
            "current_regime": self.current_regime.value,
            "entry_regime": self.entry_regime.value,
            "recommendation": self.recommendation,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExitEvent":
        return cls(
            event_id=d["event_id"],
            trade_id=d["trade_id"],
            signal_id=d["signal_id"],
            variant_role=VariantRole(d["variant_role"]),
            exit_type=ExitType(d["exit_type"]),
            urgency=ExitUrgency(d["urgency"]),
            status=ExitStatus(d.get("status", "pending")),
            detected_at=dt.datetime.fromisoformat(d["detected_at"]),
            acknowledged_at=dt.datetime.fromisoformat(d["acknowledged_at"]) if d.get("acknowledged_at") else None,
            executed_at=dt.datetime.fromisoformat(d["executed_at"]) if d.get("executed_at") else None,
            current_price=d.get("current_price", 0.0),
            target_price=d.get("target_price", 0.0),
            stop_price=d.get("stop_price", 0.0),
            pnl_at_detection=d.get("pnl_at_detection", 0.0),
            current_regime=VolatilityRegime(d.get("current_regime", "CALM")),
            entry_regime=VolatilityRegime(d.get("entry_regime", "CALM")),
            recommendation=d.get("recommendation", ""),
            notes=d.get("notes", ""),
        )


class ExitStore:
    """Storage for exit events."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or EXIT_STORE_PATH
        self.events: Dict[str, ExitEvent] = {}
        self._load()
    
    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.events = {
                    eid: ExitEvent.from_dict(edata)
                    for eid, edata in data.get("events", {}).items()
                }
            except Exception as e:
                print(f"Warning: Could not load exit store: {e}")
                self.events = {}
    
    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump(
                    {"events": {eid: e.to_dict() for eid, e in self.events.items()}},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: Could not save exit store: {e}")
    
    def add_event(self, event: ExitEvent) -> str:
        self.events[event.event_id] = event
        self._save()
        return event.event_id
    
    def get_event(self, event_id: str) -> Optional[ExitEvent]:
        return self.events.get(event_id)
    
    def get_pending_events(self) -> List[ExitEvent]:
        return [e for e in self.events.values() if e.status == ExitStatus.PENDING]
    
    def get_events_for_trade(self, trade_id: str) -> List[ExitEvent]:
        return [e for e in self.events.values() if e.trade_id == trade_id]
    
    def acknowledge_event(self, event_id: str) -> Optional[ExitEvent]:
        event = self.get_event(event_id)
        if event:
            event.status = ExitStatus.ACKNOWLEDGED
            event.acknowledged_at = dt.datetime.now()
            self._save()
        return event
    
    def execute_event(self, event_id: str, notes: str = "") -> Optional[ExitEvent]:
        event = self.get_event(event_id)
        if event:
            event.status = ExitStatus.EXECUTED
            event.executed_at = dt.datetime.now()
            event.notes = notes
            self._save()
        return event
    
    def dismiss_event(self, event_id: str, notes: str = "") -> Optional[ExitEvent]:
        event = self.get_event(event_id)
        if event:
            event.status = ExitStatus.DISMISSED
            event.notes = notes
            self._save()
        return event


# Singleton instance
_exit_store_instance: Optional[ExitStore] = None


def get_exit_store(storage_path: Optional[Path] = None) -> ExitStore:
    """Get or create ExitStore instance."""
    global _exit_store_instance
    if _exit_store_instance is None:
        _exit_store_instance = ExitStore(storage_path)
    return _exit_store_instance


def _check_target_hit(
    trade: Any,
    current_value: float,
) -> Optional[ExitEvent]:
    """Check if profit target was hit."""
    if trade.target_price > 0 and current_value >= trade.target_price:
        return ExitEvent(
            event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            variant_role=trade.variant_role,
            exit_type=ExitType.TARGET_HIT,
            urgency=ExitUrgency.SOON,
            current_price=current_value,
            target_price=trade.target_price,
            stop_price=trade.stop_price,
            pnl_at_detection=current_value - trade.entry_debit,
            recommendation="Profit target reached - consider taking profits",
        )
    return None


def _check_stop_hit(
    trade: Any,
    current_value: float,
) -> Optional[ExitEvent]:
    """Check if stop loss was hit."""
    if trade.stop_price > 0 and current_value <= trade.stop_price:
        return ExitEvent(
            event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            variant_role=trade.variant_role,
            exit_type=ExitType.STOP_HIT,
            urgency=ExitUrgency.IMMEDIATE,
            current_price=current_value,
            target_price=trade.target_price,
            stop_price=trade.stop_price,
            pnl_at_detection=current_value - trade.entry_debit,
            recommendation="Stop loss triggered - exit to limit losses",
        )
    return None


def _check_regime_exit(
    trade: Any,
    current_regime: VolatilityRegime,
) -> Optional[ExitEvent]:
    """Check if regime has changed unfavorably."""
    # V1 Income should exit if regime becomes STRESSED or EXTREME
    if trade.variant_role == VariantRole.INCOME:
        if current_regime in (VolatilityRegime.STRESSED, VolatilityRegime.EXTREME):
            return ExitEvent(
                event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
                trade_id=trade.trade_id,
                signal_id=trade.signal_id,
                variant_role=trade.variant_role,
                exit_type=ExitType.REGIME_EXIT,
                urgency=ExitUrgency.SOON,
                current_regime=current_regime,
                entry_regime=trade.entry_regime,
                recommendation=f"Regime changed to {current_regime.value} - V1 Income should exit",
            )
    
    # V3 Hedge should exit when regime calms
    if trade.variant_role == VariantRole.HEDGE:
        if current_regime == VolatilityRegime.CALM:
            return ExitEvent(
                event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
                trade_id=trade.trade_id,
                signal_id=trade.signal_id,
                variant_role=trade.variant_role,
                exit_type=ExitType.REGIME_EXIT,
                urgency=ExitUrgency.OPTIONAL,
                current_regime=current_regime,
                entry_regime=trade.entry_regime,
                recommendation="Regime calmed - hedge no longer needed",
            )
    
    return None


def _check_time_stop(
    trade: Any,
    max_hold_days: int = 84,  # ~12 weeks
) -> Optional[ExitEvent]:
    """Check if maximum hold time exceeded."""
    if trade.days_held >= max_hold_days:
        return ExitEvent(
            event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
            trade_id=trade.trade_id,
            signal_id=trade.signal_id,
            variant_role=trade.variant_role,
            exit_type=ExitType.TIME_STOP,
            urgency=ExitUrgency.SOON,
            recommendation=f"Maximum hold time ({max_hold_days} days) exceeded",
        )
    return None


def _check_expiration(
    trade: Any,
    days_warning: int = 7,
) -> Optional[ExitEvent]:
    """Check if near expiration."""
    for leg in trade.legs:
        days_to_exp = (leg.expiration - dt.date.today()).days
        if days_to_exp <= days_warning:
            return ExitEvent(
                event_id=f"exit_{trade.trade_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
                trade_id=trade.trade_id,
                signal_id=trade.signal_id,
                variant_role=trade.variant_role,
                exit_type=ExitType.EXPIRATION,
                urgency=ExitUrgency.IMMEDIATE if days_to_exp <= 2 else ExitUrgency.SOON,
                recommendation=f"Option expiring in {days_to_exp} days - close or roll",
            )
    return None


def detect_all_exits(
    trades: List[Any],
    current_regime: VolatilityRegime,
    current_prices: Optional[Dict[str, float]] = None,
) -> List[ExitEvent]:
    """
    Detect exit signals for a list of trades.
    
    Args:
        trades: List of Trade objects
        current_regime: Current volatility regime
        current_prices: Dict of trade_id -> current position value
    
    Returns:
        List of ExitEvent objects
    """
    events = []
    current_prices = current_prices or {}
    
    for trade in trades:
        if trade.status != TradeStatus.OPEN:
            continue
        
        current_value = current_prices.get(trade.trade_id, trade.entry_debit)
        
        # Check each exit condition
        event = _check_target_hit(trade, current_value)
        if event:
            events.append(event)
            continue  # Only one exit event per trade
        
        event = _check_stop_hit(trade, current_value)
        if event:
            events.append(event)
            continue
        
        event = _check_regime_exit(trade, current_regime)
        if event:
            events.append(event)
            continue
        
        event = _check_time_stop(trade)
        if event:
            events.append(event)
            continue
        
        event = _check_expiration(trade)
        if event:
            events.append(event)
    
    # Sort by urgency (IMMEDIATE first)
    urgency_order = {
        ExitUrgency.IMMEDIATE: 0,
        ExitUrgency.SOON: 1,
        ExitUrgency.OPTIONAL: 2,
    }
    events.sort(key=lambda e: urgency_order.get(e.urgency, 99))
    
    return events
