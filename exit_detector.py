"""
Exit Detector for VIX 5% Weekly Suite
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from enums import ExitType, ExitUrgency, ExitStatus, VolatilityRegime

EXIT_STORE_PATH = Path.home() / ".vix_suite" / "exit_events.json"

@dataclass
class ExitEvent:
    event_id: str
    trade_id: str
    exit_type: ExitType
    urgency: ExitUrgency
    status: ExitStatus = ExitStatus.PENDING
    triggered_at: datetime = field(default_factory=datetime.now)
    reason: str = ""
    current_pnl: float = 0.0
    target_pnl: float = 0.0
    stop_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "exit_type": self.exit_type.value,
            "urgency": self.urgency.value,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat(),
            "reason": self.reason,
            "current_pnl": self.current_pnl,
            "target_pnl": self.target_pnl,
            "stop_pnl": self.stop_pnl,
        }

class ExitStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or EXIT_STORE_PATH
        self.events: Dict[str, ExitEvent] = {}
        self._load()
    
    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for eid, edata in data.get("events", {}).items():
                    self.events[eid] = ExitEvent(
                        event_id=edata["event_id"],
                        trade_id=edata["trade_id"],
                        exit_type=ExitType(edata["exit_type"]),
                        urgency=ExitUrgency(edata["urgency"]),
                        status=ExitStatus(edata["status"]),
                        triggered_at=datetime.fromisoformat(edata["triggered_at"]),
                        reason=edata.get("reason", ""),
                    )
            except:
                self.events = {}
    
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"events": {k: v.to_dict() for k, v in self.events.items()}}, f, indent=2)
    
    def add_event(self, event: ExitEvent):
        self.events[event.event_id] = event
        self._save()
    
    def get_pending(self) -> List[ExitEvent]:
        return [e for e in self.events.values() if e.status == ExitStatus.PENDING]

_exit_store: Optional[ExitStore] = None

def get_exit_store() -> ExitStore:
    global _exit_store
    if _exit_store is None:
        _exit_store = ExitStore()
    return _exit_store

def detect_all_exits(trades: List, regime_state, current_prices: Dict[str, float] = None) -> List[ExitEvent]:
    events = []
    import uuid
    for trade in trades:
        if hasattr(trade, 'status') and trade.status.value != "open":
            continue
        
        pnl_pct = trade.return_pct if hasattr(trade, 'return_pct') else 0.0
        target = getattr(trade, 'target_mult', 1.2) - 1
        stop = getattr(trade, 'stop_mult', 0.5) - 1
        
        if pnl_pct >= target:
            events.append(ExitEvent(
                event_id=f"EXIT-{uuid.uuid4().hex[:8]}",
                trade_id=trade.trade_id,
                exit_type=ExitType.TARGET,
                urgency=ExitUrgency.END_OF_DAY,
                reason=f"Target hit: {pnl_pct:.1%} >= {target:.1%}",
            ))
        elif pnl_pct <= stop:
            events.append(ExitEvent(
                event_id=f"EXIT-{uuid.uuid4().hex[:8]}",
                trade_id=trade.trade_id,
                exit_type=ExitType.STOP,
                urgency=ExitUrgency.IMMEDIATE,
                reason=f"Stop hit: {pnl_pct:.1%} <= {stop:.1%}",
            ))
    return events

def get_exit_urgency_color(urgency: ExitUrgency) -> str:
    colors = {
        ExitUrgency.IMMEDIATE: "#f44336",
        ExitUrgency.END_OF_DAY: "#FF9800",
        ExitUrgency.END_OF_WEEK: "#2196F3",
        ExitUrgency.MONITOR: "#4CAF50",
    }
    return colors.get(urgency, "#757575")

def get_exit_type_icon(exit_type: ExitType) -> str:
    icons = {
        ExitType.TARGET: "🎯",
        ExitType.STOP: "🛑",
        ExitType.EXPIRY: "⏰",
        ExitType.REGIME_CHANGE: "🔄",
        ExitType.MANUAL: "✋",
        ExitType.TIME_STOP: "⏱️",
    }
    return icons.get(exit_type, "❓")
