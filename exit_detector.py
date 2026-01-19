"""
Exit Detector for VIX 5% Weekly Suite

Detects exit conditions for open positions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


class ExitType(Enum):
    """Type of exit signal."""
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TIME_DECAY = "time_decay"
    REGIME_CHANGE = "regime_change"
    EXPIRATION = "expiration"
    MANUAL = "manual"


class ExitUrgency(Enum):
    """Urgency level for exit."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExitStatus(Enum):
    """Status of exit event."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    EXECUTED = "executed"
    DISMISSED = "dismissed"


@dataclass
class ExitEvent:
    """An exit event/signal."""
    event_id: str
    trade_id: str
    variant_id: str
    exit_type: ExitType
    urgency: ExitUrgency
    status: ExitStatus = ExitStatus.PENDING
    trigger_price: float = 0.0
    current_price: float = 0.0
    pnl_if_exit: float = 0.0
    reason: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "variant_id": self.variant_id,
            "exit_type": self.exit_type.value,
            "urgency": self.urgency.value,
            "status": self.status.value,
            "trigger_price": self.trigger_price,
            "current_price": self.current_price,
            "pnl_if_exit": self.pnl_if_exit,
            "reason": self.reason,
            "detected_at": self.detected_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


class ExitStore:
    """Store for exit events."""
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = str(Path.home() / ".vix_suite" / "exit_events.json")
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: Dict[str, ExitEvent] = {}
        self._load()
    
    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                # Reconstruct events
                for event_id, event_data in data.get("events", {}).items():
                    event_data["exit_type"] = ExitType(event_data["exit_type"])
                    event_data["urgency"] = ExitUrgency(event_data["urgency"])
                    event_data["status"] = ExitStatus(event_data["status"])
                    event_data["detected_at"] = datetime.fromisoformat(event_data["detected_at"])
                    if event_data.get("executed_at"):
                        event_data["executed_at"] = datetime.fromisoformat(event_data["executed_at"])
                    self.events[event_id] = ExitEvent(**event_data)
            except Exception as e:
                print(f"Warning: Could not load exit events: {e}")
    
    def _save(self):
        try:
            data = {
                "events": {k: v.to_dict() for k, v in self.events.items()},
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save exit events: {e}")
    
    def add_event(self, event: ExitEvent):
        self.events[event.event_id] = event
        self._save()
    
    def get_pending_events(self) -> List[ExitEvent]:
        return [e for e in self.events.values() if e.status == ExitStatus.PENDING]
    
    def acknowledge_event(self, event_id: str):
        if event_id in self.events:
            self.events[event_id].status = ExitStatus.ACKNOWLEDGED
            self._save()
    
    def execute_event(self, event_id: str):
        if event_id in self.events:
            self.events[event_id].status = ExitStatus.EXECUTED
            self.events[event_id].executed_at = datetime.now()
            self._save()


# Singleton
_exit_store: Optional[ExitStore] = None

def get_exit_store() -> ExitStore:
    global _exit_store
    if _exit_store is None:
        _exit_store = ExitStore()
    return _exit_store


def detect_all_exits(trade_log, current_prices: Dict[str, float] = None) -> List[ExitEvent]:
    """
    Detect exit conditions for all open positions.
    
    Returns list of ExitEvent objects for positions that should be exited.
    """
    events = []
    
    # Get open positions
    try:
        open_positions = trade_log.get_open_diagonals()
    except:
        open_positions = []
    
    for pos in open_positions:
        # Check for target/stop hits
        if hasattr(pos, 'total_pnl') and hasattr(pos, 'long_entry_price'):
            if pos.long_entry_price > 0:
                pnl_pct = pos.total_pnl / (pos.long_entry_price * pos.contracts * 100)
                
                # Check target
                target_pct = getattr(pos, 'target_pct', 0.40)
                if pnl_pct >= target_pct:
                    events.append(ExitEvent(
                        event_id=f"EXIT-{pos.position_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        trade_id=pos.position_id,
                        variant_id=pos.variant_id,
                        exit_type=ExitType.TARGET_HIT,
                        urgency=ExitUrgency.HIGH,
                        pnl_if_exit=pos.total_pnl,
                        reason=f"Target reached: {pnl_pct:.0%} >= {target_pct:.0%}",
                    ))
                
                # Check stop
                stop_pct = getattr(pos, 'stop_pct', -0.60)
                if pnl_pct <= -abs(stop_pct):
                    events.append(ExitEvent(
                        event_id=f"EXIT-{pos.position_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        trade_id=pos.position_id,
                        variant_id=pos.variant_id,
                        exit_type=ExitType.STOP_HIT,
                        urgency=ExitUrgency.CRITICAL,
                        pnl_if_exit=pos.total_pnl,
                        reason=f"Stop hit: {pnl_pct:.0%} <= -{abs(stop_pct):.0%}",
                    ))
        
        # Check for expiration
        long_dte = pos.days_to_long_expiry() if hasattr(pos, 'days_to_long_expiry') else 999
        if long_dte <= 7:
            events.append(ExitEvent(
                event_id=f"EXIT-{pos.position_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                trade_id=pos.position_id,
                variant_id=pos.variant_id,
                exit_type=ExitType.EXPIRATION,
                urgency=ExitUrgency.CRITICAL if long_dte <= 3 else ExitUrgency.HIGH,
                pnl_if_exit=pos.total_pnl if hasattr(pos, 'total_pnl') else 0,
                reason=f"Long leg expiring in {long_dte} days",
            ))
    
    return events


def get_exit_urgency_color(urgency: ExitUrgency) -> str:
    """Get color for urgency level."""
    colors = {
        ExitUrgency.CRITICAL: "#F44336",
        ExitUrgency.HIGH: "#FF9800",
        ExitUrgency.MEDIUM: "#FFC107",
        ExitUrgency.LOW: "#4CAF50",
    }
    return colors.get(urgency, "#757575")


def get_exit_type_icon(exit_type: ExitType) -> str:
    """Get icon for exit type."""
    icons = {
        ExitType.TARGET_HIT: "🎯",
        ExitType.STOP_HIT: "🛑",
        ExitType.TIME_DECAY: "⏰",
        ExitType.REGIME_CHANGE: "🔄",
        ExitType.EXPIRATION: "📅",
        ExitType.MANUAL: "👆",
    }
    return icons.get(exit_type, "❓")
