#!/usr/bin/env python3
"""
Exit Detector for VIX/UVXY Suite

Detects exit conditions and generates exit suggestions.

Exit Types:
- PLANNED: Defined at entry (decay target, time-based, expiration)
- REGIME: Triggered by regime change (highest priority)
- RISK: Triggered by risk thresholds (rare but loud)
- INFORMATIONAL: Status updates only (no action required)

Key Principles:
- Exits are SUGGESTIONS, not automatic orders
- Regime-based logic dominates price-based logic
- Legs are treated independently (short may close before long)
- Each suggestion requires explicit acknowledgment
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from regime_detector import RegimeState, VolatilityRegime
from trade_log import Trade, TradeLeg, LegSide, LegStatus, TradeStatus
from variant_generator import VariantRole


class ExitType(Enum):
    """Types of exit conditions."""
    PLANNED = "planned"           # Expected exit (decay target, time, expiry)
    REGIME = "regime"             # Regime change triggered
    RISK = "risk"                 # Risk threshold triggered
    INFORMATIONAL = "info"        # Status update only


class ExitUrgency(Enum):
    """Urgency level for exit suggestions."""
    INFO = "info"                 # No action required
    ACTIONABLE = "actionable"     # Should consider action
    CRITICAL = "critical"         # Immediate attention needed


class ExitStatus(Enum):
    """Status of an exit event."""
    PENDING = "pending"           # Detected, not yet processed
    EMAILED = "emailed"           # Email sent
    ACKNOWLEDGED = "acknowledged" # User acknowledged
    EXECUTED = "executed"         # Action taken
    IGNORED = "ignored"           # User chose to ignore
    SNOOZED = "snoozed"           # Temporarily delayed


@dataclass
class ExitEvent:
    """A detected exit condition."""
    event_id: str
    trade_id: str
    leg_id: Optional[str]  # None if trade-level
    
    # Event type
    exit_type: ExitType
    urgency: ExitUrgency
    
    # Timing
    detected_at: datetime
    valid_until: datetime
    
    # Recommendation
    suggested_action: str         # "CLOSE_SHORT", "CLOSE_LONG", "CLOSE_ALL", "HOLD"
    rationale: str                # Human-readable explanation
    confidence: str               # "HIGH", "MEDIUM", "LOW"
    
    # Status tracking
    status: ExitStatus = ExitStatus.PENDING
    emailed_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    
    # Metadata
    fingerprint: str = ""         # For deduplication
    retry_count: int = 0
    ignore_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trade_id": self.trade_id,
            "leg_id": self.leg_id,
            "exit_type": self.exit_type.value,
            "urgency": self.urgency.value,
            "detected_at": self.detected_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "suggested_action": self.suggested_action,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "status": self.status.value,
            "emailed_at": self.emailed_at.isoformat() if self.emailed_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "fingerprint": self.fingerprint,
            "retry_count": self.retry_count,
            "ignore_reason": self.ignore_reason,
        }


def _generate_event_id(trade_id: str, exit_type: ExitType) -> str:
    """Generate unique event ID."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"EXIT_{trade_id}_{exit_type.value}_{ts}"


def _generate_fingerprint(trade_id: str, exit_type: ExitType, condition: str) -> str:
    """Generate fingerprint for deduplication."""
    content = f"{trade_id}|{exit_type.value}|{condition}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def detect_planned_exits(
    trade: Trade,
    current_prices: Dict[str, float],
) -> List[ExitEvent]:
    """
    Detect planned exit conditions:
    - Decay target reached
    - Time-based exit
    - Approaching expiration
    """
    events = []
    now = datetime.utcnow()
    
    for leg in trade.legs:
        if leg.status != LegStatus.OPEN:
            continue
        
        current_price = current_prices.get(leg.instrument, leg.current_price)
        
        # Check TP
        if leg.side == LegSide.SHORT:
            # For shorts, TP is below entry (want price to decay)
            if current_price <= leg.tp_price and leg.tp_price > 0:
                events.append(ExitEvent(
                    event_id=_generate_event_id(trade.trade_id, ExitType.PLANNED),
                    trade_id=trade.trade_id,
                    leg_id=leg.leg_id,
                    exit_type=ExitType.PLANNED,
                    urgency=ExitUrgency.ACTIONABLE,
                    detected_at=now,
                    valid_until=now.replace(hour=now.hour + 24),
                    suggested_action="CLOSE_SHORT",
                    rationale=f"Short leg decay target reached. Current: ${current_price:.2f}, Target: ${leg.tp_price:.2f}",
                    confidence="HIGH",
                    fingerprint=_generate_fingerprint(trade.trade_id, ExitType.PLANNED, "short_tp"),
                ))
        else:
            # For longs, TP is above entry
            if current_price >= leg.tp_price and leg.tp_price > 0:
                events.append(ExitEvent(
                    event_id=_generate_event_id(trade.trade_id, ExitType.PLANNED),
                    trade_id=trade.trade_id,
                    leg_id=leg.leg_id,
                    exit_type=ExitType.PLANNED,
                    urgency=ExitUrgency.ACTIONABLE,
                    detected_at=now,
                    valid_until=now.replace(hour=now.hour + 24),
                    suggested_action="CLOSE_LONG",
                    rationale=f"Long leg profit target reached. Current: ${current_price:.2f}, Target: ${leg.tp_price:.2f}",
                    confidence="HIGH",
                    fingerprint=_generate_fingerprint(trade.trade_id, ExitType.PLANNED, "long_tp"),
                ))
        
        # Check expiration proximity
        try:
            exp_date = datetime.fromisoformat(leg.expiration.replace("Z", "+00:00"))
            days_to_exp = (exp_date - now).days
            
            if days_to_exp <= 7 and days_to_exp > 0:
                events.append(ExitEvent(
                    event_id=_generate_event_id(trade.trade_id, ExitType.PLANNED),
                    trade_id=trade.trade_id,
                    leg_id=leg.leg_id,
                    exit_type=ExitType.PLANNED,
                    urgency=ExitUrgency.INFO if days_to_exp > 3 else ExitUrgency.ACTIONABLE,
                    detected_at=now,
                    valid_until=exp_date,
                    suggested_action="REVIEW",
                    rationale=f"Approaching expiration: {days_to_exp} days remaining",
                    confidence="MEDIUM",
                    fingerprint=_generate_fingerprint(trade.trade_id, ExitType.PLANNED, f"exp_{days_to_exp}"),
                ))
        except Exception:
            pass
    
    return events


def detect_regime_exits(
    trade: Trade,
    current_regime: RegimeState,
    previous_regime: Optional[RegimeState],
) -> List[ExitEvent]:
    """
    Detect regime-based exit conditions.
    
    Regime exits have highest priority and override price-based logic.
    """
    events = []
    now = datetime.utcnow()
    
    if previous_regime is None:
        return events
    
    # Check for regime transition
    if current_regime.regime == previous_regime.regime:
        return events
    
    variant_role = trade.variant_role
    
    # Define critical transitions per variant role
    critical_transitions = {
        # V1 Income: Exit on rising/stressed
        VariantRole.INCOME.value: [
            (VolatilityRegime.CALM, VolatilityRegime.RISING),
            (VolatilityRegime.CALM, VolatilityRegime.STRESSED),
        ],
        # V2 Decay: Exit if regime reverses to rising
        VariantRole.DECAY.value: [
            (VolatilityRegime.DECLINING, VolatilityRegime.RISING),
            (VolatilityRegime.DECLINING, VolatilityRegime.STRESSED),
        ],
        # V4 Convex: Don't exit on regime - hold for explosion
        VariantRole.CONVEX.value: [],
    }
    
    transitions = critical_transitions.get(variant_role, [])
    transition = (previous_regime.regime, current_regime.regime)
    
    if transition in transitions:
        # Determine action based on structure
        if trade.structure == "diagonal":
            suggested_action = "CLOSE_SHORT"
            rationale = f"Regime flipped {previous_regime.regime.value} → {current_regime.regime.value}. Close short leg first."
        else:
            suggested_action = "REVIEW"
            rationale = f"Regime changed: {previous_regime.regime.value} → {current_regime.regime.value}. Review position."
        
        events.append(ExitEvent(
            event_id=_generate_event_id(trade.trade_id, ExitType.REGIME),
            trade_id=trade.trade_id,
            leg_id=None,  # Trade-level
            exit_type=ExitType.REGIME,
            urgency=ExitUrgency.CRITICAL if current_regime.regime == VolatilityRegime.STRESSED else ExitUrgency.ACTIONABLE,
            detected_at=now,
            valid_until=now.replace(hour=now.hour + 48),
            suggested_action=suggested_action,
            rationale=rationale,
            confidence="HIGH",
            fingerprint=_generate_fingerprint(trade.trade_id, ExitType.REGIME, str(transition)),
        ))
    
    return events


def detect_risk_exits(
    trade: Trade,
    current_prices: Dict[str, float],
    max_loss_pct: float = 0.50,
) -> List[ExitEvent]:
    """
    Detect risk-based exit conditions.
    
    These are rare but loud alerts for significant losses.
    """
    events = []
    now = datetime.utcnow()
    
    # Calculate total position value
    total_entry_value = 0.0
    total_current_value = 0.0
    
    for leg in trade.legs:
        if leg.status != LegStatus.OPEN:
            continue
        
        current_price = current_prices.get(leg.instrument, leg.current_price)
        entry_value = leg.entry_price * abs(leg.quantity) * 100
        current_value = current_price * abs(leg.quantity) * 100
        
        total_entry_value += entry_value
        total_current_value += current_value
    
    if total_entry_value > 0:
        loss_pct = (total_entry_value - total_current_value) / total_entry_value
        
        if loss_pct >= max_loss_pct:
            events.append(ExitEvent(
                event_id=_generate_event_id(trade.trade_id, ExitType.RISK),
                trade_id=trade.trade_id,
                leg_id=None,
                exit_type=ExitType.RISK,
                urgency=ExitUrgency.CRITICAL,
                detected_at=now,
                valid_until=now.replace(hour=now.hour + 24),
                suggested_action="CLOSE_ALL",
                rationale=f"Loss threshold exceeded: {loss_pct:.1%} loss (threshold: {max_loss_pct:.0%})",
                confidence="LOW",  # Price-based = low confidence
                fingerprint=_generate_fingerprint(trade.trade_id, ExitType.RISK, f"loss_{loss_pct:.2f}"),
            ))
    
    return events


def detect_all_exits(
    trades: List[Trade],
    current_regime: RegimeState,
    previous_regime: Optional[RegimeState],
    current_prices: Dict[str, float],
) -> List[ExitEvent]:
    """
    Detect all exit conditions for a list of trades.
    
    Returns deduplicated list of exit events.
    """
    all_events = []
    seen_fingerprints = set()
    
    for trade in trades:
        if trade.status != TradeStatus.OPEN:
            continue
        
        # Planned exits
        planned = detect_planned_exits(trade, current_prices)
        
        # Regime exits (highest priority)
        regime = detect_regime_exits(trade, current_regime, previous_regime)
        
        # Risk exits
        risk = detect_risk_exits(trade, current_prices)
        
        # Combine and deduplicate
        for event in planned + regime + risk:
            if event.fingerprint not in seen_fingerprints:
                all_events.append(event)
                seen_fingerprints.add(event.fingerprint)
    
    # Sort by urgency (critical first) then by type (regime first)
    urgency_order = {ExitUrgency.CRITICAL: 0, ExitUrgency.ACTIONABLE: 1, ExitUrgency.INFO: 2}
    type_order = {ExitType.REGIME: 0, ExitType.RISK: 1, ExitType.PLANNED: 2, ExitType.INFORMATIONAL: 3}
    
    all_events.sort(key=lambda e: (urgency_order.get(e.urgency, 99), type_order.get(e.exit_type, 99)))
    
    return all_events


class ExitEventStore:
    """
    Persistent storage for exit events.
    
    Tracks event lifecycle and prevents duplicate emails.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.home() / ".vix_suite" / "exit_events.json"
        
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.events: Dict[str, ExitEvent] = {}
        self._load()
    
    def _load(self):
        """Load events from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                
                for event_data in data.get("events", []):
                    event = self._deserialize_event(event_data)
                    self.events[event.event_id] = event
            except Exception as e:
                print(f"Warning: Could not load exit events: {e}")
    
    def _save(self):
        """Save events to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "events": [e.to_dict() for e in self.events.values()],
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save exit events: {e}")
    
    def _deserialize_event(self, data: Dict[str, Any]) -> ExitEvent:
        """Deserialize event from dict."""
        data["exit_type"] = ExitType(data["exit_type"])
        data["urgency"] = ExitUrgency(data["urgency"])
        data["status"] = ExitStatus(data["status"])
        data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        data["valid_until"] = datetime.fromisoformat(data["valid_until"])
        
        if data.get("emailed_at"):
            data["emailed_at"] = datetime.fromisoformat(data["emailed_at"])
        if data.get("acknowledged_at"):
            data["acknowledged_at"] = datetime.fromisoformat(data["acknowledged_at"])
        if data.get("executed_at"):
            data["executed_at"] = datetime.fromisoformat(data["executed_at"])
        
        return ExitEvent(**data)
    
    def add_event(self, event: ExitEvent) -> bool:
        """
        Add event if not duplicate.
        
        Returns True if added, False if duplicate.
        """
        # Check for existing event with same fingerprint
        for existing in self.events.values():
            if existing.fingerprint == event.fingerprint:
                if existing.status not in [ExitStatus.EXECUTED, ExitStatus.IGNORED]:
                    return False
        
        self.events[event.event_id] = event
        self._save()
        return True
    
    def mark_emailed(self, event_id: str):
        """Mark event as emailed."""
        if event_id in self.events:
            self.events[event_id].emailed_at = datetime.utcnow()
            self.events[event_id].status = ExitStatus.EMAILED
            self._save()
    
    def acknowledge(self, event_id: str):
        """Mark event as acknowledged."""
        if event_id in self.events:
            self.events[event_id].acknowledged_at = datetime.utcnow()
            self.events[event_id].status = ExitStatus.ACKNOWLEDGED
            self._save()
    
    def mark_executed(self, event_id: str):
        """Mark event as executed."""
        if event_id in self.events:
            self.events[event_id].executed_at = datetime.utcnow()
            self.events[event_id].status = ExitStatus.EXECUTED
            self._save()
    
    def ignore(self, event_id: str, reason: str = ""):
        """Mark event as ignored."""
        if event_id in self.events:
            self.events[event_id].status = ExitStatus.IGNORED
            self.events[event_id].ignore_reason = reason
            self._save()
    
    def snooze(self, event_id: str, hours: int = 24):
        """Snooze event for specified hours."""
        if event_id in self.events:
            self.events[event_id].status = ExitStatus.SNOOZED
            self.events[event_id].valid_until = datetime.utcnow().replace(
                hour=datetime.utcnow().hour + hours
            )
            self._save()
    
    def get_pending_events(self) -> List[ExitEvent]:
        """Get events that need attention."""
        now = datetime.utcnow()
        return [
            e for e in self.events.values()
            if e.status in [ExitStatus.PENDING, ExitStatus.EMAILED]
            and e.valid_until > now
        ]
    
    def get_events_for_email(self) -> List[ExitEvent]:
        """Get events that should be emailed."""
        return [
            e for e in self.events.values()
            if e.status == ExitStatus.PENDING
            and e.urgency in [ExitUrgency.ACTIONABLE, ExitUrgency.CRITICAL]
            and e.emailed_at is None
            and e.retry_count < 3
        ]
    
    def increment_retry(self, event_id: str):
        """Increment retry count for failed email."""
        if event_id in self.events:
            self.events[event_id].retry_count += 1
            self._save()


# Global instance
_exit_store: Optional[ExitEventStore] = None


def get_exit_store(storage_path: Optional[Path] = None) -> ExitEventStore:
    """Get or create the global exit event store."""
    global _exit_store
    if _exit_store is None:
        _exit_store = ExitEventStore(storage_path)
    return _exit_store


def get_exit_urgency_color(urgency: ExitUrgency) -> str:
    """Get display color for urgency."""
    colors = {
        ExitUrgency.INFO: "#17a2b8",       # Cyan
        ExitUrgency.ACTIONABLE: "#ffc107", # Yellow
        ExitUrgency.CRITICAL: "#dc3545",   # Red
    }
    return colors.get(urgency, "#6c757d")


def get_exit_type_icon(exit_type: ExitType) -> str:
    """Get icon for exit type."""
    icons = {
        ExitType.PLANNED: "📅",
        ExitType.REGIME: "🔄",
        ExitType.RISK: "⚠️",
        ExitType.INFORMATIONAL: "ℹ️",
    }
    return icons.get(exit_type, "•")
