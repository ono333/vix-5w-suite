"""
Notification Engine for VIX 5% Weekly Suite

Exports:
    - get_notifier
    - Notifier
    - NotificationEvent
"""

from __future__ import annotations

import datetime as dt
import json
import smtplib
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any

from enums import VolatilityRegime, VariantRole, ExitUrgency


NOTIFICATION_LOG_PATH = Path.home() / ".vix_suite" / "notification_log.json"


@dataclass
class NotificationEvent:
    """A notification event."""
    event_id: str
    timestamp: dt.datetime
    event_type: str  # "signal", "exit", "regime_change", "system"
    title: str
    message: str
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    
    # Delivery status
    delivered: bool = False
    delivery_method: Optional[str] = None
    delivery_time: Optional[dt.datetime] = None
    
    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "title": self.title,
            "message": self.message,
            "urgency": self.urgency,
            "delivered": self.delivered,
            "delivery_method": self.delivery_method,
            "delivery_time": self.delivery_time.isoformat() if self.delivery_time else None,
            "metadata": self.metadata,
        }


class Notifier:
    """Handles notifications for the suite."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.email_enabled = bool(self.config.get("email_enabled", False))
        self.email_to = self.config.get("email_to", "")
        self.smtp_server = self.config.get("smtp_server", "")
        self.smtp_port = self.config.get("smtp_port", 587)
        self.smtp_user = self.config.get("smtp_user", "")
        self.smtp_password = self.config.get("smtp_password", "")
        
        self.log_path = NOTIFICATION_LOG_PATH
        self.history: List[NotificationEvent] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load notification history from disk."""
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                # Just keep the last 100 events
                events = data.get("events", [])[-100:]
                self.history = []  # We don't need to reconstruct full objects
            except:
                self.history = []
    
    def _save_event(self, event: NotificationEvent) -> None:
        """Save event to log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.log_path.exists():
                with open(self.log_path, "r") as f:
                    data = json.load(f)
            else:
                data = {"events": []}
            
            data["events"].append(event.to_dict())
            # Keep last 500 events
            data["events"] = data["events"][-500:]
            
            with open(self.log_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save notification: {e}")
    
    def _send_email(self, event: NotificationEvent) -> bool:
        """Send email notification."""
        if not self.email_enabled or not self.email_to:
            return False
        
        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            msg["Subject"] = f"[VIX Suite] {event.title}"
            
            body = f"""
{event.title}
{'=' * len(event.title)}

{event.message}

---
Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Type: {event.event_type}
Urgency: {event.urgency}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Email send failed: {e}")
            return False
    
    def notify(
        self,
        title: str,
        message: str,
        event_type: str = "system",
        urgency: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationEvent:
        """Send a notification."""
        event = NotificationEvent(
            event_id=f"notif_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=dt.datetime.now(),
            event_type=event_type,
            title=title,
            message=message,
            urgency=urgency,
            metadata=metadata or {},
        )
        
        # Try email for high urgency
        if urgency in ("high", "critical") and self.email_enabled:
            if self._send_email(event):
                event.delivered = True
                event.delivery_method = "email"
                event.delivery_time = dt.datetime.now()
        
        # Always log
        self._save_event(event)
        
        return event
    
    def notify_signal_generated(
        self,
        variant_role: VariantRole,
        regime: VolatilityRegime,
        vix_level: float,
    ) -> NotificationEvent:
        """Notify about a new signal."""
        return self.notify(
            title=f"New Signal: {variant_role.value}",
            message=f"Signal generated for {variant_role.value} in {regime.value} regime (VIX: {vix_level:.2f})",
            event_type="signal",
            urgency="normal",
            metadata={"variant": variant_role.value, "regime": regime.value, "vix": vix_level},
        )
    
    def notify_exit_triggered(
        self,
        trade_id: str,
        exit_type: str,
        urgency: ExitUrgency,
        pnl: float,
    ) -> NotificationEvent:
        """Notify about exit trigger."""
        urgency_map = {
            ExitUrgency.IMMEDIATE: "critical",
            ExitUrgency.SOON: "high",
            ExitUrgency.OPTIONAL: "normal",
        }
        return self.notify(
            title=f"Exit Signal: {exit_type}",
            message=f"Exit triggered for trade {trade_id}: {exit_type}. Current P&L: ${pnl:,.2f}",
            event_type="exit",
            urgency=urgency_map.get(urgency, "normal"),
            metadata={"trade_id": trade_id, "exit_type": exit_type, "pnl": pnl},
        )
    
    def notify_regime_change(
        self,
        old_regime: VolatilityRegime,
        new_regime: VolatilityRegime,
        vix_level: float,
    ) -> NotificationEvent:
        """Notify about regime change."""
        return self.notify(
            title=f"Regime Change: {old_regime.value} → {new_regime.value}",
            message=f"Volatility regime changed from {old_regime.value} to {new_regime.value}. VIX: {vix_level:.2f}",
            event_type="regime_change",
            urgency="high",
            metadata={"old": old_regime.value, "new": new_regime.value, "vix": vix_level},
        )


# Singleton instance
_notifier_instance: Optional[Notifier] = None


def get_notifier(config: Optional[Dict[str, Any]] = None) -> Notifier:
    """Get or create Notifier instance."""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = Notifier(config)
    return _notifier_instance
