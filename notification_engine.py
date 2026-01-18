"""
Notification Engine for VIX 5% Weekly Suite

Handles notifications via email, Slack, etc.
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json
from datetime import datetime


@dataclass
class Notification:
    """A notification message."""
    notification_id: str
    title: str
    message: str
    channel: str = "email"
    priority: str = "normal"
    sent_at: Optional[datetime] = None
    acknowledged: bool = False


class NotificationEngine:
    """Handles sending notifications."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path.home() / ".vix_suite" / "notification_config.json")
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.history: List[Notification] = []
    
    def _load_config(self) -> dict:
        """Load notification configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "email_enabled": False,
            "email_recipient": "",
            "slack_enabled": False,
            "slack_webhook": "",
        }
    
    def save_config(self):
        """Save configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def send_email(self, subject: str, body: str, recipient: Optional[str] = None) -> tuple:
        """Send email notification. Returns (success, message)."""
        import os
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_pass = os.environ.get("SMTP_PASS")
        
        if not smtp_user or not smtp_pass:
            return False, "SMTP credentials not configured"
        
        recipient = recipient or self.config.get("email_recipient")
        if not recipient:
            return False, "No recipient specified"
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = recipient
            msg.attach(MIMEText(body, "html"))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient, msg.as_string())
            
            return True, f"Email sent to {recipient}"
        except Exception as e:
            return False, str(e)
    
    def notify(self, title: str, message: str, channel: str = "email", priority: str = "normal") -> tuple:
        """Send a notification via specified channel."""
        notification = Notification(
            notification_id=f"NOTIF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=title,
            message=message,
            channel=channel,
            priority=priority,
        )
        
        success, result = False, "Unknown channel"
        
        if channel == "email":
            success, result = self.send_email(title, message)
        elif channel == "slack":
            # Slack integration placeholder
            success, result = False, "Slack not implemented"
        
        if success:
            notification.sent_at = datetime.now()
        
        self.history.append(notification)
        return success, result


# Singleton
_notifier: Optional[NotificationEngine] = None

def get_notifier() -> NotificationEngine:
    """Get global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationEngine()
    return _notifier
