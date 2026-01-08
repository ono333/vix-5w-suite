#!/usr/bin/env python3
"""
Notification Engine for VIX/UVXY Suite

Handles email notifications for:
- New signals (Thursday 4:30 PM)
- Exit suggestions
- Regime changes
- System alerts

Key Principles:
- Email is an exception channel, not chatter
- Emails are sent ONLY when conditions are met
- Never duplicate emails for same event
- Include direct links back to Streamlit app

Configuration via environment variables:
- VIX_SMTP_HOST: SMTP server
- VIX_SMTP_PORT: SMTP port
- VIX_SMTP_USER: Username/email
- VIX_SMTP_PASS: Password/app password
- VIX_EMAIL_TO: Recipient email
- VIX_EMAIL_FROM: Sender email (defaults to SMTP_USER)
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from exit_detector import ExitEvent, ExitType, ExitUrgency


@dataclass
class EmailConfig:
    """Email configuration."""
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    email_to: str
    email_from: str
    enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Load configuration from environment variables."""
        return cls(
            smtp_host=os.environ.get("VIX_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("VIX_SMTP_PORT", "587")),
            smtp_user=os.environ.get("VIX_SMTP_USER", ""),
            smtp_pass=os.environ.get("VIX_SMTP_PASS", ""),
            email_to=os.environ.get("VIX_EMAIL_TO", ""),
            email_from=os.environ.get("VIX_EMAIL_FROM", os.environ.get("VIX_SMTP_USER", "")),
            enabled=os.environ.get("VIX_NOTIFICATIONS_ENABLED", "true").lower() == "true",
        )
    
    def is_valid(self) -> bool:
        """Check if configuration is valid for sending emails."""
        return all([
            self.smtp_host,
            self.smtp_port > 0,
            self.smtp_user,
            self.smtp_pass,
            self.email_to,
            self.email_from,
            self.enabled,
        ])


def build_exit_email(
    event: ExitEvent,
    trade_context: Optional[Dict[str, Any]] = None,
    app_url: str = "http://localhost:8501",
) -> tuple[str, str]:
    """
    Build email subject and body for an exit event.
    
    Returns (subject, body_html)
    """
    # Subject line - deterministic format
    urgency_prefix = {
        ExitUrgency.CRITICAL: "🔴 CRITICAL",
        ExitUrgency.ACTIONABLE: "⚠️ ACTION",
        ExitUrgency.INFO: "ℹ️ INFO",
    }
    
    prefix = urgency_prefix.get(event.urgency, "")
    type_label = event.exit_type.value.upper()
    
    subject = f"[UVXY][{event.trade_id}][{type_label}] {event.suggested_action}"
    if prefix:
        subject = f"{prefix} - {subject}"
    
    # Body
    body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; background: {'#dc3545' if event.urgency == ExitUrgency.CRITICAL else '#ffc107' if event.urgency == ExitUrgency.ACTIONABLE else '#17a2b8'}; color: white; }}
        .header h1 {{ margin: 0; font-size: 18px; }}
        .content {{ padding: 20px; }}
        .recommendation {{ background: #f8f9fa; padding: 15px; border-left: 4px solid {'#dc3545' if event.urgency == ExitUrgency.CRITICAL else '#ffc107'}; margin: 15px 0; }}
        .recommendation h2 {{ margin: 0 0 10px 0; font-size: 16px; }}
        .details {{ margin: 15px 0; }}
        .details table {{ width: 100%; border-collapse: collapse; }}
        .details td {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        .details td:first-child {{ color: #666; width: 120px; }}
        .button {{ display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 15px; }}
        .disclaimer {{ font-size: 12px; color: #999; margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{prefix} Exit Suggestion</h1>
        </div>
        <div class="content">
            <div class="recommendation">
                <h2>Suggested Action: {event.suggested_action}</h2>
                <p>{event.rationale}</p>
            </div>
            
            <div class="details">
                <table>
                    <tr>
                        <td>Trade ID</td>
                        <td><strong>{event.trade_id}</strong></td>
                    </tr>
                    <tr>
                        <td>Exit Type</td>
                        <td>{event.exit_type.value.title()}</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td>{event.confidence}</td>
                    </tr>
                    <tr>
                        <td>Detected</td>
                        <td>{event.detected_at.strftime('%Y-%m-%d %H:%M UTC')}</td>
                    </tr>
                    <tr>
                        <td>Valid Until</td>
                        <td>{event.valid_until.strftime('%Y-%m-%d %H:%M UTC')}</td>
                    </tr>
                </table>
            </div>
            
            <a href="{app_url}" class="button">Review in Dashboard</a>
            
            <p class="disclaimer">
                This is a suggestion, not an execution command.<br>
                Review the full context in the dashboard before taking action.
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, body_html


def build_signal_email(
    batch_id: str,
    regime: str,
    variants_summary: List[Dict[str, Any]],
    app_url: str = "http://localhost:8501",
) -> tuple[str, str]:
    """
    Build email for new signal batch.
    
    Returns (subject, body_html)
    """
    subject = f"[VIX Suite] New Signal Batch: {batch_id}"
    
    # Build variants table
    variants_rows = ""
    for v in variants_summary:
        status_color = "#28a745" if v.get("status") == "TRADE" else "#dc3545"
        variants_rows += f"""
        <tr>
            <td>{v.get('role', 'Unknown')}</td>
            <td style="color: {status_color};">{v.get('status', 'Unknown')}</td>
            <td>{v.get('robustness', 0):.0f}</td>
            <td>{v.get('alloc_pct', 0):.1%}</td>
        </tr>
        """
    
    body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; background: #007bff; color: white; }}
        .header h1 {{ margin: 0; font-size: 18px; }}
        .content {{ padding: 20px; }}
        .regime {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .button {{ display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 New Signal Batch Generated</h1>
        </div>
        <div class="content">
            <p><strong>Batch ID:</strong> {batch_id}</p>
            
            <div class="regime">
                <strong>Current Regime:</strong> {regime}
            </div>
            
            <table>
                <tr>
                    <th>Variant</th>
                    <th>Status</th>
                    <th>Robustness</th>
                    <th>Allocation</th>
                </tr>
                {variants_rows}
            </table>
            
            <a href="{app_url}" class="button">Review Signals</a>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, body_html


def build_regime_alert_email(
    previous_regime: str,
    current_regime: str,
    confidence: float,
    app_url: str = "http://localhost:8501",
) -> tuple[str, str]:
    """
    Build email for regime change alert.
    
    Returns (subject, body_html)
    """
    subject = f"[VIX Suite] Regime Change: {previous_regime} → {current_regime}"
    
    body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; background: #fd7e14; color: white; }}
        .header h1 {{ margin: 0; font-size: 18px; }}
        .content {{ padding: 20px; }}
        .transition {{ font-size: 24px; text-align: center; padding: 20px; }}
        .arrow {{ color: #666; margin: 0 15px; }}
        .button {{ display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔄 Volatility Regime Change Detected</h1>
        </div>
        <div class="content">
            <div class="transition">
                <span>{previous_regime.upper()}</span>
                <span class="arrow">→</span>
                <span>{current_regime.upper()}</span>
            </div>
            
            <p><strong>Confidence:</strong> {confidence:.0%}</p>
            <p>Review your open positions and variant allocations.</p>
            
            <a href="{app_url}" class="button">Review Positions</a>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, body_html


class NotificationEngine:
    """
    Email notification engine with throttling and logging.
    """
    
    def __init__(self, config: Optional[EmailConfig] = None):
        if config is None:
            config = EmailConfig.from_env()
        
        self.config = config
        self.log_path = Path.home() / ".vix_suite" / "notification_log.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._sent_log: Dict[str, datetime] = {}
        self._load_log()
    
    def _load_log(self):
        """Load sent email log."""
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                self._sent_log = {
                    k: datetime.fromisoformat(v) 
                    for k, v in data.items()
                }
            except Exception:
                self._sent_log = {}
    
    def _save_log(self):
        """Save sent email log."""
        try:
            data = {k: v.isoformat() for k, v in self._sent_log.items()}
            with open(self.log_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _can_send(self, event_id: str, throttle_hours: int = 24) -> bool:
        """Check if we can send email (throttling)."""
        if event_id in self._sent_log:
            last_sent = self._sent_log[event_id]
            hours_since = (datetime.utcnow() - last_sent).total_seconds() / 3600
            if hours_since < throttle_hours:
                return False
        return True
    
    def send_email(
        self,
        subject: str,
        body_html: str,
        event_id: Optional[str] = None,
    ) -> bool:
        """
        Send an email.
        
        Returns True if sent successfully.
        """
        if not self.config.is_valid():
            print("Warning: Email configuration invalid, skipping send")
            return False
        
        if event_id and not self._can_send(event_id):
            print(f"Throttled: email for {event_id} already sent recently")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.email_from
            msg["To"] = self.config.email_to
            
            # Add HTML content
            msg.attach(MIMEText(body_html, "html"))
            
            # Send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_pass)
                server.send_message(msg)
            
            # Log success
            if event_id:
                self._sent_log[event_id] = datetime.utcnow()
                self._save_log()
            
            print(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            print(f"Email failed: {e}")
            return False
    
    def send_exit_alert(
        self,
        event: ExitEvent,
        trade_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send exit suggestion email."""
        subject, body = build_exit_email(event, trade_context)
        return self.send_email(subject, body, event.event_id)
    
    def send_signal_alert(
        self,
        batch_id: str,
        regime: str,
        variants_summary: List[Dict[str, Any]],
    ) -> bool:
        """Send new signal batch email."""
        subject, body = build_signal_email(batch_id, regime, variants_summary)
        return self.send_email(subject, body, batch_id)
    
    def send_regime_alert(
        self,
        previous_regime: str,
        current_regime: str,
        confidence: float,
    ) -> bool:
        """Send regime change email."""
        event_id = f"regime_{previous_regime}_{current_regime}"
        subject, body = build_regime_alert_email(previous_regime, current_regime, confidence)
        return self.send_email(subject, body, event_id)
    
    def process_pending_exits(
        self,
        events: List[ExitEvent],
    ) -> List[str]:
        """
        Process list of exit events and send emails.
        
        Returns list of event IDs that were emailed.
        """
        sent = []
        
        for event in events:
            # Only email actionable/critical events
            if event.urgency not in [ExitUrgency.ACTIONABLE, ExitUrgency.CRITICAL]:
                continue
            
            if self.send_exit_alert(event):
                sent.append(event.event_id)
        
        return sent


# Global instance
_notifier: Optional[NotificationEngine] = None


def get_notifier() -> NotificationEngine:
    """Get or create the global notification engine."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationEngine()
    return _notifier
