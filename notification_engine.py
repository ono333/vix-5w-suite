"""
Notification Engine for VIX 5% Weekly Suite
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class NotificationConfig:
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    default_recipient: str = ""

class Notifier:
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig(
            smtp_user=os.environ.get("SMTP_USER", ""),
            smtp_pass=os.environ.get("SMTP_PASS", ""),
            default_recipient=os.environ.get("NOTIFY_EMAIL", ""),
        )
    
    def send_email(self, subject: str, body: str, recipient: str = None, html: bool = False) -> bool:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        if not self.config.smtp_user or not self.config.smtp_pass:
            print("SMTP credentials not configured")
            return False
        
        to_addr = recipient or self.config.default_recipient
        if not to_addr:
            print("No recipient specified")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.config.smtp_user
            msg['To'] = to_addr
            msg.attach(MIMEText(body, 'html' if html else 'plain', 'utf-8'))
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_pass)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Email failed: {e}")
            return False

_notifier: Optional[Notifier] = None

def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
