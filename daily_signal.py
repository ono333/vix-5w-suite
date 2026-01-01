#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Pure Data Emailer
Reads data from 'live_signal_data.json' and sends the report.
"""

import argparse
import datetime as dt
import json
import os
import sys
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set these in your environment variables for security
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
DATA_FILE = Path(__file__).parent / "live_signal_data.json"

def format_html(data):
    """Uses the exact same formatting as the app for consistency."""
    today = dt.date.today().strftime("%B %d, %Y")
    pct = data['percentile']
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    signal_text = ">>> ENTRY SIGNAL ACTIVE <<<" if active else "No Signal"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;padding:20px;max-width:700px;margin:auto;">
    <h1 style="color:#00aadd;text-align:center;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color:#666666;text-align:center;">Thursday Signal Report - {today}</h3>

    <div style="padding:15px;border:1px solid #ddd;margin-bottom:20px;">
    <strong>[MARKET STATE]</strong><br>
    VIX Close: ${data['vix_close']:.2f}<br>
    52w Percentile: {pct:.1f}%<br>
    Current Regime: {data['regime']}<br>
    UVXY Spot: ${data['uvxy_spot']:.2f}
    </div>

    <div style="padding:15px;border:2px solid {'#00aa00' if active else '#aa0000'};background:{'#f8fff8' if active else '#fff8f8'};">
    <strong style="font-size:20px;color:{'#00aa00' if active else '#aa0000'};">{emoji} {signal_text}</strong>
    </div>

    <h2>VARIANTS FROM DASHBOARD</h2>
    """
    for v in data['variants']:
        html += f"""
        <div style="padding:10px;border:1px solid #ddd;margin-bottom:10px;background:#f9f9f9;">
        <strong style="color:#00aadd;">{v['name']}</strong><br>
        <strong>Net:</strong> {v.get('net_position', 'N/A')}<br>
        <strong>Suggested:</strong> {v.get('suggested', 'N/A')}
        </div>
        """
    html += "</body></html>"
    return html

def send_email(recipient, html_content, data):
    if not SMTP_USER or not SMTP_PASS:
        print("[ERROR] SMTP credentials missing in environment.")
        return False

    active = data['signal_active']
    subject = f"{'🟢' if active else '🔴'} VIX Signal: {data['regime']} ({data['percentile']:.1f}%)"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = recipient
    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipient, msg.as_string())
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", help="Recipient email address")
    parser.add_argument("--force", action="store_true", help="Send even if no signal")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"[ERROR] {DATA_FILE} not found. Run Streamlit app first.")
        sys.exit(1)

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    html = format_html(data)

    if args.to:
        if data["signal_active"] or args.force:
            print(f"Sending to {args.to}...")
            if send_email(args.to, html, data):
                print("Success!")
        else:
            print("No signal active. Use --force to send.")

if __name__ == "__main__":
    main()