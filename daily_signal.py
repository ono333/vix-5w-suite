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
    today = dt.date.today().strftime("%B %d, %Y")
    pct = data['percentile']
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    
    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f7f6; padding: 20px;">
        <div style="max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; border: 1px solid #ddd;">
            <h1 style="color: #1f77b4; text-align: center; margin-bottom: 5px;">VIX 5% WEEKLY SUITE</h1>
            <p style="text-align: center; color: #666;">Signal Report: {today}</p>
            
            <div style="background: #e1f5fe; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="font-weight: bold;">VIX Close:</td><td>${data['vix_close']:.2f}</td>
                        <td style="font-weight: bold;">Percentile:</td><td>{pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="font-weight: bold;">Regime:</td><td>{data['regime']}</td>
                        <td style="font-weight: bold;">UVXY Spot:</td><td>${data['uvxy_spot']:.2f}</td>
                    </tr>
                </table>
            </div>

            <div style="text-align: center; padding: 20px; border: 2px solid {'#2e7d32' if active else '#c62828'}; border-radius: 10px; background: {'#e8f5e9' if active else '#ffebee'};">
                <span style="font-size: 24px; font-weight: bold; color: {'#2e7d32' if active else '#c62828'};">
                    {emoji} {">>> ENTRY SIGNAL ACTIVE <<<" if active else "SIGNAL STATUS: HOLD"}
                </span>
            </div>

            <h2 style="border-bottom: 2px solid #1f77b4; padding-bottom: 10px; margin-top: 30px;">Strategic Variants</h2>
    """

    for v in data['variants']:
        html += f"""
            <div style="margin-bottom: 25px; border: 1px solid #eee; border-radius: 5px; overflow: hidden;">
                <div style="background: #f8f9fa; padding: 10px; font-weight: bold; border-bottom: 1px solid #eee; color: #1f77b4;">
                    {v['name']}
                </div>
                <div style="padding: 15px;">
                    <table style="width: 100%; font-size: 14px; margin-bottom: 10px;">
                        <tr>
                            <td><strong>Position:</strong> {v.get('net_position', 'N/A')}</td>
                            <td><strong>Net Debit:</strong> {v.get('suggested', 'N/A')}</td>
                        </tr>
                    </table>
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                        <tr style="background: #fafafa;">
                            <th style="padding: 5px; border-bottom: 1px solid #ddd;">Leg</th>
                            <th style="padding: 5px; border-bottom: 1px solid #ddd;">Strike</th>
                            <th style="padding: 5px; border-bottom: 1px solid #ddd;">Expiry</th>
                            <th style="padding: 5px; border-bottom: 1px solid #ddd;">Price</th>
                        </tr>
                        <tr>
                            <td style="padding: 5px;">Short Call</td><td>{v.get('short_strike', '-')}</td><td>{v.get('short_expiry', '-')}</td><td>${v.get('short_price', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px;">Long Call</td><td>{v.get('long_strike', '-')}</td><td>{v.get('long_expiry', '-')}</td><td>${v.get('long_price', 0):.2f}</td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    
    html += "</div></body></html>"
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