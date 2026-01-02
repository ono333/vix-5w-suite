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
    """Generates a high-detail HTML report matching the dashboard's depth."""
    today = dt.date.today().strftime("%B %d, %Y")
    pct = data['percentile']
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    status_color = "#2e7d32" if active else "#c62828"
    status_bg = "#e8f5e9" if active else "#ffebee"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f6; padding: 20px;">
        <div style="max-width: 800px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            
            <div style="text-align: center; border-bottom: 2px solid #1f77b4; padding-bottom: 20px; margin-bottom: 25px;">
                <h1 style="color: #1f77b4; margin: 0; font-size: 28px;">VIX 5% WEEKLY SUITE</h1>
                <p style="color: #666; font-size: 16px; margin: 5px 0 0 0;">Thursday Signal Report • {today}</p>
            </div>

            <div style="display: flex; justify-content: space-between; margin-bottom: 25px;">
                <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 5px; margin-right: 10px; border-left: 4px solid #1f77b4;">
                    <span style="font-size: 12px; color: #666; text-transform: uppercase;">Market State</span><br>
                    <strong style="font-size: 18px;">VIX: ${data['vix_close']:.2f}</strong><br>
                    <span style="color: #444;">Percentile: {pct:.1f}%</span>
                </div>
                <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4;">
                    <span style="font-size: 12px; color: #666; text-transform: uppercase;">Regime & Asset</span><br>
                    <strong style="font-size: 18px;">{data['regime']}</strong><br>
                    <span style="color: #444;">UVXY: ${data['uvxy_spot']:.2f}</span>
                </div>
            </div>

            <div style="background: {status_bg}; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; border: 1px solid {status_color};">
                <span style="font-size: 22px; font-weight: bold; color: {status_color};">
                    {emoji} {">>> ENTRY SIGNAL ACTIVE <<<" if active else "SIGNAL STATUS: HOLD / NO ENTRY"}
                </span>
            </div>

            <h2 style="color: #333; font-size: 20px; margin-bottom: 15px;">Strategy Variants</h2>
    """

    for v in data['variants']:
        html += f"""
            <div style="margin-bottom: 25px; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden;">
                <div style="background: #1f77b4; color: white; padding: 10px 15px; font-weight: bold;">
                    {v['name']}
                </div>
                <div style="padding: 15px;">
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
                        <tr>
                            <td style="width: 50%; padding: 5px 0;"><strong>Position:</strong> {v.get('net_position', 'N/A')}</td>
                            <td style="width: 50%; padding: 5px 0; text-align: right;"><strong>Net Debit:</strong> {v.get('suggested', 'N/A')}</td>
                        </tr>
                    </table>
                    
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <thead>
                            <tr style="background: #f0f2f6; text-align: left;">
                                <th style="padding: 8px; border: 1px solid #ddd;">Leg</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Strike</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Expiry</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Price</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; color: #c62828;">SHORT CALL</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{v.get('short_strike', '-')}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{v.get('short_expiry', '-')}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">${v.get('short_price', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; color: #2e7d32;">LONG CALL</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{v.get('long_strike', '-')}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">{v.get('long_expiry', '-')}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">${v.get('long_price', 0):.2f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        """
    
    html += """
            <div style="text-align: center; color: #999; font-size: 11px; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px;">
                This is an automated signal generated by the VIX 5% Weekly Suite server. 
                Trading options involves significant risk.
            </div>
        </div>
    </body>
    </html>
    """
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