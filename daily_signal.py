#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Pure Data Emailer
Reads data from 'live_signal_data.json' and sends the compact report.

Usage:
    python daily_signal.py --to onoshin333@gmail.com
    python daily_signal.py --to onoshin333@gmail.com --force
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
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
DATA_FILE = Path(__file__).parent / "live_signal_data.json"


def format_html(data):
    """Compact HTML email with bid/ask data for trading."""
    today = dt.date.today().strftime("%b %d, %Y")
    pct = data['percentile']
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    status = "ENTRY SIGNAL" if active else "HOLD"
    status_color = "#2e7d32" if active else "#c62828"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:12px;background:#fff;color:#333;padding:10px;max-width:700px;margin:auto;">
    
    <div style="text-align:center;border-bottom:2px solid #1f77b4;padding-bottom:8px;margin-bottom:12px;">
        <span style="font-size:18px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="color:#666;font-size:11px;">Thursday Signal • {today}</span>
    </div>

    <table style="width:100%;border-collapse:collapse;margin-bottom:10px;font-size:11px;">
        <tr>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>VIX</b><br>${data['vix_close']:.2f}</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>Percentile</b><br>{pct:.1f}%</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>Regime</b><br>{data['regime']}</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>UVXY</b><br>${data['uvxy_spot']:.2f}</td>
        </tr>
    </table>

    <div style="padding:8px;background:{'#e8f5e9' if active else '#ffebee'};border:1px solid {status_color};text-align:center;margin-bottom:12px;border-radius:4px;">
        <b style="color:{status_color};font-size:14px;">{emoji} {status}</b>
    </div>

    <div style="font-size:13px;font-weight:bold;color:#1f77b4;margin-bottom:8px;">5 DIAGONAL VARIANTS</div>
    """

    for v in data['variants']:
        # Handle both old and new data formats
        long_strike = v.get('long_strike', 0)
        long_exp = v.get('long_exp', v.get('long_expiry', '-'))
        long_dte = v.get('long_dte', '-')
        long_bid = v.get('long_bid', 0)
        long_ask = v.get('long_ask', 0)
        long_mid = v.get('long_mid', v.get('long_price', 0))
        
        short_strike = v.get('short_strike', 0)
        short_exp = v.get('short_exp', v.get('short_expiry', '-'))
        short_dte = v.get('short_dte', '-')
        short_bid = v.get('short_bid', 0)
        short_ask = v.get('short_ask', 0)
        short_mid = v.get('short_mid', v.get('short_price', 0))
        
        net_debit = v.get('net_debit', 0)
        risk = v.get('risk_per_contract', abs(net_debit) * 100)
        target_mult = v.get('target_mult', 1.2)
        target_price = v.get('target_price', long_mid * target_mult)
        stop_mult = v.get('stop_mult', 0.5)
        stop_price = v.get('stop_price', long_mid * stop_mult)
        suggested = v.get('suggested_contracts', v.get('suggested', 1))
        
        html += f"""
        <div style="border:1px solid #ddd;margin-bottom:8px;border-radius:4px;overflow:hidden;">
            <div style="background:#1f77b4;color:#fff;padding:5px 8px;font-size:11px;font-weight:bold;">
                {v['name']} <span style="font-weight:normal;opacity:0.8;">— {v.get('desc', '')[:30]}</span>
            </div>
            <div style="padding:6px;font-size:10px;">
                <table style="width:100%;border-collapse:collapse;margin-bottom:4px;">
                    <tr style="background:#f8f8f8;">
                        <th style="padding:3px;border:1px solid #eee;text-align:left;width:18%;">Leg</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:12%;">Strike</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:22%;">Expiry</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:10%;">DTE</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:12%;">Bid</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:12%;">Ask</th>
                        <th style="padding:3px;border:1px solid #eee;text-align:center;width:14%;">Mid</th>
                    </tr>
                    <tr>
                        <td style="padding:3px;border:1px solid #eee;color:#c62828;font-weight:bold;">SHORT</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${short_strike:.0f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">{short_exp}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">{short_dte}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${short_bid:.2f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${short_ask:.2f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;font-weight:bold;">${short_mid:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px;border:1px solid #eee;color:#2e7d32;font-weight:bold;">LONG</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${long_strike:.0f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">{long_exp}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">{long_dte}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${long_bid:.2f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;">${long_ask:.2f}</td>
                        <td style="padding:3px;border:1px solid #eee;text-align:center;font-weight:bold;">${long_mid:.2f}</td>
                    </tr>
                </table>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#555;padding-top:3px;border-top:1px solid #eee;">
                    <span><b>Net:</b> ${net_debit:.2f} | <b>Risk:</b> ${risk:.0f}/ct</span>
                    <span><b>Target:</b> ${target_price:.2f} ({target_mult}x) | <b>Stop:</b> ${stop_price:.2f} ({stop_mult}x)</span>
                    <span><b>Suggested:</b> {suggested} ct</span>
                </div>
            </div>
        </div>
        """

    html += """
    <div style="font-size:9px;color:#999;text-align:center;margin-top:10px;padding-top:8px;border-top:1px solid #eee;">
        ⚠️ Research only — verify quotes with broker before trading.
    </div>
    </body>
    </html>
    """
    return html


def send_email(recipient, html_content, data):
    if not SMTP_USER or not SMTP_PASS:
        print("[ERROR] SMTP credentials missing in environment.")
        print("        Set SMTP_USER and SMTP_PASS environment variables.")
        return False

    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    subject = f"{emoji} VIX Signal: {data['regime']} ({data['percentile']:.1f}%)"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = recipient
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipient, msg.as_string())
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="VIX 5% Weekly Email Sender")
    parser.add_argument("--to", help="Recipient email address", default="onoshin333@gmail.com")
    parser.add_argument("--force", action="store_true", help="Send even if no signal active")
    parser.add_argument("--preview", action="store_true", help="Save HTML preview instead of sending")
    args = parser.parse_args()

    print("=" * 50)
    print("VIX 5% Weekly - Email Sender")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    if not DATA_FILE.exists():
        print(f"\n[ERROR] {DATA_FILE} not found.")
        print("        Open Streamlit app Live Signals page first to generate data.")
        sys.exit(1)

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Show summary
    print(f"\nVIX: ${data['vix_close']:.2f} | Percentile: {data['percentile']:.1f}%")
    print(f"Regime: {data['regime']} | Signal: {'ACTIVE' if data['signal_active'] else 'HOLD'}")
    print(f"UVXY: ${data['uvxy_spot']:.2f}")
    print(f"Variants: {len(data.get('variants', []))}")

    html = format_html(data)

    if args.preview:
        preview_path = Path(__file__).parent / "email_preview.html"
        with open(preview_path, "w") as f:
            f.write(html)
        print(f"\n[OK] Preview saved to: {preview_path}")
        return

    if not data["signal_active"] and not args.force:
        print("\n[INFO] No signal active. Use --force to send anyway.")
        return

    print(f"\nSending to {args.to}...")
    if send_email(args.to, html, data):
        print("[OK] Email sent successfully!")
    else:
        print("[FAILED] Email not sent.")
        sys.exit(1)


if __name__ == "__main__":
    main()
