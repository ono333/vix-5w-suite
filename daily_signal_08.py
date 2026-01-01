#!/usr/bin/env python3
"""
VIX 5% Weekly - Thursday Emailer (Pure Display)

Loads signals.json from Live Signals export and sends formatted HTML email.
Zero computation. 100% match with Streamlit app.

Workflow:
    1. Run: python live_signals_export.py (or export from Streamlit)
    2. Run: python daily_signal.py --email

Or combined:
    python live_signals_export.py && python daily_signal.py --email

Cron (4:30pm ET):
    30 20 * * 4 cd /home/shin/vix_suite && ./venv/bin/python live_signals_export.py && ./venv/bin/python daily_signal.py --email
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
from email.header import Header

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EMAIL = "onoshin333@gmail.com"
SCRIPT_DIR = Path(__file__).parent
JSON_PATH = SCRIPT_DIR / "signals.json"


# =============================================================================
# HELPERS
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="VIX Weekly Signal Emailer")
    parser.add_argument("--email", nargs="?", const=DEFAULT_EMAIL, default=None)
    parser.add_argument("--json-path", type=str, default=str(JSON_PATH))
    parser.add_argument("--force", action="store_true", help="Send even if no signal")
    parser.add_argument("--preview", action="store_true", help="Print HTML without sending")
    return parser.parse_args()


def load_signals(path: str) -> dict:
    """Load signals.json exported from Live Signals page."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# HTML FORMATTING (White background, large font, readable)
# =============================================================================

def format_html(data: dict) -> str:
    """
    Format beautiful white-background HTML email.
    Large font, clean layout, fits one screen.
    """
    today = dt.date.today().strftime("%B %d, %Y")
    pct = data["percentile"]
    active = data["signal_active"]
    regime = data["regime"]
    vix = data["vix_close"]
    uvxy = data["uvxy_spot"]
    
    # Signal styling
    if active:
        sig_color = "#008800"
        sig_bg = "#f0fff0"
        sig_border = "#00aa00"
        sig_text = ">>> ENTRY SIGNAL ACTIVE <<<"
    else:
        sig_color = "#880000"
        sig_bg = "#fff0f0"
        sig_border = "#aa0000"
        sig_text = "--- HOLD - No Entry Signal ---"
    
    # Regime color
    regime_colors = {
        "ULTRA_LOW": "#008800",
        "LOW": "#44aa44",
        "MID": "#888800",
        "HIGH": "#cc6600",
        "EXTREME": "#cc0000",
    }
    regime_color = regime_colors.get(regime, "#000000")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="background:#ffffff;color:#222222;font-family:Georgia,'Times New Roman',serif;font-size:16px;line-height:1.6;padding:20px;max-width:700px;margin:auto;">

<!-- Header -->
<div style="text-align:center;border-bottom:3px solid #00aadd;padding-bottom:15px;margin-bottom:20px;">
    <h1 style="color:#00aadd;margin:0;font-size:28px;">VIX 5% WEEKLY SUITE</h1>
    <p style="color:#666;margin:5px 0 0;font-size:16px;">Thursday Signal Report — {today}</p>
</div>

<!-- Market State -->
<div style="background:#f8f8f8;border:1px solid #ddd;border-left:4px solid #00aadd;padding:15px;margin-bottom:20px;">
    <h2 style="color:#00aadd;margin:0 0 10px;font-size:20px;">MARKET STATE</h2>
    <table style="font-size:16px;border-collapse:collapse;">
        <tr><td style="padding:3px 15px 3px 0;color:#666;">VIX Close:</td><td style="font-weight:bold;">${vix:.2f}</td></tr>
        <tr><td style="padding:3px 15px 3px 0;color:#666;">52w Percentile:</td><td style="font-weight:bold;">{pct:.1f}%</td></tr>
        <tr><td style="padding:3px 15px 3px 0;color:#666;">Current Regime:</td><td style="font-weight:bold;color:{regime_color};">{regime}</td></tr>
        <tr><td style="padding:3px 15px 3px 0;color:#666;">UVXY Spot:</td><td style="font-weight:bold;">${uvxy:.2f}</td></tr>
    </table>
</div>

<!-- Signal Status -->
<div style="background:{sig_bg};border:2px solid {sig_border};padding:15px;margin-bottom:25px;text-align:center;">
    <h2 style="color:{sig_color};margin:0;font-size:22px;">{sig_text}</h2>
    <p style="color:#666;margin:8px 0 0;font-size:14px;">
        Percentile ({pct:.1f}%) {'≤' if active else '>'} threshold ({data.get('threshold', 35):.0f}%)
    </p>
</div>

<!-- Variants Header -->
<h2 style="color:#00aadd;border-bottom:2px solid #00aadd;padding-bottom:8px;margin:25px 0 15px;font-size:22px;">
    5 DIAGONAL VARIANTS
</h2>
"""
    
    # Variants
    for i, v in enumerate(data["variants"]):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        
        html += f"""
<div style="background:{bg};border:1px solid #ddd;border-left:4px solid #00aadd;padding:15px;margin-bottom:12px;">
    <h3 style="color:#00aadd;margin:0 0 8px;font-size:18px;">{v['name']}</h3>
    <p style="color:#888;margin:0 0 12px;font-size:13px;font-style:italic;">{v.get('desc', '')}</p>
    
    <table style="font-size:14px;border-collapse:collapse;width:100%;">
        <tr>
            <td style="padding:4px 0;color:#008800;font-weight:bold;width:100px;">LONG (Buy):</td>
            <td style="padding:4px 0;">{v['long_leg']}</td>
        </tr>
        <tr>
            <td style="padding:4px 0;color:#880000;font-weight:bold;">SHORT (Sell):</td>
            <td style="padding:4px 0;">{v['short_leg']}</td>
        </tr>
        <tr style="border-top:1px solid #eee;">
            <td style="padding:8px 0 4px;font-weight:bold;">Net Position:</td>
            <td style="padding:8px 0 4px;">{v['net_position']}</td>
        </tr>
        <tr>
            <td style="padding:4px 0;color:#008800;">Target:</td>
            <td style="padding:4px 0;">{v.get('target', 'N/A')}</td>
        </tr>
        <tr>
            <td style="padding:4px 0;color:#880000;">Stop:</td>
            <td style="padding:4px 0;">{v.get('stop', 'N/A')}</td>
        </tr>
        <tr style="border-top:1px solid #eee;">
            <td style="padding:8px 0 4px;font-weight:bold;">Suggested:</td>
            <td style="padding:8px 0 4px;font-weight:bold;color:#00aadd;">{v.get('suggested_contracts', 'N/A')}</td>
        </tr>
    </table>
</div>
"""
    
    # Footer
    html += f"""
<!-- Position Sizing -->
<div style="background:#fff8e0;border:1px solid #ddcc00;padding:12px;margin:20px 0;text-align:center;">
    <strong style="color:#886600;">POSITION SIZING REMINDER</strong><br>
    <span style="font-size:14px;">Risk 1-2% of portfolio per trade • Max 3-5 contracts • Always use stops</span>
</div>

<!-- Disclaimer -->
<div style="text-align:center;padding:15px;border-top:1px solid #ddd;margin-top:20px;">
    <p style="color:#aa0000;font-size:13px;margin:0;">
        ⚠️ Research tool only — not financial advice.<br>
        Always verify quotes with your broker before trading.
    </p>
    <p style="color:#888;font-size:11px;margin:10px 0 0;">
        Generated: {data.get('generated_at', 'N/A')[:19]}
    </p>
</div>

</body>
</html>"""
    
    return html


# =============================================================================
# EMAIL SENDING
# =============================================================================

def send_email(to: str, html: str, data: dict) -> bool:
    """Send HTML email with emoji subject."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        print("[ERROR] SMTP_USER or SMTP_PASS not set in environment")
        return False
    
    active = data["signal_active"]
    regime = data["regime"]
    pct = data["percentile"]
    
    # Emoji subject for easy inbox scanning
    if active:
        subject = f"\U0001F7E2 [ENTRY] VIX {regime} ({pct:.0f}%) - Diagonals Ready"
    else:
        subject = f"\U0001F534 [HOLD] VIX {regime} ({pct:.0f}%)"
    
    try:
        msg = MIMEMultipart()
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = smtp_user
        msg["To"] = to
        msg.attach(MIMEText(html, "html", "utf-8"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = _parse_args()
    
    print("=" * 55)
    print("VIX 5% Weekly - Thursday Signal Emailer")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    # Load signals.json
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"\n[ERROR] signals.json not found at: {json_path}")
        print("        Run: python live_signals_export.py first")
        print("        Or export from Streamlit Live Signals page")
        sys.exit(1)
    
    print(f"\n[...] Loading signals from: {json_path}")
    data = load_signals(str(json_path))
    
    # Display summary
    print(f"\n[OK] Loaded signals from {data.get('generated_at', 'unknown')[:19]}")
    print(f"     VIX: ${data['vix_close']:.2f} | Percentile: {data['percentile']:.1f}%")
    print(f"     Regime: {data['regime']} | UVXY: ${data['uvxy_spot']:.2f}")
    print(f"     Signal: {'ACTIVE' if data['signal_active'] else 'HOLD'}")
    print(f"     Variants: {len(data['variants'])}")
    
    # Check if data is stale (> 6 hours old)
    try:
        gen_time = dt.datetime.fromisoformat(data["generated_at"])
        age_hours = (dt.datetime.now() - gen_time).total_seconds() / 3600
        if age_hours > 6:
            print(f"\n[WARN] Data is {age_hours:.1f} hours old - consider re-exporting")
    except:
        pass
    
    # Generate HTML
    print("\n[...] Generating HTML report...")
    html = format_html(data)
    print(f"[OK] HTML generated ({len(html)} bytes)")
    
    # Preview mode
    if args.preview:
        print("\n" + "=" * 55)
        print("HTML PREVIEW (saved to preview.html):")
        print("=" * 55)
        preview_path = SCRIPT_DIR / "preview.html"
        with open(preview_path, "w") as f:
            f.write(html)
        print(f"Open in browser: file://{preview_path}")
        return
    
    # Send email
    if args.email:
        active = data["signal_active"]
        if active or args.force:
            print(f"\n[EMAIL] Sending to {args.email}...")
            if send_email(args.email, html, data):
                print("[OK] Email sent successfully!")
            else:
                print("[ERROR] Failed to send email")
                sys.exit(1)
        else:
            print(f"\n[INFO] No signal active - email skipped")
            print("       Use --force to send anyway")
    else:
        print("\n[INFO] No --email flag - skipping email")
        print("       Use: python daily_signal.py --email")
    
    sys.exit(0 if data["signal_active"] else 1)


if __name__ == "__main__":
    main()
