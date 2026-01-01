#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (All-in-One)

Generates live signals data and sends formatted HTML email in ONE command.
No separate JSON export needed.

Usage:
    python daily_signal.py --email                      # Send to default
    python daily_signal.py --email onoshin333@gmail.com # Send to specific
    python daily_signal.py --email --force              # Send even if no signal
    python daily_signal.py --preview                    # Save HTML preview

Cron (4:30pm ET = 20:30 UTC):
    30 20 * * 4 . /home/shin/.bashrc; /home/shin/vix_suite/venv/bin/python /home/shin/vix_suite/daily_signal.py --email
"""

import argparse
import datetime as dt
import json
import os
import sys
import smtplib
from pathlib import Path
from typing import Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EMAIL = "onoshin333@gmail.com"
DEFAULT_THRESHOLD = 0.35
SCRIPT_DIR = Path(__file__).parent


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="VIX Weekly Signal Emailer")
    parser.add_argument("--email", nargs="?", const=DEFAULT_EMAIL, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--force", action="store_true", help="Send even if no signal")
    parser.add_argument("--preview", action="store_true", help="Save HTML preview only")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def _scalar(val) -> float:
    """Extract scalar from pandas objects."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return float(val.iloc[0] if isinstance(val, pd.Series) else val.iloc[0, 0])
    elif isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


def load_vix_weekly() -> pd.Series:
    """Load VIX weekly data."""
    df = yf.download("^VIX", period="2y", progress=False)
    if df.empty:
        raise RuntimeError("Failed to load VIX data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].resample("W-FRI").last().dropna()


def load_uvxy_spot() -> float:
    """Load current UVXY spot."""
    try:
        df = yf.download("UVXY", period="5d", progress=False)
        if df.empty:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return _scalar(df[col].iloc[-1])
    except Exception as e:
        print(f"[WARN] UVXY load failed: {e}")
        return 0.0


def compute_percentile(series: pd.Series, lookback: int = 52) -> float:
    """Compute rolling percentile (0-1)."""
    recent = series.iloc[-lookback:]
    if len(recent) < 2:
        return 0.5
    current = _scalar(recent.iloc[-1])
    return float((recent.values.flatten() < current).sum()) / (len(recent) - 1)


def get_regime(pct: float) -> str:
    """Determine VIX regime."""
    if pct <= 0.10:
        return "ULTRA_LOW"
    elif pct <= 0.25:
        return "LOW"
    elif pct <= 0.50:
        return "MID"
    elif pct <= 0.75:
        return "HIGH"
    return "EXTREME"


# =============================================================================
# BLACK-SCHOLES PRICING
# =============================================================================

def bs_call(S: float, K: float, T: float, r: float, sig: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0)
    d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


# =============================================================================
# 5 VARIANTS (Exact match to Streamlit Live Signals)
# =============================================================================

def generate_variants(uvxy_spot: float, vix_close: float) -> List[Dict[str, Any]]:
    """
    Generate 5 diagonal variants matching Streamlit Live Signals exactly.
    """
    r = 0.03
    base_sig = vix_close / 100 * 1.5
    base_sig = max(0.30, min(base_sig, 2.0))
    
    configs = [
        {"name": "Baseline (26w)", "desc": "Standard 6-month diagonal", 
         "otm": 10, "dte_w": 26, "sig_mult": 1.0, "target": 1.20, "stop": 0.50},
        {"name": "Aggressive (1w)", "desc": "Ultra-short for quick theta",
         "otm": 3, "dte_w": 1, "sig_mult": 0.8, "target": 1.50, "stop": 0.30},
        {"name": "Aggressive (3w)", "desc": "Short DTE, faster decay",
         "otm": 5, "dte_w": 3, "sig_mult": 0.8, "target": 1.30, "stop": 0.40},
        {"name": "Tighter Exit (1.5x)", "desc": "Quick profit target",
         "otm": 10, "dte_w": 15, "sig_mult": 1.0, "target": 1.50, "stop": 0.60},
        {"name": "Static Benchmark", "desc": "Conservative reference",
         "otm": 15, "dte_w": 26, "sig_mult": 1.0, "target": 1.20, "stop": 0.50},
    ]
    
    variants = []
    today = dt.date.today()
    
    for cfg in configs:
        otm = cfg["otm"]
        dte_w = cfg["dte_w"]
        sig = base_sig * cfg["sig_mult"]
        target_mult = cfg["target"]
        stop_mult = cfg["stop"]
        
        long_dte = dte_w * 7
        short_dte = 7
        
        long_K = round(uvxy_spot + otm, 0)
        short_K = round(uvxy_spot + otm - 2, 0)
        long_K = max(long_K, round(uvxy_spot * 1.02, 0))
        short_K = max(short_K, round(uvxy_spot * 1.01, 0))
        
        long_mid = bs_call(uvxy_spot, long_K, long_dte / 365, r, sig)
        short_mid = bs_call(uvxy_spot, short_K, short_dte / 365, r, sig)
        
        net_debit = long_mid - short_mid
        risk = abs(net_debit) * 100
        
        target_val = long_mid * target_mult
        stop_val = long_mid * stop_mult
        
        long_exp = (today + dt.timedelta(days=long_dte)).strftime("%b %d, %Y")
        short_exp = (today + dt.timedelta(days=short_dte)).strftime("%b %d, %Y")
        
        # Suggested contracts based on $2500 risk budget
        suggested = max(1, min(5, int(2500 / risk))) if risk > 0 else 2
        
        variants.append({
            "name": cfg["name"],
            "desc": cfg["desc"],
            "long_leg": f"UVXY {long_exp} ${long_K:.0f}C @ ${long_mid:.2f} (DTE: {long_dte}d)",
            "short_leg": f"UVXY {short_exp} ${short_K:.0f}C @ ${short_mid:.2f} (DTE: {short_dte}d)",
            "net_position": f"Net Debit: ${net_debit:.2f} | Risk: ~${risk:.0f}/contract",
            "target": f"Target: ${target_val:.2f} ({target_mult:.1f}x)",
            "stop": f"Stop: ${stop_val:.2f} ({stop_mult:.1f}x)",
            "suggested_contracts": f"{suggested} contracts (~${suggested * risk:.0f} total risk)",
            # Raw values
            "long_strike": long_K,
            "long_mid": long_mid,
            "short_strike": short_K,
            "short_mid": short_mid,
            "net_debit_raw": net_debit,
            "risk_per_contract": risk,
        })
    
    return variants


# =============================================================================
# GENERATE ALL SIGNALS DATA
# =============================================================================

def generate_signals(threshold: float = 0.35) -> Dict[str, Any]:
    """Generate complete signals data."""
    vix_series = load_vix_weekly()
    vix_close = _scalar(vix_series.iloc[-1])
    uvxy_spot = load_uvxy_spot()
    
    percentile = compute_percentile(vix_series)
    regime = get_regime(percentile)
    signal_active = percentile <= threshold
    
    variants = generate_variants(uvxy_spot, vix_close)
    
    return {
        "generated_at": dt.datetime.now().isoformat(),
        "date": dt.date.today().isoformat(),
        "vix_close": round(vix_close, 2),
        "percentile": round(percentile * 100, 1),
        "percentile_raw": round(percentile, 4),
        "regime": regime,
        "signal_active": signal_active,
        "threshold": threshold * 100,
        "uvxy_spot": round(uvxy_spot, 2),
        "variants": variants,
    }


# =============================================================================
# HTML FORMATTING (White background, large font, readable)
# =============================================================================

def format_html(data: dict) -> str:
    """Format beautiful white-background HTML email."""
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
    <p style="color:#666;margin:5px 0 0;font-size:16px;">Thursday Signal Report - {today}</p>
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
        Percentile ({pct:.1f}%) {'<=' if active else '>'} threshold ({data.get('threshold', 35):.0f}%)
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
    <span style="font-size:14px;">Risk 1-2% of portfolio per trade | Max 3-5 contracts | Always use stops</span>
</div>

<!-- Disclaimer -->
<div style="text-align:center;padding:15px;border-top:1px solid #ddd;margin-top:20px;">
    <p style="color:#aa0000;font-size:13px;margin:0;">
        Research tool only - not financial advice.<br>
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
    
    # Generate all signals data
    print("\n[...] Loading market data...")
    try:
        data = generate_signals(args.threshold)
    except Exception as e:
        print(f"[ERROR] Failed to generate signals: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display summary
    print(f"[OK] VIX: ${data['vix_close']:.2f}")
    print(f"[OK] Percentile: {data['percentile']:.1f}%")
    print(f"[OK] Regime: {data['regime']}")
    print(f"[OK] UVXY: ${data['uvxy_spot']:.2f}")
    
    print()
    if data["signal_active"]:
        print(f">>> ENTRY SIGNAL ACTIVE <<< (pct <= {args.threshold*100:.0f}%)")
    else:
        print(f"--- HOLD --- (pct > {args.threshold*100:.0f}%)")
    
    print(f"\n[OK] Generated {len(data['variants'])} variants:")
    for v in data["variants"]:
        print(f"     - {v['name']}: Net ${v['net_debit_raw']:.2f}")
    
    # JSON output mode
    if args.json:
        print("\n" + json.dumps(data, indent=2, default=str))
        return
    
    # Generate HTML
    print("\n[...] Generating HTML report...")
    html = format_html(data)
    print(f"[OK] HTML generated ({len(html)} bytes)")
    
    # Preview mode
    if args.preview:
        preview_path = SCRIPT_DIR / "preview.html"
        with open(preview_path, "w") as f:
            f.write(html)
        print(f"\n[OK] Preview saved to: {preview_path}")
        print(f"     Open in browser: file://{preview_path}")
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
