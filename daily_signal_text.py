#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (Text/HTML Version)

Generates clean HTML report with 3 variant diagonals and emails it every Thursday.
Run via cron at 4:30pm ET (20:30 UTC):

    30 20 * * 4 /home/shin/vix_suite/venv/bin/python /home/shin/vix_suite/daily_signal_text.py --email

Or manually:
    python daily_signal_text.py
    python daily_signal_text.py --email
    python daily_signal_text.py --email your@email.com
    python daily_signal_text.py --json

SMTP Setup (Gmail App Password):
    export SMTP_SERVER="smtp.gmail.com"
    export SMTP_PORT="587"
    export SMTP_USER="your.email@gmail.com"
    export SMTP_PASS="your-app-password"
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

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from math import log, sqrt, exp

# Constants
DEFAULT_EMAIL = "onoshin333@gmail.com"
DEFAULT_THRESHOLD = 0.35  # 35%

# SMTP settings from environment
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")

# Variant configs
VARIANT_CONFIGS = [
    {"name": "Base", "otm_pts": 10.0, "long_dte_weeks": 26},
    {"name": "Aggressive", "otm_pts": 15.0, "long_dte_weeks": 26},
    {"name": "Conservative", "otm_pts": 5.0, "long_dte_weeks": 13},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIX 5% Weekly Signal Generator")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Signal threshold (default: 0.35)",
    )
    parser.add_argument(
        "--email",
        nargs="?",
        const=DEFAULT_EMAIL,
        default=None,
        help="Send email to this address",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of text report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force email even if no signal active",
    )
    return parser.parse_args()


def load_vix_weekly() -> pd.Series:
    """Load latest weekly VIX close."""
    start = dt.date.today() - dt.timedelta(days=365 * 2)
    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download("^VIX", start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("Failed to load VIX data from yfinance")
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    weekly = df[col].resample("W-FRI").last().dropna()
    return weekly


def compute_vix_percentile(vix_series: pd.Series, lookback_weeks: int = 52) -> float:
    """Compute rolling percentile for latest VIX."""
    if len(vix_series) < lookback_weeks:
        lookback_weeks = len(vix_series)
    
    recent = vix_series.iloc[-lookback_weeks:]
    current_val = float(recent.iloc[-1])
    
    # Count how many values are below current
    below_count = int((recent.values < current_val).sum())
    percentile = below_count / (len(recent) - 1) if len(recent) > 1 else 0.5
    
    return float(percentile)


def determine_regime(percentile: float) -> str:
    """Determine VIX regime based on percentile."""
    if percentile <= 0.10:
        return "ULTRA_LOW"
    elif percentile <= 0.25:
        return "LOW"
    elif percentile <= 0.50:
        return "MID"
    elif percentile <= 0.75:
        return "HIGH"
    else:
        return "EXTREME"


def get_uvxy_spot() -> float:
    """Get current UVXY spot price."""
    try:
        uvxy = yf.download("UVXY", period="5d", progress=False)
        if uvxy.empty:
            return 0.0
        
        # Handle multi-level columns
        if isinstance(uvxy.columns, pd.MultiIndex):
            uvxy.columns = uvxy.columns.get_level_values(0)
        
        col = "Adj Close" if "Adj Close" in uvxy.columns else "Close"
        return float(uvxy[col].iloc[-1])
    except Exception as e:
        print(f"[WARN] Failed to get UVXY spot: {e}")
        return 0.0


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    # Ensure all inputs are scalar floats
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def get_diagonal_quote(uvxy_spot: float, vix_close: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get diagonal spread quote using Black-Scholes (synthetic)."""
    otm_pts = float(config["otm_pts"])
    long_dte_weeks = int(config["long_dte_weeks"])
    long_dte_days = long_dte_weeks * 7
    short_dte_days = 7  # Weekly short
    r = 0.03  # Risk-free rate
    
    # VIX-based volatility estimate for UVXY (UVXY is ~1.5x leveraged)
    sigma = float(vix_close) / 100 * 1.5
    sigma = max(0.30, min(sigma, 2.0))  # Clamp to reasonable range
    
    long_T = long_dte_days / 365.0
    short_T = short_dte_days / 365.0
    long_K = uvxy_spot + otm_pts
    short_K = uvxy_spot + otm_pts - 2  # Short is slightly closer to ATM
    
    # Ensure strikes are positive
    long_K = max(long_K, uvxy_spot * 1.05)
    short_K = max(short_K, uvxy_spot * 1.02)
    
    long_mid = bs_call_price(uvxy_spot, long_K, long_T, r, sigma)
    long_bid = long_mid * 0.95
    long_ask = long_mid * 1.05
    
    short_mid = bs_call_price(uvxy_spot, short_K, short_T, r, sigma)
    short_bid = short_mid * 0.95
    short_ask = short_mid * 1.05
    
    net_debit_mid = long_mid - short_mid
    
    # Calculate expiration dates
    long_exp_date = dt.date.today() + dt.timedelta(days=long_dte_days)
    short_exp_date = dt.date.today() + dt.timedelta(days=short_dte_days)
    
    return {
        "name": config["name"],
        "uvxy_spot": uvxy_spot,
        "long_strike": long_K,
        "long_bid": long_bid,
        "long_ask": long_ask,
        "long_mid": long_mid,
        "long_dte_days": long_dte_days,
        "long_exp": long_exp_date.strftime("%Y-%m-%d"),
        "short_strike": short_K,
        "short_bid": short_bid,
        "short_ask": short_ask,
        "short_mid": short_mid,
        "short_dte_days": short_dte_days,
        "short_exp": short_exp_date.strftime("%Y-%m-%d"),
        "net_debit_mid": net_debit_mid,
    }


def format_text_report(vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> str:
    """Format plain text report (ASCII-safe)."""
    signal_active = vix_data['percentile'] <= threshold * 100
    
    lines = [
        "=" * 60,
        "VIX 5% Weekly Suite - Thursday Signal Report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "[MARKET STATE]",
        "-" * 40,
        f"VIX Close:        ${vix_data['close']:.2f}",
        f"52w Percentile:   {vix_data['percentile']:.1f}%",
        f"Current Regime:   {vix_data['regime']}",
        "",
    ]
    
    if signal_active:
        lines.append(">>> ENTRY SIGNAL ACTIVE <<<")
        lines.append(f"    Percentile ({vix_data['percentile']:.1f}%) <= threshold ({threshold*100:.0f}%)")
    else:
        lines.append("--- HOLD - No Entry Signal ---")
        lines.append(f"    Percentile ({vix_data['percentile']:.1f}%) > threshold ({threshold*100:.0f}%)")
    
    lines.append("")
    lines.append("[3 DIAGONAL VARIANTS]")
    lines.append("-" * 40)
    
    for qd in quote_datas:
        lines.extend([
            "",
            f"=== {qd['name']} ===",
            f"UVXY Spot: ${qd['uvxy_spot']:.2f}",
            "",
            "LONG LEG (Buy):",
            f"  UVXY {qd['long_exp']} ${qd['long_strike']:.0f} Call",
            f"  Bid: ${qd['long_bid']:.2f}  Ask: ${qd['long_ask']:.2f}  Mid: ${qd['long_mid']:.2f}",
            f"  DTE: {qd['long_dte_days']} days",
            "",
            "SHORT LEG (Sell):",
            f"  UVXY {qd['short_exp']} ${qd['short_strike']:.0f} Call",
            f"  Bid: ${qd['short_bid']:.2f}  Ask: ${qd['short_ask']:.2f}  Mid: ${qd['short_mid']:.2f}",
            f"  DTE: {qd['short_dte_days']} days",
            "",
            "NET POSITION:",
            f"  Net Debit (mid): ${qd['net_debit_mid']:.2f}",
        ])
    
    lines.extend([
        "",
        "=" * 60,
        "[!] Research tool only - not financial advice.",
        "    Always verify quotes with your broker.",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def format_html_report(vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> str:
    """Format HTML report (for email)."""
    signal_active = vix_data['percentile'] <= threshold * 100
    today = dt.date.today().strftime("%B %d, %Y")
    
    if signal_active:
        signal_html = '<h2 style="color: #00ff00;">&gt;&gt;&gt; ENTRY SIGNAL ACTIVE &lt;&lt;&lt;</h2>'
        signal_detail = f'<p>Percentile ({vix_data["percentile"]:.1f}%) &lt;= threshold ({threshold*100:.0f}%)</p>'
    else:
        signal_html = '<h2 style="color: #ff6666;">--- HOLD - No Entry Signal ---</h2>'
        signal_detail = f'<p>Percentile ({vix_data["percentile"]:.1f}%) &gt; threshold ({threshold*100:.0f}%)</p>'
    
    # Build variants HTML
    variants_html = ""
    for qd in quote_datas:
        variants_html += f"""
        <div style="border: 1px solid #333355; padding: 10px; margin: 10px 0;">
            <h3 style="color: #ffff00;">{qd['name']}</h3>
            <p>UVXY Spot: ${qd['uvxy_spot']:.2f}</p>
            
            <h4 style="color: #00ff00;">LONG LEG (Buy):</h4>
            <p>UVXY {qd['long_exp']} ${qd['long_strike']:.0f} Call</p>
            <p>Bid: ${qd['long_bid']:.2f} | Ask: ${qd['long_ask']:.2f} | Mid: ${qd['long_mid']:.2f}</p>
            <p>DTE: {qd['long_dte_days']} days</p>
            
            <h4 style="color: #ff6666;">SHORT LEG (Sell):</h4>
            <p>UVXY {qd['short_exp']} ${qd['short_strike']:.0f} Call</p>
            <p>Bid: ${qd['short_bid']:.2f} | Ask: ${qd['short_ask']:.2f} | Mid: ${qd['short_mid']:.2f}</p>
            <p>DTE: {qd['short_dte_days']} days</p>
            
            <h4 style="color: #00ffff;">NET POSITION:</h4>
            <p>Net Debit (mid): ${qd['net_debit_mid']:.2f}</p>
        </div>
        """
    
    html = f"""
    <html>
    <body style="background-color: #001133; color: #dddddd; font-family: monospace; padding: 20px; max-width: 700px; margin: auto;">
        <h1 style="color: #00ffff;">VIX 5% WEEKLY SUITE</h1>
        <h3 style="color: #888888;">Thursday Signal Report - {today}</h3>
        
        <div style="border: 1px solid #333355; padding: 10px; margin: 10px 0;">
            <h2 style="color: #00ffff;">[MARKET STATE]</h2>
            <p>VIX Close: ${vix_data['close']:.2f}</p>
            <p>52w Percentile: {vix_data['percentile']:.1f}%</p>
            <p>Current Regime: <strong style="color: #ffff00;">{vix_data['regime']}</strong></p>
        </div>
        
        {signal_html}
        {signal_detail}
        
        <h2 style="color: #00ffff;">[3 DIAGONAL VARIANTS]</h2>
        {variants_html}
        
        <p style="color: #888888; font-size: 11px; margin-top: 20px;">
            [!] Research tool only - not financial advice. Always verify quotes with your broker.
        </p>
    </body>
    </html>
    """
    
    return html


def send_signal_email(recipient: str, vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> bool:
    """Send HTML email report."""
    if not SMTP_USER or not SMTP_PASS:
        print("[WARN] SMTP credentials not configured")
        print("       Set SMTP_USER and SMTP_PASS environment variables")
        return False
    
    try:
        signal_active = vix_data['percentile'] <= threshold * 100
        regime = vix_data['regime']
        
        # Subject line (ASCII-safe)
        if signal_active:
            subject = f"[ENTRY SIGNAL] VIX {regime} Regime ({vix_data['percentile']:.1f}%)"
        else:
            subject = f"[WEEKLY] VIX {regime} Regime ({vix_data['percentile']:.1f}%)"
        
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = SMTP_USER
        msg['To'] = recipient
        
        html = format_html_report(vix_data, quote_datas, threshold)
        html_part = MIMEText(html, 'html', 'utf-8')
        msg.attach(html_part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        print(f"Email error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = _parse_args()
    
    print("=" * 50)
    print("VIX 5% Weekly - Thursday Signal Check")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print()
    
    # Load VIX data
    print("[...] Loading VIX data...")
    try:
        vix_weekly = load_vix_weekly()
        vix_close = float(vix_weekly.iloc[-1])
        print(f"[OK] VIX loaded: ${vix_close:.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to load VIX: {e}")
        sys.exit(1)
    
    # Compute percentile and regime
    percentile = compute_vix_percentile(vix_weekly)
    percentile_pct = percentile * 100
    regime = determine_regime(percentile)
    
    print(f"[OK] Percentile: {percentile_pct:.1f}%")
    print(f"[OK] Regime: {regime}")
    
    vix_data = {
        "close": vix_close,
        "percentile": percentile_pct,
        "regime": regime,
    }
    
    signal_active = percentile <= args.threshold
    
    print()
    if signal_active:
        print(f">>> ENTRY SIGNAL ACTIVE <<< (percentile <= {args.threshold*100:.0f}%)")
    else:
        print(f"--- HOLD --- (percentile > {args.threshold*100:.0f}%)")
    print()
    
    # Get UVXY spot
    print("[...] Loading UVXY spot...")
    uvxy_spot = get_uvxy_spot()
    if uvxy_spot <= 0:
        print("[ERROR] Failed to get UVXY spot price")
        sys.exit(1)
    print(f"[OK] UVXY: ${uvxy_spot:.2f}")
    
    # Generate quotes for all variants
    print("[...] Generating diagonal quotes...")
    quote_datas = []
    for config in VARIANT_CONFIGS:
        qd = get_diagonal_quote(uvxy_spot, vix_close, config)
        quote_datas.append(qd)
        print(f"[OK] {config['name']}: Net Debit ${qd['net_debit_mid']:.2f}")
    
    print()
    
    # Output
    if args.json:
        output = {
            "vix": vix_data,
            "variants": quote_datas,
            "signal_active": signal_active,
            "threshold": args.threshold,
        }
        print(json.dumps(output, indent=2))
    else:
        report = format_text_report(vix_data, quote_datas, args.threshold)
        print(report)
    
    # Send email
    if args.email and (signal_active or args.force):
        print()
        print(f"[EMAIL] Sending to {args.email}...")
        if send_signal_email(args.email, vix_data, quote_datas, args.threshold):
            print("[OK] Email sent successfully!")
        else:
            print("[ERROR] Failed to send email")
            print("        Check SMTP_USER and SMTP_PASS environment variables")
    elif args.email and not signal_active:
        print()
        print(f"[INFO] No signal active - email skipped (use --force to override)")
    
    sys.exit(0 if signal_active else 1)


if __name__ == "__main__":
    main()
