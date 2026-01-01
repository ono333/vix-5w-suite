#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer

Generates clean HTML report with 5 variant diagonals and emails it every Thursday.
Run via cron at 4:30pm ET (20:30 UTC):

    30 20 * * 4 /home/shin/vix_suite/venv/bin/python /home/shin/vix_suite/daily_signal.py --email

Or manually:
    python daily_signal.py
    python daily_signal.py --email
    python daily_signal.py --email your@email.com
    python daily_signal.py --json

SMTP Setup (Gmail App Password):
    export SMTP_SERVER="smtp.gmail.com"
    export SMTP_PORT="587"
    export SMTP_USER="your.email@gmail.com"
    export SMTP_PASS="your-app-password"

Edit DEFAULT_EMAIL below for hardcoded delivery.
"""

import argparse
import datetime as dt
import json
import os
import sys
import smtplib
from pathlib import Path
from typing import Dict, Any, Optional, List
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

# Variant configs (expanded to 5)
VARIANT_CONFIGS = [
    {"name": "Aggressive", "otm_pts_long": 15.0, "otm_pts_short": 16.0, "long_dte_weeks": 26, "desc": "Higher theta bleed"},
    {"name": "Moderate", "otm_pts_long": 10.0, "otm_pts_short": 11.0, "long_dte_weeks": 26, "desc": "Balanced"},
    {"name": "Conservative", "otm_pts_long": 5.0, "otm_pts_short": 6.0, "long_dte_weeks": 26, "desc": "Lower risk"},
    {"name": "Far OTM Long", "otm_pts_long": 20.0, "otm_pts_short": 21.0, "long_dte_weeks": 26, "desc": "Cheaper entry, bigger upside"},
    {"name": "Closer Short", "otm_pts_long": 10.0, "otm_pts_short": 8.0, "long_dte_weeks": 26, "desc": "Potential credit spread"},
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
        help="Send email to this address (default: hardcoded)",
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
    start = dt.date.today() - dt.timedelta(days=365 * 2)  # 2 years for percentile
    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download("^VIX", start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("Failed to load VIX data from yfinance")
    weekly = df["Close"].resample("W-FRI").last().dropna()
    return weekly

def compute_vix_percentile(vix_series: pd.Series, lookback_weeks: int = 52) -> float:
    """Compute rolling percentile for latest VIX."""
    recent = vix_series.iloc[-lookback_weeks:]
    return (recent.iloc[-1] - recent.min()) / (recent.max() - recent.min() + 1e-6)

def determine_regime(percentile: float) -> str:
    """Determine VIX regime based on percentile."""
    if percentile <= 0.10:
        return "ULTRA_LOW"
    elif percentile <= 0.30:
        return "LOW"
    elif percentile <= 0.70:
        return "MID"
    else:
        return "HIGH"

def get_diagonal_quote(vix_close: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get diagonal spread quote using Black-Scholes (synthetic)."""
    otm_pts_long = config["otm_pts_long"]
    otm_pts_short = config["otm_pts_short"]
    long_dte_weeks = config["long_dte_weeks"]
    long_dte_days = long_dte_weeks * 7
    short_dte_days = 1 * 7  # Weekly short
    r = 0.03  # Risk-free rate
    sigma_mult = 1.0
    sigma = vix_close / 100 * sigma_mult  # Base vol from VIX
    uvxy = yf.download("UVXY", period="1d")["Close"].iloc[-1]  # Latest UVXY spot

    long_T = long_dte_days / 365
    short_T = short_dte_days / 365
    long_K = uvxy + otm_pts_long
    short_K = uvxy + otm_pts_short

    long_mid = bs_call_price(uvxy, long_K, long_T, r, sigma)
    long_bid = long_mid * 0.95
    long_ask = long_mid * 1.05

    short_mid = bs_call_price(uvxy, short_K, short_T, r, sigma)
    short_bid = short_mid * 0.95
    short_ask = short_mid * 1.05

    net_debit_mid = long_mid - short_mid
    risk_per_contract = abs(net_debit_mid) * 100  # Assuming 1 contract = 100 shares

    return {
        "long_bid": long_bid,
        "long_ask": long_ask,
        "long_mid": long_mid,
        "long_K": long_K,
        "long_dte_days": long_dte_days,
        "long_expiry": (dt.date.today() + dt.timedelta(days=long_dte_days)).strftime("%b %d"),
        "short_bid": short_bid,
        "short_ask": short_ask,
        "short_mid": short_mid,
        "short_K": short_K,
        "short_dte_days": short_dte_days,
        "short_expiry": (dt.date.today() + dt.timedelta(days=short_dte_days)).strftime("%b %d"),
        "net_debit_mid": net_debit_mid,
        "risk_per_contract": risk_per_contract,
        "current_price": uvxy,
    }

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def format_html_report(vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> str:
    """Format compact HTML report with emojis and 5 variants."""
    today = dt.date.today().strftime("%B %d, %Y")
    percentile = vix_data['percentile'] * 100
    signal_emoji = "🟢" if percentile <= threshold * 100 else "🔴"
    signal_text = ">>> ENTRY SIGNAL ACTIVE <<<" if percentile <= threshold * 100 else "No Signal Active"
    html = f"""
    <html>
    <body style="background-color: #001133; color: #dddddd; font-family: monospace; padding: 10px; max-width: 600px; margin: auto; line-height: 1.2;">
    <h1 style="color: #00ffff; margin-bottom: 5px;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color: #cccccc; margin-top: 0;">Thursday Signal Report - {today}</h3>

    <h2 style="color: #00ffff; margin-bottom: 5px;">[MARKET STATE]</h2>
    <p style="margin: 0;">VIX Close: ${vix_data['close']:.2f}</p>
    <p style="margin: 0;">52w Percentile: {percentile:.1f}%</p>
    <p style="margin: 0;">Current Regime: {vix_data['regime']}</p>

    <h2 style="color: #00ff00; margin-bottom: 5px;">{signal_emoji} {signal_text}</h2>
    <p style="margin: 0;">Percentile ({percentile:.1f}%) ≤ threshold ({threshold*100:.0f}%)</p>

    <h2 style="color: #00ffff; margin-bottom: 5px;">[5 VARIANTS]</h2>
    """

    for i, qd in enumerate(quote_datas, 1):
        debit_str = f"${qd['net_debit_mid']:.2f}" if qd['net_debit_mid'] > 0 else f"Credit +${abs(qd['net_debit_mid']):.2f}"
        html += f"""
        <h3 style="color: #ffff00; margin-bottom: 5px;">{i}. {qd['name']} ({qd['desc']})</h3>
        <p style="margin: 0;">UVXY Spot: ${qd['current_price']:.2f}</p>
        <h4 style="color: #00ff00; margin-bottom: 5px; margin-top: 5px;">LONG LEG (Buy):</h4>
        <p style="margin: 0;">UVXY {qd['long_expiry']} ${qd['long_K']:.0f}C</p>
        <p style="margin: 0;">Bid: ${qd['long_bid']:.2f} Ask: ${qd['long_ask']:.2f} Mid: ${qd['long_mid']:.2f}</p>
        <p style="margin: 0;">DTE: {qd['long_dte_days']} days</p>
        <h4 style="color: #ff0000; margin-bottom: 5px; margin-top: 5px;">SHORT LEG (Sell):</h4>
        <p style="margin: 0;">UVXY {qd['short_expiry']} ${qd['short_K']:.0f}C</p>
        <p style="margin: 0;">Bid: ${qd['short_bid']:.2f} Ask: ${qd['short_ask']:.2f} Mid: ${qd['short_mid']:.2f}</p>
        <p style="margin: 0;">DTE: {qd['short_dte_days']} days</p>
        <h4 style="color: #cccccc; margin-bottom: 5px; margin-top: 5px;">NET POSITION:</h4>
        <p style="margin: 0;">Net Debit (mid): {debit_str} | Risk/Contract: ~${qd['risk_per_contract']:.0f}</p>
        """

    html += """
    <h2 style="color: #00ffff; margin-bottom: 5px;">POSITION SIZING SUGGESTION</h2>
    <p style="margin: 0;">Suggested Contracts: 2–3</p>
    <p style="margin: 0;">Total Risk: ~$1,600–$2,400</p>

    <p style="color: #ffff00; margin-top: 10px;">⚠️ Research tool only — not financial advice. Verify quotes with your broker.</p>
    </body>
    </html>
    """

    return html

def send_signal_email(recipient: str, vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> bool:
    """Send HTML email report with emoji in subject."""
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not all([smtp_server, smtp_user, smtp_pass]):
        return False

    percentile = vix_data['percentile'] * 100
    signal_emoji = "🟢" if percentile <= threshold * 100 else "🔴"
    signal_text = "[ENTRY SIGNAL]" if percentile <= threshold * 100 else "[NO SIGNAL]"
    subject = f"{signal_emoji} {signal_text} VIX {vix_data['regime']} Regime ({percentile:.1f}%)"

    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient

        html = format_html_report(vix_data, quote_datas, threshold)
        html_part = MIMEText(html, 'html', 'utf-8')
        msg.attach(html_part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def main():
    args = _parse_args()
    vix_weekly = load_vix_weekly()
    vix_close = vix_weekly.iloc[-1]
    percentile = compute_vix_percentile(vix_weekly)
    regime = determine_regime(percentile)
    vix_data = {"close": vix_close, "percentile": percentile, "regime": regime}

    signal_active = percentile <= args.threshold

    quote_datas = []
    for config in VARIANT_CONFIGS:
        qd = get_diagonal_quote(vix_close, config)
        qd["name"] = config["name"]
        qd["desc"] = config["desc"]
        quote_datas.append(qd)

    if args.json:
        output = {
            "vix": vix_data,
            "variants": quote_datas,
            "signal_active": signal_active,
            "threshold": args.threshold,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_html_report(vix_data, quote_datas, args.threshold))  # For console test

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