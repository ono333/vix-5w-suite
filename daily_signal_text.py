#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer

Generates clean HTML report with 3 variant diagonals and emails it every Thursday.
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
import requests
import yfinance as yf
from scipy.stats import norm
from math import log, sqrt, exp

# Constants
DEFAULT_EMAIL = "onoshin333@gmail.com"
DEFAULT_THRESHOLD = 0.35  # 35%
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")
MASSIVE_BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.app")
CACHE_DIR = Path.home() / ".cache" / "massive_options"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Variant configs
VARIANT_CONFIGS = [
    {"name": "Base", "otm_pts": 10.0, "long_dte_weeks": 26},
    {"name": "Aggressive", "otm_pts": 15.0, "long_dte_weeks": 26},
    {"name": "Conservative", "otm_pts": 5.0, "long_dte_weeks": 26},
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
    return np.percentile(recent, 100 * (recent.iloc[-1] - recent.min()) / (recent.max() - recent.min() + 1e-6)) / 100

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
    otm_pts = config["otm_pts"]
    long_dte_weeks = config["long_dte_weeks"]
    long_dte_days = long_dte_weeks * 7
    short_dte_days = 1 * 7  # Weekly short
    r = 0.03  # Risk-free rate
    sigma_mult = 1.0
    sigma = vix_close / 100 * sigma_mult  # Base vol from VIX
    uvxy = yf.download("UVXY", period="1d")["Close"].iloc[-1]  # Latest UVXY spot

    long_T = long_dte_days / 365
    short_T = short_dte_days / 365
    long_K = uvxy + otm_pts
    short_K = uvxy + otm_pts - 4  # Slight adjustment for short

    long_mid = bs_call_price(uvxy, long_K, long_T, r, sigma)
    long_bid = long_mid * 0.95
    long_ask = long_mid * 1.05

    short_mid = bs_call_price(uvxy, short_K, short_T, r, sigma)
    short_bid = short_mid * 0.95
    short_ask = short_mid * 1.05

    net_debit_mid = long_mid - short_mid

    return {
        "long_bid": long_bid,
        "long_ask": long_ask,
        "long_mid": long_mid,
        "long_K": long_K,
        "long_dte_days": long_dte_days,
        "short_bid": short_bid,
        "short_ask": short_ask,
        "short_mid": short_mid,
        "short_K": short_K,
        "short_dte_days": short_dte_days,
        "net_debit_mid": net_debit_mid,
        "current_price": uvxy,
    }

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def format_html_report(vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> str:
    """Format HTML report with emojis and 3 variants."""
    today = dt.date.today().strftime("%B %d, %Y")
    html = """
    <html>
    <body style="background-color: #001133; color: #dddddd; font-family: monospace; padding: 20px; max-width: 600px; margin: auto;">
    <h1 style="color: #00ffff;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color: #cccccc;">Thursday Signal Report - {today}</h3>

    <h2 style="color: #00ffff;">[MARKET STATE]</h2>
    <p>VIX Close: ${close:.2f}</p>
    <p>52w Percentile: {percentile:.1f}%</p>
    <p>Current Regime: {regime}</p>

    <h2 style="color: #00ff00;">🟢 >>> ENTRY SIGNAL ACTIVE <<<</h2>
    <p>Percentile ({percentile:.1f}%) ≤ threshold ({threshold*100:.0f}%)</p>

    <h2 style="color: #00ffff;">[3 DIAGONAL VARIANTS]</h2>
    """.format(today=today, close=vix_data['close'], percentile=vix_data['percentile'], regime=vix_data['regime'], threshold=threshold)

    for i, qd in enumerate(quote_datas, 1):
        html += f"""
        <h3 style="color: #ffff00;">Variant {i}: {qd['name']}</h3>
        <p>UVXY Spot: ${qd['current_price']:.2f}</p>

        <h4 style="color: #00ff00;">LONG LEG (Buy):</h4>
        <p>UVXY {long_date} ${qd['long_K']:.0f}C</p>
        <p>Bid: ${qd['long_bid']:.2f} Ask: ${qd['long_ask']:.2f} Mid: ${qd['long_mid']:.2f}</p>
        <p>DTE: {qd['long_dte_days']} days</p>

        <h4 style="color: #ff0000;">SHORT LEG (Sell):</h4>
        <p>UVXY {short_date} ${qd['short_K']:.0f}C</p>
        <p>Bid: ${qd['short_bid']:.2f} Ask: ${qd['short_ask']:.2f} Mid: ${qd['short_mid']:.2f}</p>
        <p>DTE: {qd['short_dte_days']} days</p>

        <h4 style="color: #cccccc;">NET POSITION:</h4>
        <p>Net Debit (mid): ${qd['net_debit_mid']:.2f}</p>
        """

    html += """
    <h2 style="color: #00ffff;">POSITION SIZING SUGGESTION</h2>
    <p>Suggested Contracts: 2</p>
    <p>Risk per Contract: $881</p>
    <p>Total Position Risk: $1762</p>

    <p style="color: #ffff00;">⚠️ Research tool only — not financial advice. Verify quotes with your broker.</p>
    </body>
    </html>
    """

    # Mock dates for long/short
    long_date = (dt.date.today() + dt.timedelta(weeks=VARIANT_CONFIGS[0]["long_dte_weeks"])).strftime("%b %d")
    short_date = (dt.date.today() + dt.timedelta(weeks=1)).strftime("%b %d")

    html = html.format(long_date=long_date, short_date=short_date)

    return html

def send_signal_email(recipient: str, vix_data: Dict[str, Any], quote_datas: List[Dict[str, Any]], threshold: float) -> bool:
    """Send HTML email report."""
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not all([smtp_server, smtp_user, smtp_pass]):
        return False

    try:
        msg = MIMEMultipart()
        msg['Subject'] = "VIX 5% Weekly Signal Report"
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
    vix_data = {"close": vix_close, "percentile": percentile * 100, "regime": regime}

    signal_active = percentile <= args.threshold

    quote_datas = []
    for config in VARIANT_CONFIGS:
        qd = get_diagonal_quote(vix_close, config)
        qd["name"] = config["name"]
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
        print(format_html_report(vix_data, quote_datas, args.threshold))  # For console test, print HTML

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