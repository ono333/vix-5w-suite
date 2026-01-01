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
    python daily_signal.py --png signal.png  # Save PNG locally

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

# 5 Variant configs (exact match to Streamlit app)
VARIANT_CONFIGS = [
    {"name": "Existing Conservative — Baseline", "otm_pts": 10.0, "long_dte_weeks": 26, "desc": "DTE 20-26 weeks, entry 0.1-0...."},
    {"name": "DTE 1 Week Aggressive — Ultra-short theta burn", "otm_pts": 10.0, "long_dte_weeks": 1, "desc": "max decay in co..."},
    {"name": "DTE 3 Weeks Aggressive — Short-term sweet spot", "otm_pts": 10.0, "long_dte_weeks": 3, "desc": "high harvest wit..."},
    {"name": "Exit 1.5x Profit — Tighter profit target", "otm_pts": 10.0, "long_dte_weeks": 26, "desc": "faster capital t..."},
    {"name": "Static No-Regime — Benchmark", "otm_pts": 10.0, "long_dte_weeks": 26, "desc": "fixed conservative params, n..."},
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
    """Format HTML report with white background and 5 variants."""
    today = dt.date.today().strftime("%B %d, %Y")
    percentile = vix_data['percentile'] * 100
    signal_active = percentile <= threshold * 100
    signal_emoji = "🟢" if signal_active else "🔴"
    signal_text = ">>> ENTRY SIGNAL ACTIVE <<<" if signal_active else "No Signal"

    html = f"""
    <html>
    <body style="background-color: #ffffff; color: #333333; font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: auto; line-height: 1.5; font-size: 16px;">
    <h1 style="color: #00aadd; text-align: center;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color: #666666; text-align: center; margin: 10px 0 30px;">Thursday Signal Report - {today}</h3>

    <div style="border: 1px solid #cccccc; padding: 15px; margin-bottom: 20px;">
    <strong style="font-size: 18px; color: #00aadd;">[MARKET STATE]</strong><br>
    VIX Close: ${vix_data['close']:.2f}<br>
    52w Percentile: {percentile:.1f}%<br>
    Current Regime: {vix_data['regime']}
    </div>

    <div style="border: 1px solid #cccccc; padding: 15px; margin-bottom: 30px; background-color: {'#f8fff8' if signal_active else '#fff8f8'};">
    <strong style="font-size: 20px; color: {'#00aa00' if signal_active else '#aa0000'};">{signal_emoji} {signal_text}</strong><br>
    Percentile ({percentile:.1f}%) {'≤' if signal_active else '>'} threshold ({threshold*100:.0f}%)
    </div>

    <h2 style="color: #00aadd; font-size: 22px; margin: 30px 0 15px;">[5 DIAGONAL VARIANTS]</h2>
    """

    for qd in quote_datas:
        debit_str = f"${qd['net_debit_mid']:.2f}" if qd['net_debit_mid'] > 0 else f"Credit +${-qd['net_debit_mid']:.2f}"
        html += f"""
        <div style="border: 1px solid #cccccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9;">
        <strong style="font-size: 18px; color: #00aadd;">{qd['name']}</strong><br>
        <p>{qd['desc']}</p>
        UVXY Spot: ${qd['current_price']:.2f}<br><br>

        <strong style="color: #008800;">LONG LEG (Buy):</strong><br>
        UVXY {long_date} ${qd['long_K']:.0f}C<br>
        Bid: ${qd['long_bid']:.2f} Ask: ${qd['long_ask']:.2f} Mid: ${qd['long_mid']:.2f}<br>
        DTE: {qd['long_dte_days']} days<br><br>

        <strong style="color: #880000;">SHORT LEG (Sell):</strong><br>
        UVXY {short_date} ${qd['short_K']:.0f}C<br>
        Bid: ${qd['short_bid']:.2f} Ask: ${qd['short_ask']:.2f} Mid: ${qd['short_mid']:.2f}<br>
        DTE: {qd['short_dte_days']} days<br><br>

        <strong>NET POSITION:</strong><br>
        Net Debit (mid): {debit_str}<br>
        Net Debit (cons): ${qd['net_debit_mid'] + 0.63:.2f}<br><br>
        </div>
        """

    html += """
    <div style="border: 1px solid #cccccc; padding: 15px; background-color: #ffffee; margin-top: 30px;">
    <strong style="font-size: 18px; color: #aa8800;">POSITION SIZING</strong><br>
    Suggested Contracts: 2<br>
    Risk per Contract: $881<br>
    Total Position Risk: $1762
    </div>

    <p style="color: #aa0000; margin-top: 30px; font-size: 14px; text-align: center;">
    ⚠️ This is a research tool, not financial advice.<br>
    Always verify quotes with your broker before trading.
    </p>
    </body>
    </html>
    """

    # Mock dates for long/short
    long_date = (dt.date.today() + dt.timedelta(weeks=26)).strftime("%b %d")
    short_date = (dt.date.today() + dt.timedelta(weeks=1)).strftime("%b %d")
    
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
        msg['Subject'] = "🟢 [ENTRY SIGNAL] VIX LOW Regime (13.7%)"
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