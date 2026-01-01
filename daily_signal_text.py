#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (Text + Emoji Version)

Generates clean HTML report with 5 variant diagonals and emails it every Thursday.
No PNG attachment. Compact layout, emojis, subject with emoji.
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

# 5 Variant configs
VARIANT_CONFIGS = [
    {"name": "Aggressive",   "otm_long": 15.0, "otm_short": 13.0, "desc": "Higher theta bleed"},
    {"name": "Moderate",     "otm_long": 10.0, "otm_short": 8.0,  "desc": "Balanced"},
    {"name": "Conservative", "otm_long": 5.0,  "otm_short": 3.0,  "desc": "Lower risk"},
    {"name": "Far OTM Long", "otm_long": 20.0, "otm_short": 18.0, "desc": "Cheaper entry, bigger upside"},
    {"name": "Closer Short", "otm_long": 10.0, "otm_short": 12.0, "desc": "Potential credit spread"},
]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--email", nargs="?", const=DEFAULT_EMAIL, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def load_vix_weekly() -> pd.Series:
    start = dt.date.today() - dt.timedelta(days=730)
    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    return df.resample("W-FRI").last().dropna()

def compute_vix_percentile(vix_series: pd.Series) -> float:
    # Return percentile as float 0.0 to 1.0
    recent = vix_series.iloc[-52:]
    rank = (recent <= recent.iloc[-1]).sum() - 1
    return rank / (len(recent) - 1)

def determine_regime(percentile: float) -> str:
    if percentile <= 0.10:
        return "ULTRA_LOW"
    elif percentile <= 0.30:
        return "LOW"
    elif percentile <= 0.70:
        return "MID"
    else:
        return "HIGH"

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def get_diagonal_quote(vix_close: float, config: Dict[str, Any]) -> Dict[str, Any]:
    otm_long = config["otm_long"]
    otm_short = config["otm_short"]
    long_dte_days = 26 * 7
    short_dte_days = 7
    r = 0.03
    sigma = vix_close / 100 * 1.0
    uvxy = yf.download("UVXY", period="1d")["Close"].iloc[-1]

    long_T = long_dte_days / 365
    short_T = short_dte_days / 365
    long_K = round(uvxy + otm_long, 1)
    short_K = round(uvxy + otm_short, 1)

    long_mid = bs_call_price(uvxy, long_K, long_T, r, sigma)
    short_mid = bs_call_price(uvxy, short_K, short_T, r, sigma)

    net_debit = long_mid - short_mid
    risk = abs(net_debit) * 100

    long_expiry = (dt.date.today() + dt.timedelta(days=long_dte_days)).strftime("%b %d")
    short_expiry = (dt.date.today() + dt.timedelta(days=short_dte_days)).strftime("%b %d")

    return {
        "name": config["name"],
        "desc": config["desc"],
        "uvxy": uvxy,
        "long_K": long_K,
        "long_mid": long_mid,
        "long_expiry": long_expiry,
        "short_K": short_K,
        "short_mid": short_mid,
        "short_expiry": short_expiry,
        "net_debit": net_debit,
        "risk": risk,
    }

def format_html_report(vix_close: float, percentile_pct: float, regime: str, variants: List[Dict], threshold: float) -> str:
    today = dt.date.today().strftime("%B %d, %Y")
    signal_active = percentile_pct <= threshold * 100
    signal_emoji = "🟢" if signal_active else "🔴"
    signal_text = ">>> ENTRY SIGNAL ACTIVE <<<" if signal_active else "No Signal"

    html = f"""
    <html>
    <body style="background:#001133;color:#ddd;font-family:monospace;padding:15px;max-width:650px;margin:auto;line-height:1.4;">
    <h1 style="color:#00ffff;margin:0;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color:#ccc;margin:5px 0 15px;">Thursday Signal Report - {today}</h3>

    <h2 style="color:#00ffff;margin-bottom:8px;">[MARKET STATE]</h2>
    <p style="margin:4px 0;">VIX Close: ${vix_close:.2f}</p>
    <p style="margin:4px 0;">52w Percentile: {percentile_pct:.1f}%</p>
    <p style="margin:4px 0;">Current Regime: {regime}</p>

    <h2 style="color:#00ff00;margin:12px 0 8px;">{signal_emoji} {signal_text}</h2>
    <p style="margin:4px 0;">Percentile ({percentile_pct:.1f}%) {'≤' if signal_active else '>'} threshold ({threshold*100:.0f}%)</p>

    <h2 style="color:#00ffff;margin:15px 0 8px;">[5 DIAGONAL VARIANTS]</h2>
    """

    for v in variants:
        debit_str = f"${v['net_debit']:.2f}" if v['net_debit'] > 0 else f"Credit ${-v['net_debit']:.2f}"
        html += f"""
        <div style="background:#112255;padding:10px;margin:8px 0;border-left:4px solid #00ffff;">
        <strong style="color:#ffff00;">{v['name']}</strong> — {v['desc']}<br>
        UVXY Spot: ${v['uvxy']:.2f}<br><br>

        <strong style="color:#00ff00;">LONG (Buy):</strong> UVXY {v['long_expiry']} ${v['long_K']:.0f}C
        Mid: ${v['long_mid']:.2f} (DTE: 182 days)<br>

        <strong style="color:#ff4444;">SHORT (Sell):</strong> UVXY {v['short_expiry']} ${v['short_K']:.0f}C
        Mid: ${v['short_mid']:.2f} (DTE: 7 days)<br><br>

        <strong>Net Debit (mid):</strong> {debit_str} | 
        <strong>Risk/Contract:</strong> ~${v['risk']:.0f}
        </div>
        """

    html += """
    <h2 style="color:#00ffff;margin:15px 0 8px;">POSITION SIZING</h2>
    <p style="margin:4px 0;">Suggested Contracts: 2–3</p>
    <p style="margin:4px 0;">Total Risk: ~$1,600–$2,400</p>

    <p style="color:#ffff00;margin-top:20px;">⚠️ Research tool only — verify quotes with your broker.</p>
    </body>
    </html>
    """
    return html

def send_signal_email(recipient: str, html: str, vix_close: float, percentile_pct: float, regime: str, threshold: float) -> bool:
    smtp_server = os.environ["SMTP_SERVER"]
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]

    signal_active = percentile_pct <= threshold * 100
    subject_emoji = "🟢" if signal_active else "🔴"
    subject = f"{subject_emoji} [ENTRY SIGNAL] VIX {regime} Regime ({percentile_pct:.1f}%)"

    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient
        msg.attach(MIMEText(html, 'html', 'utf-8'))

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
    vix_series = load_vix_weekly()
    vix_close = float(vix_series.iloc[-1])
    percentile = compute_vix_percentile(vix_series)  # 0.0 to 1.0
    percentile_pct = percentile * 100
    regime = determine_regime(percentile)
    signal_active = percentile <= args.threshold

    variants = [get_diagonal_quote(vix_close, cfg) for cfg in VARIANT_CONFIGS]

    html_report = format_html_report(vix_close, percentile_pct, regime, variants, args.threshold)

    if args.json:
        print(json.dumps({"vix_close": vix_close, "percentile": percentile_pct, "regime": regime, "variants": variants}, indent=2))
    else:
        print(html_report)  # preview in console

    if args.email and (signal_active or args.force):
        print(f"\n[EMAIL] Sending to {args.email}...")
        if send_signal_email(args.email, html_report, vix_close, percentile_pct, regime, args.threshold):
            print("[OK] Email sent successfully!")
        else:
            print("[ERROR] Failed to send email")

if __name__ == "__main__":
    main()