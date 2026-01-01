#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (Text + HTML Version)

Generates clean HTML report with 5 variant diagonals and emails it every Thursday.
No PNG attachment. Compact layout, ASCII-safe for Ubuntu SMTP.
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


def _extract_scalar(val) -> float:
    """Safely extract scalar float from pandas Series/DataFrame/scalar."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0]) if len(val) > 0 else 0.0
    elif isinstance(val, pd.DataFrame):
        return float(val.iloc[0, 0]) if val.size > 0 else 0.0
    elif isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 else 0.0
    else:
        return float(val)


def load_vix_weekly() -> pd.Series:
    """Load VIX weekly data, handling yfinance MultiIndex columns."""
    start = dt.date.today() - dt.timedelta(days=730)
    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download("^VIX", start=start, end=end, progress=False)
    
    if df.empty:
        raise RuntimeError("Failed to load VIX data")
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    series = df[col].resample("W-FRI").last().dropna()
    
    # Ensure it's a 1D Series with float values
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    return series


def load_uvxy_spot() -> float:
    """Load current UVXY spot price."""
    try:
        df = yf.download("UVXY", period="5d", progress=False)
        if df.empty:
            return 0.0
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return _extract_scalar(df[col].iloc[-1])
    except Exception as e:
        print(f"[WARN] Failed to load UVXY: {e}")
        return 0.0


def compute_vix_percentile(vix_series: pd.Series) -> float:
    """Return percentile as float 0.0 to 1.0."""
    recent = vix_series.iloc[-52:]
    if len(recent) < 2:
        return 0.5
    
    # Extract current value as scalar
    current_val = _extract_scalar(recent.iloc[-1])
    
    # Count how many values are below current
    values = recent.values.flatten()
    below_count = int((values < current_val).sum())
    
    return float(below_count) / float(len(recent) - 1)


def determine_regime(percentile: float) -> str:
    """Determine VIX regime based on percentile (must be float, not Series)."""
    pct = float(percentile)  # Ensure it's a scalar
    if pct <= 0.10:
        return "ULTRA_LOW"
    elif pct <= 0.30:
        return "LOW"
    elif pct <= 0.70:
        return "MID"
    else:
        return "HIGH"


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price with scalar inputs."""
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def get_diagonal_quote(uvxy_spot: float, vix_close: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate diagonal spread quote using Black-Scholes."""
    otm_long = float(config["otm_long"])
    otm_short = float(config["otm_short"])
    long_dte_days = 26 * 7  # ~6 months
    short_dte_days = 7      # 1 week
    r = 0.03
    
    # VIX-based volatility estimate
    sigma = float(vix_close) / 100 * 1.5
    sigma = max(0.30, min(sigma, 2.0))

    long_T = long_dte_days / 365.0
    short_T = short_dte_days / 365.0
    long_K = round(uvxy_spot + otm_long, 1)
    short_K = round(uvxy_spot + otm_short, 1)

    long_mid = bs_call_price(uvxy_spot, long_K, long_T, r, sigma)
    short_mid = bs_call_price(uvxy_spot, short_K, short_T, r, sigma)

    net_debit = long_mid - short_mid
    risk = abs(net_debit) * 100

    long_expiry = (dt.date.today() + dt.timedelta(days=long_dte_days)).strftime("%b %d")
    short_expiry = (dt.date.today() + dt.timedelta(days=short_dte_days)).strftime("%b %d")

    return {
        "name": config["name"],
        "desc": config["desc"],
        "uvxy": uvxy_spot,
        "long_K": long_K,
        "long_mid": long_mid,
        "long_expiry": long_expiry,
        "short_K": short_K,
        "short_mid": short_mid,
        "short_expiry": short_expiry,
        "net_debit": net_debit,
        "risk": risk,
    }


def format_html_report(vix_close: float, percentile_pct: float, regime: str, 
                       variants: List[Dict], threshold: float) -> str:
    """Format HTML report (ASCII-safe for SMTP)."""
    today = dt.date.today().strftime("%B %d, %Y")
    signal_active = percentile_pct <= threshold * 100
    
    # ASCII-safe signal indicators
    if signal_active:
        signal_color = "#00ff00"
        signal_text = "&gt;&gt;&gt; ENTRY SIGNAL ACTIVE &lt;&lt;&lt;"
        compare_sym = "&lt;="
    else:
        signal_color = "#ff4444"
        signal_text = "--- HOLD - No Signal ---"
        compare_sym = "&gt;"

    html = f"""
    <html>
    <body style="background:#001133;color:#ddd;font-family:monospace;padding:15px;max-width:650px;margin:auto;line-height:1.4;">
    <h1 style="color:#00ffff;margin:0;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color:#ccc;margin:5px 0 15px;">Thursday Signal Report - {today}</h3>

    <h2 style="color:#00ffff;margin-bottom:8px;">[MARKET STATE]</h2>
    <p style="margin:4px 0;">VIX Close: ${vix_close:.2f}</p>
    <p style="margin:4px 0;">52w Percentile: {percentile_pct:.1f}%</p>
    <p style="margin:4px 0;">Current Regime: <strong style="color:#ffff00;">{regime}</strong></p>

    <h2 style="color:{signal_color};margin:12px 0 8px;">{signal_text}</h2>
    <p style="margin:4px 0;">Percentile ({percentile_pct:.1f}%) {compare_sym} threshold ({threshold*100:.0f}%)</p>

    <h2 style="color:#00ffff;margin:15px 0 8px;">[5 DIAGONAL VARIANTS]</h2>
    """

    for v in variants:
        if v['net_debit'] > 0:
            debit_str = f"${v['net_debit']:.2f}"
        else:
            debit_str = f"Credit ${-v['net_debit']:.2f}"
            
        html += f"""
        <div style="background:#112255;padding:10px;margin:8px 0;border-left:4px solid #00ffff;">
        <strong style="color:#ffff00;">{v['name']}</strong> - {v['desc']}<br>
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
    <p style="margin:4px 0;">Suggested Contracts: 2-3</p>
    <p style="margin:4px 0;">Total Risk: ~$1,600-$2,400</p>

    <p style="color:#888888;margin-top:20px;font-size:11px;">[!] Research tool only - verify quotes with your broker.</p>
    </body>
    </html>
    """
    return html


def send_signal_email(recipient: str, html: str, vix_close: float, 
                      percentile_pct: float, regime: str, threshold: float) -> bool:
    """Send HTML email report."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print("[WARN] SMTP_USER or SMTP_PASS not set")
        return False

    signal_active = percentile_pct <= threshold * 100
    
    # ASCII-safe subject line (no emoji)
    if signal_active:
        subject = f"[ENTRY SIGNAL] VIX {regime} Regime ({percentile_pct:.1f}%)"
    else:
        subject = f"[WEEKLY] VIX {regime} Regime ({percentile_pct:.1f}%)"

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
        print(f"[ERROR] Email failed: {e}")
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
        vix_series = load_vix_weekly()
        vix_close = _extract_scalar(vix_series.iloc[-1])
        print(f"[OK] VIX: ${vix_close:.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to load VIX: {e}")
        sys.exit(1)

    # Compute percentile (returns float 0.0-1.0)
    percentile = compute_vix_percentile(vix_series)
    percentile_pct = float(percentile) * 100.0
    
    # Determine regime (percentile must be float)
    regime = determine_regime(percentile)
    
    print(f"[OK] Percentile: {percentile_pct:.1f}%")
    print(f"[OK] Regime: {regime}")
    
    signal_active = float(percentile) <= args.threshold
    
    print()
    if signal_active:
        print(f">>> ENTRY SIGNAL ACTIVE <<< (percentile <= {args.threshold*100:.0f}%)")
    else:
        print(f"--- HOLD --- (percentile > {args.threshold*100:.0f}%)")
    print()

    # Load UVXY spot
    print("[...] Loading UVXY spot...")
    uvxy_spot = load_uvxy_spot()
    if uvxy_spot <= 0:
        print("[ERROR] Failed to load UVXY spot")
        sys.exit(1)
    print(f"[OK] UVXY: ${uvxy_spot:.2f}")

    # Generate variant quotes
    print("[...] Generating diagonal quotes...")
    variants = []
    for cfg in VARIANT_CONFIGS:
        v = get_diagonal_quote(uvxy_spot, vix_close, cfg)
        variants.append(v)
        print(f"[OK] {cfg['name']}: Net Debit ${v['net_debit']:.2f}")

    print()

    # Generate HTML report
    html_report = format_html_report(vix_close, percentile_pct, regime, variants, args.threshold)

    if args.json:
        output = {
            "vix_close": vix_close, 
            "percentile": percentile_pct, 
            "regime": regime, 
            "signal_active": signal_active,
            "variants": variants
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        # Print text summary
        print("=" * 50)
        print(f"VIX: ${vix_close:.2f} | Percentile: {percentile_pct:.1f}% | Regime: {regime}")
        print("=" * 50)

    # Send email
    if args.email and (signal_active or args.force):
        print()
        print(f"[EMAIL] Sending to {args.email}...")
        if send_signal_email(args.email, html_report, vix_close, percentile_pct, regime, args.threshold):
            print("[OK] Email sent successfully!")
        else:
            print("[ERROR] Failed to send email")
    elif args.email and not signal_active:
        print()
        print(f"[INFO] No signal active - email skipped (use --force to override)")

    sys.exit(0 if signal_active else 1)


if __name__ == "__main__":
    main()