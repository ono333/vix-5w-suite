#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (Regime-Tuned Profiles)

Loads top-performing parameters from param_history.json grid scans.
Generates 5 authentic variants with backtest stats (CAGR, Sharpe, Win Rate).
Emoji in subject line only, clean tight HTML body.

Cron (4:30pm ET = 20:30 UTC):
    30 20 * * 4 /home/shin/vix_suite/venv/bin/python /home/shin/vix_suite/daily_signal.py --email

Manual:
    python daily_signal.py --email onoshin333@gmail.com --force
"""

import argparse
import datetime as dt
import json
import os
import sys
import smtplib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from math import log, sqrt, exp

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EMAIL = "onoshin333@gmail.com"
DEFAULT_THRESHOLD = 0.35  # 35% percentile threshold for entry signal

# Path to param_history.json (adjust if needed)
SCRIPT_DIR = Path(__file__).parent
PARAM_HISTORY_PATH = SCRIPT_DIR / "core" / "param_history.json"

# Fallback if no param_history.json found
FALLBACK_VARIANTS = [
    {"name": "Aggressive",   "otm_pts": 15.0, "long_dte_weeks": 26, "sigma_mult": 1.0},
    {"name": "Moderate",     "otm_pts": 10.0, "long_dte_weeks": 26, "sigma_mult": 1.0},
    {"name": "Conservative", "otm_pts": 5.0,  "long_dte_weeks": 26, "sigma_mult": 0.8},
    {"name": "Short DTE",    "otm_pts": 10.0, "long_dte_weeks": 13, "sigma_mult": 1.0},
    {"name": "Far OTM",      "otm_pts": 20.0, "long_dte_weeks": 26, "sigma_mult": 1.0},
]

# Regime thresholds (percentile boundaries)
REGIME_THRESHOLDS = {
    "ULTRA_LOW": 0.10,
    "LOW": 0.25,
    "MID": 0.50,
    "HIGH": 0.75,
    "EXTREME": 1.00,
}


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIX 5% Weekly Signal Generator")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--email", nargs="?", const=DEFAULT_EMAIL, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

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
    """Load VIX weekly data."""
    start = dt.date.today() - dt.timedelta(days=730)
    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download("^VIX", start=start, end=end, progress=False)
    
    if df.empty:
        raise RuntimeError("Failed to load VIX data")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    series = df[col].resample("W-FRI").last().dropna()
    
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    return series


def load_uvxy_spot() -> float:
    """Load current UVXY spot price."""
    try:
        df = yf.download("UVXY", period="5d", progress=False)
        if df.empty:
            return 0.0
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return _extract_scalar(df[col].iloc[-1])
    except Exception as e:
        print(f"[WARN] Failed to load UVXY: {e}")
        return 0.0


def compute_vix_percentile(vix_series: pd.Series, lookback: int = 52) -> float:
    """Return percentile as float 0.0 to 1.0."""
    recent = vix_series.iloc[-lookback:]
    if len(recent) < 2:
        return 0.5
    
    current_val = _extract_scalar(recent.iloc[-1])
    values = recent.values.flatten()
    below_count = int((values < current_val).sum())
    
    return float(below_count) / float(len(recent) - 1)


def determine_regime(percentile: float) -> str:
    """Determine VIX regime based on percentile."""
    pct = float(percentile)
    if pct <= REGIME_THRESHOLDS["ULTRA_LOW"]:
        return "ULTRA_LOW"
    elif pct <= REGIME_THRESHOLDS["LOW"]:
        return "LOW"
    elif pct <= REGIME_THRESHOLDS["MID"]:
        return "MID"
    elif pct <= REGIME_THRESHOLDS["HIGH"]:
        return "HIGH"
    else:
        return "EXTREME"


# =============================================================================
# PARAM HISTORY LOADING (REGIME-TUNED PROFILES)
# =============================================================================

def load_regime_profiles(regime: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Load top N profiles from param_history.json.
    
    Returns list of dicts with:
        - name: display name
        - otm_pts, long_dte_weeks, sigma_mult, entry_percentile: scan params
        - cagr, max_dd, win_rate, trades, score: backtest metrics
    """
    if not PARAM_HISTORY_PATH.exists():
        # Try alternate path
        alt_path = SCRIPT_DIR / "param_history.json"
        if not alt_path.exists():
            print(f"[WARN] param_history.json not found, using fallback variants")
            return []
        path = alt_path
    else:
        path = PARAM_HISTORY_PATH
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load param_history.json: {e}")
        return []
    
    # Extract all diagonal strategy entries
    all_entries = []
    strategies = data.get('strategies', {})
    
    for strat_name, entries in strategies.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            row = entry.get('row', {})
            if not row:
                continue
            
            # Extract performance metrics from row
            profile = {
                'strategy': strat_name,
                'otm_pts': float(row.get('otm_pts', 10.0)),
                'long_dte_weeks': int(row.get('long_dte_weeks', 26)),
                'sigma_mult': float(row.get('sigma_mult', 1.0)),
                'entry_percentile': float(row.get('entry_percentile', 0.1)),
                'cagr': float(row.get('cagr', 0)),
                'max_dd': float(row.get('max_dd', 0)),
                'win_rate': float(row.get('win_rate', 0)),
                'trades': int(row.get('trades', 0)),
                'score': float(row.get('score', 0)),
                'avg_trade_dur': float(row.get('avg_trade_dur', 0)),
                'total_return': float(row.get('total_return', 0)),
                'timestamp': entry.get('timestamp', ''),
            }
            all_entries.append(profile)
    
    if not all_entries:
        return []
    
    # Sort by score (higher is better), then by CAGR
    all_entries.sort(key=lambda x: (x['score'], x['cagr']), reverse=True)
    
    # Deduplicate by (otm_pts, long_dte_weeks, sigma_mult) to get unique configs
    seen_configs = set()
    unique_profiles = []
    
    for p in all_entries:
        config_key = (p['otm_pts'], p['long_dte_weeks'], p['sigma_mult'])
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            unique_profiles.append(p)
        
        if len(unique_profiles) >= top_n:
            break
    
    # Assign descriptive names based on characteristics
    for i, p in enumerate(unique_profiles):
        otm = p['otm_pts']
        dte = p['long_dte_weeks']
        
        if otm >= 15:
            name = f"Far OTM ({otm:.0f}pt)"
        elif otm <= 5:
            name = f"Near ATM ({otm:.0f}pt)"
        elif dte <= 13:
            name = f"Short DTE ({dte}w)"
        elif dte >= 26:
            name = f"LEAP ({dte}w)"
        else:
            name = f"Balanced ({otm:.0f}pt/{dte}w)"
        
        p['name'] = f"#{i+1} {name}"
    
    return unique_profiles


# =============================================================================
# BLACK-SCHOLES PRICING
# =============================================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def get_diagonal_quote(uvxy_spot: float, vix_close: float, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Generate diagonal spread quote from profile params."""
    otm_pts = float(profile.get('otm_pts', 10.0))
    long_dte_weeks = int(profile.get('long_dte_weeks', 26))
    sigma_mult = float(profile.get('sigma_mult', 1.0))
    
    long_dte_days = long_dte_weeks * 7
    short_dte_days = 7  # Weekly short
    r = 0.03
    
    # VIX-based volatility estimate for UVXY
    base_sigma = float(vix_close) / 100 * 1.5
    sigma = base_sigma * sigma_mult
    sigma = max(0.30, min(sigma, 2.0))
    
    long_T = long_dte_days / 365.0
    short_T = short_dte_days / 365.0
    
    long_K = round(uvxy_spot + otm_pts, 0)
    short_K = round(uvxy_spot + otm_pts - 2, 0)  # Short slightly closer to ATM
    
    # Ensure strikes are reasonable
    long_K = max(long_K, uvxy_spot * 1.02)
    short_K = max(short_K, uvxy_spot * 1.01)
    
    long_mid = bs_call_price(uvxy_spot, long_K, long_T, r, sigma)
    short_mid = bs_call_price(uvxy_spot, short_K, short_T, r, sigma)
    
    net_debit = long_mid - short_mid
    risk_per_contract = abs(net_debit) * 100
    
    long_exp = (dt.date.today() + dt.timedelta(days=long_dte_days)).strftime("%b %d")
    short_exp = (dt.date.today() + dt.timedelta(days=short_dte_days)).strftime("%b %d")
    
    return {
        'name': profile.get('name', 'Variant'),
        'uvxy_spot': uvxy_spot,
        'long_K': long_K,
        'long_mid': long_mid,
        'long_dte_days': long_dte_days,
        'long_exp': long_exp,
        'short_K': short_K,
        'short_mid': short_mid,
        'short_dte_days': short_dte_days,
        'short_exp': short_exp,
        'net_debit': net_debit,
        'risk': risk_per_contract,
        # Backtest metrics
        'cagr': profile.get('cagr', 0),
        'max_dd': profile.get('max_dd', 0),
        'win_rate': profile.get('win_rate', 0),
        'trades': profile.get('trades', 0),
        'score': profile.get('score', 0),
        # Params for reference
        'otm_pts': otm_pts,
        'long_dte_weeks': long_dte_weeks,
        'sigma_mult': sigma_mult,
    }


# =============================================================================
# HTML REPORT FORMATTING
# =============================================================================

def format_html_report(vix_close: float, percentile_pct: float, regime: str,
                       uvxy_spot: float, variants: List[Dict], threshold: float) -> str:
    """Format clean, tight HTML report with backtest stats."""
    today = dt.date.today().strftime("%Y-%m-%d")
    signal_active = percentile_pct <= threshold * 100
    
    # Signal styling
    if signal_active:
        signal_color = "#00ff00"
        signal_text = "&gt;&gt;&gt; ENTRY SIGNAL ACTIVE &lt;&lt;&lt;"
        compare_sym = "&lt;="
    else:
        signal_color = "#ff4444"
        signal_text = "--- HOLD ---"
        compare_sym = "&gt;"
    
    # Regime color
    regime_colors = {
        "ULTRA_LOW": "#00ff00",
        "LOW": "#88ff88",
        "MID": "#ffff00",
        "HIGH": "#ff8800",
        "EXTREME": "#ff0000",
    }
    regime_color = regime_colors.get(regime, "#ffffff")
    
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#0a0a1a;color:#ccc;font-family:Consolas,Monaco,monospace;font-size:13px;padding:12px;max-width:640px;margin:auto;line-height:1.35;">

<div style="border-bottom:2px solid #00ffff;padding-bottom:8px;margin-bottom:12px;">
<span style="color:#00ffff;font-size:18px;font-weight:bold;">VIX 5% WEEKLY SUITE</span><br>
<span style="color:#666;">Signal Report - {today}</span>
</div>

<div style="background:#111;padding:10px;margin-bottom:10px;border-left:3px solid #00ffff;">
<span style="color:#00ffff;">MARKET STATE</span><br>
VIX: <b>${vix_close:.2f}</b> | 
Percentile: <b>{percentile_pct:.1f}%</b> | 
Regime: <b style="color:{regime_color};">{regime}</b><br>
UVXY Spot: <b>${uvxy_spot:.2f}</b>
</div>

<div style="background:#111;padding:10px;margin-bottom:12px;border-left:3px solid {signal_color};">
<span style="color:{signal_color};font-size:15px;font-weight:bold;">{signal_text}</span><br>
<span style="color:#888;">Percentile ({percentile_pct:.1f}%) {compare_sym} threshold ({threshold*100:.0f}%)</span>
</div>

<div style="color:#00ffff;font-weight:bold;margin-bottom:8px;">TOP 5 REGIME-TUNED VARIANTS (from grid scans)</div>
"""
    
    for v in variants:
        # Format backtest stats
        cagr_pct = v['cagr'] * 100
        dd_pct = abs(v['max_dd']) * 100
        win_pct = v['win_rate'] * 100
        
        # Net debit formatting
        if v['net_debit'] >= 0:
            debit_str = f"${v['net_debit']:.2f}"
        else:
            debit_str = f"Cr ${abs(v['net_debit']):.2f}"
        
        # Performance badge color
        if cagr_pct >= 30:
            perf_color = "#00ff00"
        elif cagr_pct >= 15:
            perf_color = "#88ff88"
        else:
            perf_color = "#ffff00"
        
        html += f"""
<div style="background:#0d1a2a;padding:8px;margin-bottom:6px;border-left:3px solid #444;">
<div style="color:#ffcc00;font-weight:bold;margin-bottom:4px;">{v['name']}</div>
<div style="color:#aaa;font-size:11px;margin-bottom:6px;">OTM: {v['otm_pts']:.0f}pt | DTE: {v['long_dte_weeks']}w | Sigma: {v['sigma_mult']:.1f}x</div>

<span style="color:#00cc00;">BUY:</span> UVXY {v['long_exp']} ${v['long_K']:.0f}C @ ${v['long_mid']:.2f} ({v['long_dte_days']}d)<br>
<span style="color:#ff6666;">SELL:</span> UVXY {v['short_exp']} ${v['short_K']:.0f}C @ ${v['short_mid']:.2f} ({v['short_dte_days']}d)<br>
<b>Net Debit:</b> {debit_str} | <b>Risk:</b> ~${v['risk']:.0f}/contract<br>

<div style="margin-top:4px;padding-top:4px;border-top:1px solid #333;font-size:11px;">
<span style="color:{perf_color};">CAGR: {cagr_pct:.0f}%</span> | 
MaxDD: {dd_pct:.0f}% | 
Win: {win_pct:.0f}% | 
Trades: {v['trades']:.0f}
</div>
</div>
"""
    
    html += f"""
<div style="background:#111;padding:8px;margin-top:10px;border-left:3px solid #ffcc00;">
<span style="color:#ffcc00;">POSITION SIZING</span><br>
Suggested: 2-3 contracts | Total Risk: ~$1,500-$2,500
</div>

<div style="color:#555;font-size:10px;margin-top:12px;padding-top:8px;border-top:1px solid #333;">
Research tool only - verify quotes with broker. Backtest stats from param_history.json grid scans.
</div>

</body>
</html>"""
    
    return html


# =============================================================================
# EMAIL SENDING
# =============================================================================

def send_signal_email(recipient: str, html: str, vix_close: float,
                      percentile_pct: float, regime: str, threshold: float) -> bool:
    """Send HTML email with emoji in subject line."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        print("[WARN] SMTP_USER or SMTP_PASS not set")
        return False
    
    signal_active = percentile_pct <= threshold * 100
    
    # Subject line WITH emoji for inbox scanning
    if signal_active:
        subject = f"\U0001F7E2 [ENTRY] VIX {regime} ({percentile_pct:.1f}%) - Diagonals Ready"
    else:
        subject = f"\U0001F534 [HOLD] VIX {regime} ({percentile_pct:.1f}%)"
    
    try:
        msg = MIMEMultipart()
        msg['Subject'] = Header(subject, 'utf-8')
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = _parse_args()
    
    print("=" * 55)
    print("VIX 5% Weekly Suite - Thursday Signal (Regime-Tuned)")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print()
    
    # Load VIX
    print("[...] Loading VIX data...")
    try:
        vix_series = load_vix_weekly()
        vix_close = _extract_scalar(vix_series.iloc[-1])
        print(f"[OK] VIX: ${vix_close:.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to load VIX: {e}")
        sys.exit(1)
    
    # Compute percentile and regime
    percentile = compute_vix_percentile(vix_series)
    percentile_pct = float(percentile) * 100.0
    regime = determine_regime(percentile)
    
    print(f"[OK] Percentile: {percentile_pct:.1f}%")
    print(f"[OK] Regime: {regime}")
    
    signal_active = float(percentile) <= args.threshold
    
    print()
    if signal_active:
        print(f">>> ENTRY SIGNAL ACTIVE <<< (pct <= {args.threshold*100:.0f}%)")
    else:
        print(f"--- HOLD --- (pct > {args.threshold*100:.0f}%)")
    print()
    
    # Load UVXY
    print("[...] Loading UVXY spot...")
    uvxy_spot = load_uvxy_spot()
    if uvxy_spot <= 0:
        print("[ERROR] Failed to load UVXY spot")
        sys.exit(1)
    print(f"[OK] UVXY: ${uvxy_spot:.2f}")
    
    # Load regime-tuned profiles from param_history.json
    print("[...] Loading regime profiles from param_history.json...")
    profiles = load_regime_profiles(regime, top_n=5)
    
    if profiles:
        print(f"[OK] Loaded {len(profiles)} profiles from grid scans")
    else:
        print("[WARN] No profiles found, using fallback variants")
        profiles = FALLBACK_VARIANTS
    
    # Generate quotes
    print("[...] Generating diagonal quotes...")
    variants = []
    for p in profiles:
        v = get_diagonal_quote(uvxy_spot, vix_close, p)
        variants.append(v)
        print(f"[OK] {v['name']}: Net ${v['net_debit']:.2f} | CAGR {v['cagr']*100:.0f}%")
    
    print()
    
    # Generate HTML report
    html_report = format_html_report(
        vix_close, percentile_pct, regime, uvxy_spot, variants, args.threshold
    )
    
    # Output
    if args.json:
        output = {
            "vix_close": vix_close,
            "percentile": percentile_pct,
            "regime": regime,
            "uvxy_spot": uvxy_spot,
            "signal_active": signal_active,
            "variants": variants,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print("=" * 55)
        print(f"VIX: ${vix_close:.2f} | Pct: {percentile_pct:.1f}% | Regime: {regime}")
        print(f"UVXY: ${uvxy_spot:.2f} | Signal: {'ACTIVE' if signal_active else 'HOLD'}")
        print("=" * 55)
        for v in variants:
            print(f"  {v['name']}: ${v['long_K']:.0f}C/${v['short_K']:.0f}C Net ${v['net_debit']:.2f}")
    
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
        print(f"[INFO] No signal - email skipped (use --force to override)")
    
    sys.exit(0 if signal_active else 1)


if __name__ == "__main__":
    main()
