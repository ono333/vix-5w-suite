#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer (Full Integration)

Loads param_history.json regime-tuned profiles, generates 5 variants matching
Streamlit exactly, shows target/stop prices, backtest stats, one-screen layout.

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
from typing import Dict, Any, List
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
DEFAULT_THRESHOLD = 0.35

SCRIPT_DIR = Path(__file__).parent
PARAM_HISTORY_PATHS = [
    SCRIPT_DIR / "core" / "param_history.json",
    SCRIPT_DIR / "param_history.json",
]

# 5 Streamlit-matched variants (fallback if no JSON)
# Matches: baseline 26w, 1w/3w aggressive, tighter 1.5x profit, static benchmark
STREAMLIT_VARIANTS = [
    {
        "name": "Baseline (26w)",
        "desc": "Standard 6-month diagonal",
        "otm_pts": 10.0,
        "long_dte_weeks": 26,
        "sigma_mult": 1.0,
        "target_mult": 1.20,
        "exit_mult": 0.50,
    },
    {
        "name": "Aggressive (3w)",
        "desc": "Short DTE, faster theta",
        "otm_pts": 5.0,
        "long_dte_weeks": 3,
        "sigma_mult": 0.8,
        "target_mult": 1.30,
        "exit_mult": 0.40,
    },
    {
        "name": "Aggressive (1w)",
        "desc": "Ultra-short DTE",
        "otm_pts": 3.0,
        "long_dte_weeks": 1,
        "sigma_mult": 0.8,
        "target_mult": 1.50,
        "exit_mult": 0.30,
    },
    {
        "name": "Tighter (1.5x)",
        "desc": "Quick profit target",
        "otm_pts": 10.0,
        "long_dte_weeks": 15,
        "sigma_mult": 1.0,
        "target_mult": 1.50,
        "exit_mult": 0.60,
    },
    {
        "name": "Static Benchmark",
        "desc": "Conservative baseline",
        "otm_pts": 15.0,
        "long_dte_weeks": 26,
        "sigma_mult": 1.0,
        "target_mult": 1.20,
        "exit_mult": 0.50,
    },
]

REGIME_THRESHOLDS = {
    "ULTRA_LOW": 0.10,
    "LOW": 0.25,
    "MID": 0.50,
    "HIGH": 0.75,
}


# =============================================================================
# HELPERS
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--email", nargs="?", const=DEFAULT_EMAIL, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _scalar(val) -> float:
    """Extract scalar from pandas objects."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return float(val.iloc[0] if isinstance(val, pd.Series) else val.iloc[0, 0])
    elif isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_vix_weekly() -> pd.Series:
    """Load 2-year VIX weekly data."""
    df = yf.download("^VIX", period="2y", progress=False)
    if df.empty:
        raise RuntimeError("Failed to load VIX")
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
    except:
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
    if pct <= REGIME_THRESHOLDS["ULTRA_LOW"]:
        return "ULTRA_LOW"
    elif pct <= REGIME_THRESHOLDS["LOW"]:
        return "LOW"
    elif pct <= REGIME_THRESHOLDS["MID"]:
        return "MID"
    elif pct <= REGIME_THRESHOLDS["HIGH"]:
        return "HIGH"
    return "EXTREME"


# =============================================================================
# PARAM HISTORY LOADING
# =============================================================================

def load_profiles_from_json(top_n: int = 5) -> List[Dict]:
    """Load top profiles from param_history.json, sorted by Sharpe/CAGR."""
    path = None
    for p in PARAM_HISTORY_PATHS:
        if p.exists():
            path = p
            break
    
    if not path:
        return []
    
    try:
        with open(path) as f:
            data = json.load(f)
    except:
        return []
    
    entries = []
    for strat_entries in data.get("strategies", {}).values():
        if not isinstance(strat_entries, list):
            continue
        for e in strat_entries:
            row = e.get("row", {})
            if not row:
                continue
            
            # Compute Sharpe proxy: CAGR / abs(MaxDD) if MaxDD != 0
            cagr = float(row.get("cagr", 0))
            max_dd = abs(float(row.get("max_dd", 0.01)))
            sharpe_proxy = cagr / max_dd if max_dd > 0 else 0
            
            entries.append({
                "otm_pts": float(row.get("otm_pts", 10)),
                "long_dte_weeks": int(row.get("long_dte_weeks", 26)),
                "sigma_mult": float(row.get("sigma_mult", 1.0)),
                "entry_percentile": float(row.get("entry_percentile", 0.1)),
                "target_mult": 1.20,  # Default if not in JSON
                "exit_mult": 0.50,
                "cagr": cagr,
                "max_dd": float(row.get("max_dd", 0)),
                "win_rate": float(row.get("win_rate", 0)),
                "trades": int(row.get("trades", 0)),
                "score": float(row.get("score", 0)),
                "sharpe_proxy": sharpe_proxy,
            })
    
    if not entries:
        return []
    
    # Sort by Sharpe proxy, then CAGR
    entries.sort(key=lambda x: (x["sharpe_proxy"], x["cagr"]), reverse=True)
    
    # Dedupe by config
    seen = set()
    unique = []
    for e in entries:
        key = (e["otm_pts"], e["long_dte_weeks"], e["sigma_mult"])
        if key not in seen:
            seen.add(key)
            # Generate name
            otm, dte = e["otm_pts"], e["long_dte_weeks"]
            if dte <= 3:
                name = f"Aggressive ({dte}w)"
            elif dte <= 13:
                name = f"Short DTE ({dte}w)"
            elif otm >= 15:
                name = f"Far OTM ({otm:.0f}pt)"
            elif otm <= 5:
                name = f"Near ATM ({otm:.0f}pt)"
            else:
                name = f"Balanced ({dte}w)"
            e["name"] = name
            e["desc"] = f"OTM:{otm:.0f} DTE:{dte}w Sig:{e['sigma_mult']:.1f}"
            unique.append(e)
        if len(unique) >= top_n:
            break
    
    return unique


def get_variants() -> List[Dict]:
    """Get 5 variants: prefer JSON profiles, fallback to Streamlit defaults."""
    profiles = load_profiles_from_json(5)
    if len(profiles) >= 3:
        # Pad with Streamlit variants if needed
        while len(profiles) < 5:
            idx = len(profiles) - 3
            if idx < len(STREAMLIT_VARIANTS):
                profiles.append(STREAMLIT_VARIANTS[idx])
        return profiles[:5]
    return STREAMLIT_VARIANTS[:5]


# =============================================================================
# BLACK-SCHOLES & QUOTE GENERATION
# =============================================================================

def bs_call(S: float, K: float, T: float, r: float, sig: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0)
    d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def generate_quote(uvxy: float, vix: float, v: Dict) -> Dict:
    """Generate diagonal spread quote with target/stop prices."""
    otm = float(v.get("otm_pts", 10))
    dte_w = int(v.get("long_dte_weeks", 26))
    sig_mult = float(v.get("sigma_mult", 1.0))
    target_mult = float(v.get("target_mult", 1.20))
    exit_mult = float(v.get("exit_mult", 0.50))
    
    r = 0.03
    sig = vix / 100 * 1.5 * sig_mult
    sig = max(0.30, min(sig, 2.0))
    
    long_dte = dte_w * 7
    short_dte = 7
    
    long_K = round(uvxy + otm, 0)
    short_K = round(uvxy + otm - 2, 0)
    long_K = max(long_K, round(uvxy * 1.02, 0))
    short_K = max(short_K, round(uvxy * 1.01, 0))
    
    long_mid = bs_call(uvxy, long_K, long_dte / 365, r, sig)
    short_mid = bs_call(uvxy, short_K, short_dte / 365, r, sig)
    
    net_debit = long_mid - short_mid
    risk = abs(net_debit) * 100
    
    # Target & Stop prices (based on long leg value)
    target_val = long_mid * target_mult
    stop_val = long_mid * exit_mult
    
    long_exp = (dt.date.today() + dt.timedelta(days=long_dte)).strftime("%m/%d")
    short_exp = (dt.date.today() + dt.timedelta(days=short_dte)).strftime("%m/%d")
    
    return {
        "name": v.get("name", "Variant"),
        "desc": v.get("desc", ""),
        "uvxy": uvxy,
        "long_K": long_K,
        "long_mid": long_mid,
        "long_exp": long_exp,
        "long_dte": long_dte,
        "short_K": short_K,
        "short_mid": short_mid,
        "short_exp": short_exp,
        "short_dte": short_dte,
        "net_debit": net_debit,
        "risk": risk,
        "target_mult": target_mult,
        "exit_mult": exit_mult,
        "target_val": target_val,
        "stop_val": stop_val,
        "cagr": v.get("cagr", 0),
        "max_dd": v.get("max_dd", 0),
        "win_rate": v.get("win_rate", 0),
        "trades": v.get("trades", 0),
    }


# =============================================================================
# HTML REPORT (ONE-SCREEN COMPACT)
# =============================================================================

def build_html(vix: float, pct: float, regime: str, uvxy: float, 
               quotes: List[Dict], threshold: float) -> str:
    """Build compact one-screen HTML email."""
    today = dt.date.today().strftime("%Y-%m-%d")
    signal = pct <= threshold * 100
    
    sig_color = "#0f0" if signal else "#f44"
    sig_text = "ENTRY SIGNAL" if signal else "HOLD"
    cmp = "&lt;=" if signal else "&gt;"
    
    reg_colors = {"ULTRA_LOW": "#0f0", "LOW": "#8f8", "MID": "#ff0", "HIGH": "#f80", "EXTREME": "#f00"}
    reg_color = reg_colors.get(regime, "#fff")
    
    # Header
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#080818;color:#bbb;font-family:Consolas,monospace;font-size:12px;padding:10px;max-width:620px;margin:auto;line-height:1.3;">

<div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #00bfff;padding-bottom:6px;margin-bottom:8px;">
<span style="color:#00bfff;font-size:15px;font-weight:bold;">VIX 5% WEEKLY</span>
<span style="color:#666;font-size:11px;">{today}</span>
</div>

<div style="display:flex;gap:15px;margin-bottom:8px;">
<div style="flex:1;background:#111;padding:6px;border-left:2px solid #00bfff;">
<div style="color:#888;font-size:10px;">MARKET</div>
VIX <b>${vix:.2f}</b> | Pct <b>{pct:.1f}%</b><br>
Regime <b style="color:{reg_color};">{regime}</b> | UVXY <b>${uvxy:.2f}</b>
</div>
<div style="flex:1;background:#111;padding:6px;border-left:2px solid {sig_color};">
<div style="color:{sig_color};font-weight:bold;">{sig_text}</div>
<span style="color:#666;font-size:10px;">{pct:.1f}% {cmp} {threshold*100:.0f}%</span>
</div>
</div>

<div style="color:#00bfff;font-size:11px;margin:8px 0 4px;">TOP 5 VARIANTS (Regime-Tuned)</div>
<table style="width:100%;border-collapse:collapse;font-size:11px;">
<tr style="background:#1a1a2e;color:#888;">
<th style="text-align:left;padding:4px;">Variant</th>
<th>Long</th>
<th>Short</th>
<th>Net</th>
<th>Target</th>
<th>Stop</th>
<th style="color:#0f0;">CAGR</th>
<th>DD</th>
<th>Win</th>
</tr>
"""
    
    for i, q in enumerate(quotes):
        bg = "#0d1520" if i % 2 == 0 else "#111"
        cagr_c = "#0f0" if q["cagr"] >= 0.30 else "#8f8" if q["cagr"] >= 0.15 else "#ff0"
        
        html += f"""<tr style="background:{bg};">
<td style="padding:4px;color:#fc0;"><b>{q['name'][:15]}</b><br><span style="color:#666;font-size:9px;">{q['desc'][:20]}</span></td>
<td style="text-align:center;color:#0c0;">${q['long_K']:.0f}C<br><span style="color:#888;">${q['long_mid']:.2f}</span></td>
<td style="text-align:center;color:#f66;">${q['short_K']:.0f}C<br><span style="color:#888;">${q['short_mid']:.2f}</span></td>
<td style="text-align:center;"><b>${q['net_debit']:.2f}</b><br><span style="color:#666;font-size:9px;">~${q['risk']:.0f}</span></td>
<td style="text-align:center;color:#0f0;">${q['target_val']:.2f}<br><span style="color:#666;font-size:9px;">{q['target_mult']:.1f}x</span></td>
<td style="text-align:center;color:#f44;">${q['stop_val']:.2f}<br><span style="color:#666;font-size:9px;">{q['exit_mult']:.1f}x</span></td>
<td style="text-align:center;color:{cagr_c};">{q['cagr']*100:.0f}%</td>
<td style="text-align:center;">{abs(q['max_dd'])*100:.0f}%</td>
<td style="text-align:center;">{q['win_rate']*100:.0f}%</td>
</tr>
"""
    
    html += f"""</table>

<div style="display:flex;gap:10px;margin-top:8px;">
<div style="flex:1;background:#111;padding:5px;font-size:10px;border-left:2px solid #fc0;">
<b style="color:#fc0;">SIZING</b><br>
2-3 contracts | Risk ~$1.5-2.5k
</div>
<div style="flex:1;background:#111;padding:5px;font-size:10px;border-left:2px solid #888;">
<b style="color:#888;">EXPIRY</b><br>
Long: {quotes[0]['long_exp']} ({quotes[0]['long_dte']}d) | Short: {quotes[0]['short_exp']} (7d)
</div>
</div>

<div style="color:#555;font-size:9px;margin-top:8px;text-align:center;">
Research only - verify with broker | Stats from param_history.json grid scans
</div>
</body></html>"""
    
    return html


# =============================================================================
# EMAIL
# =============================================================================

def send_email(to: str, html: str, vix: float, pct: float, regime: str, threshold: float) -> bool:
    """Send email with emoji subject."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        print("[WARN] SMTP credentials not set")
        return False
    
    signal = pct <= threshold * 100
    
    # Emoji subject for inbox scanning
    if signal:
        subj = f"\U0001F7E2 [ENTRY] VIX {regime} {pct:.0f}% - Diagonals Ready"
    else:
        subj = f"\U0001F534 [HOLD] VIX {regime} {pct:.0f}%"
    
    try:
        msg = MIMEMultipart()
        msg["Subject"] = Header(subj, "utf-8")
        msg["From"] = smtp_user
        msg["To"] = to
        msg.attach(MIMEText(html, "html", "utf-8"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = _parse_args()
    
    print("=" * 50)
    print("VIX 5% Weekly - Thursday Signal (Full Integration)")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Load data
    print("\n[...] Loading market data...")
    try:
        vix_series = load_vix_weekly()
        vix = _scalar(vix_series.iloc[-1])
        print(f"[OK] VIX: ${vix:.2f}")
    except Exception as e:
        print(f"[ERROR] VIX: {e}")
        sys.exit(1)
    
    pct = compute_percentile(vix_series)
    pct_disp = pct * 100
    regime = get_regime(pct)
    signal = pct <= args.threshold
    
    print(f"[OK] Percentile: {pct_disp:.1f}%")
    print(f"[OK] Regime: {regime}")
    print(f"\n{'>>> ENTRY SIGNAL <<<' if signal else '--- HOLD ---'} (threshold: {args.threshold*100:.0f}%)\n")
    
    uvxy = load_uvxy_spot()
    if uvxy <= 0:
        print("[ERROR] UVXY load failed")
        sys.exit(1)
    print(f"[OK] UVXY: ${uvxy:.2f}")
    
    # Get variants (from JSON or fallback)
    print("[...] Loading variants...")
    variants = get_variants()
    src = "param_history.json" if load_profiles_from_json(1) else "fallback"
    print(f"[OK] {len(variants)} variants from {src}")
    
    # Generate quotes
    print("[...] Generating quotes...")
    quotes = [generate_quote(uvxy, vix, v) for v in variants]
    
    for q in quotes:
        print(f"    {q['name'][:20]:20} Net ${q['net_debit']:5.2f} | "
              f"Tgt ${q['target_val']:.2f} | Stop ${q['stop_val']:.2f} | "
              f"CAGR {q['cagr']*100:.0f}%")
    
    # Build HTML
    html = build_html(vix, pct_disp, regime, uvxy, quotes, args.threshold)
    
    # Output
    if args.json:
        out = {"vix": vix, "pct": pct_disp, "regime": regime, "uvxy": uvxy, 
               "signal": signal, "quotes": quotes}
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"\n{'='*50}")
        print(f"VIX ${vix:.2f} | Pct {pct_disp:.1f}% | {regime} | UVXY ${uvxy:.2f}")
        print(f"Signal: {'ACTIVE' if signal else 'HOLD'}")
        print(f"{'='*50}")
    
    # Email
    if args.email and (signal or args.force):
        print(f"\n[EMAIL] Sending to {args.email}...")
        if send_email(args.email, html, vix, pct_disp, regime, args.threshold):
            print("[OK] Email sent!")
        else:
            print("[ERROR] Email failed")
    elif args.email:
        print(f"\n[INFO] No signal - skipped (use --force)")
    
    sys.exit(0 if signal else 1)


if __name__ == "__main__":
    main()
