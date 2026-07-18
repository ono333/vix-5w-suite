#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Live Signals JSON Exporter

This module can be:
1. Imported by app.py to add export button to Live Signals page
2. Run standalone to generate signals.json for the emailer

Usage:
    python live_signals_export.py                    # Generate signals.json
    python live_signals_export.py --output /path/to  # Custom output path
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================================
# DATA LOADING (same as Streamlit app)
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
# 5 VARIANTS (Exact match to Streamlit Live Signals page)
# =============================================================================

def generate_variants(uvxy_spot: float, vix_close: float) -> List[Dict[str, Any]]:
    """
    Generate 5 diagonal variants matching Streamlit Live Signals exactly.
    
    Variants:
    1. Baseline (26w) - Standard 6-month diagonal
    2. Aggressive (1w) - Ultra-short for quick theta
    3. Aggressive (3w) - Short DTE, faster decay
    4. Tighter Exit (1.5x) - Quick profit target
    5. Static Benchmark - Conservative reference
    """
    from math import log, sqrt, exp
    from scipy.stats import norm
    
    def bs_call(S, K, T, r, sig):
        if T <= 0 or sig <= 0:
            return max(S - K, 0)
        d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    
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
            # Raw values for JSON consumers
            "long_strike": long_K,
            "long_mid": long_mid,
            "long_exp_date": long_exp,
            "long_dte_days": long_dte,
            "short_strike": short_K,
            "short_mid": short_mid,
            "short_exp_date": short_exp,
            "short_dte_days": short_dte,
            "net_debit_raw": net_debit,
            "risk_per_contract": risk,
            "target_value": target_val,
            "target_mult": target_mult,
            "stop_value": stop_val,
            "stop_mult": stop_mult,
        })
    
    return variants


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def generate_signals_json(threshold: float = 0.35) -> Dict[str, Any]:
    """
    Generate complete signals data matching Streamlit Live Signals page.
    
    Returns dict ready for JSON export.
    """
    # Load market data
    vix_series = load_vix_weekly()
    vix_close = _scalar(vix_series.iloc[-1])
    uvxy_spot = load_uvxy_spot()
    
    # Compute percentile and regime
    percentile = compute_percentile(vix_series)
    regime = get_regime(percentile)
    signal_active = percentile <= threshold
    
    # Generate variants
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


def export_signals_json(output_path: str = None) -> str:
    """Export signals to JSON file. Returns path to file."""
    data = generate_signals_json()
    
    if output_path is None:
        output_path = Path(__file__).parent / "signals.json"
    else:
        output_path = Path(output_path)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    return str(output_path)


# =============================================================================
# STREAMLIT INTEGRATION
# =============================================================================

def add_export_button_to_streamlit(st, data: Dict[str, Any] = None):
    """
    Call this from your Streamlit Live Signals page to add export button.
    
    Usage in app.py:
        from live_signals_export import add_export_button_to_streamlit, generate_signals_json
        
        # In page_live_signals():
        data = generate_signals_json()
        add_export_button_to_streamlit(st, data)
    """
    if data is None:
        data = generate_signals_json()
    
    json_str = json.dumps(data, indent=2, default=str)
    
    st.download_button(
        label="📥 Export Live Signals as JSON",
        data=json_str,
        file_name=f"vix_signals_{dt.date.today()}.json",
        mime="application/json",
        help="Download for email automation"
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VIX Live Signals to JSON")
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output path (default: ./signals.json)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("VIX 5% Weekly - Live Signals Export")
    print(f"    {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        path = export_signals_json(args.output)
        print(f"\n[OK] Exported to: {path}")
        
        # Print summary
        with open(path) as f:
            data = json.load(f)
        
        print(f"\nVIX: ${data['vix_close']:.2f} | Percentile: {data['percentile']:.1f}%")
        print(f"Regime: {data['regime']} | Signal: {'ACTIVE' if data['signal_active'] else 'HOLD'}")
        print(f"UVXY: ${data['uvxy_spot']:.2f}")
        print(f"\nVariants exported: {len(data['variants'])}")
        for v in data['variants']:
            print(f"  - {v['name']}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
