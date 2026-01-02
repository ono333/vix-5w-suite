#!/usr/bin/env python3
"""
VIX 5% Weekly Suite — ENHANCED APP
==================================

Pages:
- Dashboard
- Backtester (with grid scan)
- Live Signals (5 diagonal variants + Thursday email)
- Trade Explorer

Features:
- Quick jump buttons to sections
- All 5 variants EXPANDED by default (with collapse all option)
- Send Thursday Email directly from app (exact data, no recompute)
- Regime-adaptive 5-variant comparison
"""

import io
import os
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# Try importing yfinance for live data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# =====================================================================
# BLACK-SCHOLES PRICING
# =====================================================================

def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Vanilla Black-Scholes call price."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if T <= 0 else 0.0
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    except:
        return 0.0


# =====================================================================
# HELPERS
# =====================================================================

def _fmt_dollar(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except:
        return str(x)


def _fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
    except:
        return "n/a"


def _compute_vix_percentile_local(vix_weekly: pd.Series, lookback_weeks: int) -> pd.Series:
    """Rolling percentile of underlying level."""
    prices = vix_weekly.values.astype(float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    lb = max(1, int(lookback_weeks))
    for i in range(lb, n):
        window = prices[i - lb: i]
        out[i] = (window < prices[i]).mean()
    return pd.Series(out, index=vix_weekly.index, name="vix_pct")


def get_regime(percentile: float) -> str:
    """Classify VIX regime based on percentile."""
    if percentile <= 0.10:
        return "Ultra Low"
    elif percentile <= 0.25:
        return "Low"
    elif percentile <= 0.50:
        return "Medium"
    elif percentile <= 0.75:
        return "High"
    else:
        return "Extreme"


def load_live_data() -> Dict[str, Any]:
    """Load current VIX and UVXY data from yfinance."""
    if not HAS_YFINANCE:
        return {"error": "yfinance not installed"}
    
    try:
        # Get VIX data (52 weeks for percentile)
        end = dt.date.today()
        start = end - dt.timedelta(days=400)  # ~56 weeks buffer
        
        vix_df = yf.download("^VIX", start=start, end=end, progress=False)
        uvxy_df = yf.download("UVXY", start=start, end=end, progress=False)
        
        if vix_df.empty:
            return {"error": "No VIX data available"}
        
        # Get latest closes
        vix_close = float(vix_df["Close"].iloc[-1])
        uvxy_close = float(uvxy_df["Close"].iloc[-1]) if not uvxy_df.empty else 0.0
        
        # Compute 52-week percentile
        vix_closes = vix_df["Close"].values.flatten()
        if len(vix_closes) >= 52:
            lookback = vix_closes[-52:]
            current = vix_closes[-1]
            percentile = (lookback < current).mean()
        else:
            percentile = 0.5  # default if not enough data
        
        return {
            "vix_close": vix_close,
            "uvxy_close": uvxy_close,
            "percentile": percentile,
            "regime": get_regime(percentile),
            "date": end,
        }
    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# 5 DIAGONAL VARIANTS - QUOTE GENERATION
# =====================================================================

def generate_5_variants(uvxy_spot: float, percentile: float, 
                        initial_capital: float = 250000.0,
                        alloc_pct: float = 0.01) -> List[Dict[str, Any]]:
    """
    Generate 5 diagonal spread variants based on current market conditions.
    Returns list of dicts with all quote details for display and email.
    """
    r = 0.05  # risk-free rate
    sigma = 0.80  # UVXY typical vol
    capital = initial_capital * alloc_pct
    
    variants = []
    
    # =========================================================
    # VARIANT 1: Baseline Conservative (26-week LEAP)
    # =========================================================
    long_dte_weeks = 26
    short_dte_weeks = 1
    otm_pts = 2.0
    target_mult = 1.5
    stop_mult = 0.5
    
    long_strike = round(uvxy_spot + otm_pts, 1)
    short_strike = round(uvxy_spot + otm_pts + 1, 1)
    T_long = long_dte_weeks / 52.0
    T_short = short_dte_weeks / 52.0
    
    long_price = bs_call_price(uvxy_spot, long_strike, r, sigma, T_long)
    short_price = bs_call_price(uvxy_spot, short_strike, r, sigma, T_short)
    net_debit = long_price - short_price
    contracts = max(1, int(capital / (net_debit * 100))) if net_debit > 0 else 1
    
    variants.append({
        "name": "Baseline Conservative (26w)",
        "desc": "6-month LEAP + weekly short. Lower gamma, slower decay.",
        "long_strike": long_strike,
        "long_dte": long_dte_weeks * 7,
        "long_price": long_price,
        "short_strike": short_strike,
        "short_dte": short_dte_weeks * 7,
        "short_price": short_price,
        "net_debit": net_debit,
        "contracts": contracts,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
        "long_leg": f"BUY {contracts}x UVXY ${long_strike:.1f}C @ ${long_price:.2f} ({long_dte_weeks}w DTE)",
        "short_leg": f"SELL {contracts}x UVXY ${short_strike:.1f}C @ ${short_price:.2f} ({short_dte_weeks}w DTE)",
        "net_position": f"Net Debit: ${net_debit:.2f}/spread (${net_debit * contracts * 100:.0f} total)",
        "target": f"Target: {target_mult}x (${net_debit * target_mult:.2f})",
        "stop": f"Stop: {stop_mult}x (${net_debit * stop_mult:.2f})",
        "suggested": f"{contracts} contracts",
    })
    
    # =========================================================
    # VARIANT 2: Aggressive Short-Term (3-week)
    # =========================================================
    long_dte_weeks = 3
    short_dte_weeks = 1
    otm_pts = 1.5
    target_mult = 2.0
    stop_mult = 0.4
    
    long_strike = round(uvxy_spot + otm_pts, 1)
    short_strike = round(uvxy_spot + otm_pts + 0.5, 1)
    T_long = long_dte_weeks / 52.0
    T_short = short_dte_weeks / 52.0
    
    long_price = bs_call_price(uvxy_spot, long_strike, r, sigma, T_long)
    short_price = bs_call_price(uvxy_spot, short_strike, r, sigma, T_short)
    net_debit = long_price - short_price
    contracts = max(1, int(capital / (net_debit * 100))) if net_debit > 0 else 1
    
    variants.append({
        "name": "Aggressive Short-Term (3w)",
        "desc": "3-week long + weekly short. High gamma, fast moves.",
        "long_strike": long_strike,
        "long_dte": long_dte_weeks * 7,
        "long_price": long_price,
        "short_strike": short_strike,
        "short_dte": short_dte_weeks * 7,
        "short_price": short_price,
        "net_debit": net_debit,
        "contracts": contracts,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
        "long_leg": f"BUY {contracts}x UVXY ${long_strike:.1f}C @ ${long_price:.2f} ({long_dte_weeks}w DTE)",
        "short_leg": f"SELL {contracts}x UVXY ${short_strike:.1f}C @ ${short_price:.2f} ({short_dte_weeks}w DTE)",
        "net_position": f"Net Debit: ${net_debit:.2f}/spread (${net_debit * contracts * 100:.0f} total)",
        "target": f"Target: {target_mult}x (${net_debit * target_mult:.2f})",
        "stop": f"Stop: {stop_mult}x (${net_debit * stop_mult:.2f})",
        "suggested": f"{contracts} contracts",
    })
    
    # =========================================================
    # VARIANT 3: Aggressive Ultra-Short (1-week)
    # =========================================================
    long_dte_weeks = 1
    short_dte_weeks = 1
    otm_pts = 1.0
    target_mult = 3.0
    stop_mult = 0.3
    
    long_strike = round(uvxy_spot + otm_pts, 1)
    short_strike = round(uvxy_spot + otm_pts + 1.0, 1)
    T_long = long_dte_weeks / 52.0
    T_short = short_dte_weeks / 52.0
    
    long_price = bs_call_price(uvxy_spot, long_strike, r, sigma, T_long)
    short_price = bs_call_price(uvxy_spot, short_strike, r, sigma, T_short)
    net_debit = long_price - short_price
    contracts = max(1, int(capital / (net_debit * 100))) if net_debit > 0 else 1
    
    variants.append({
        "name": "Aggressive Ultra-Short (1w)",
        "desc": "Weekly vertical spread. Maximum gamma, binary outcome.",
        "long_strike": long_strike,
        "long_dte": long_dte_weeks * 7,
        "long_price": long_price,
        "short_strike": short_strike,
        "short_dte": short_dte_weeks * 7,
        "short_price": short_price,
        "net_debit": net_debit,
        "contracts": contracts,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
        "long_leg": f"BUY {contracts}x UVXY ${long_strike:.1f}C @ ${long_price:.2f} ({long_dte_weeks}w DTE)",
        "short_leg": f"SELL {contracts}x UVXY ${short_strike:.1f}C @ ${short_price:.2f} ({short_dte_weeks}w DTE)",
        "net_position": f"Net Debit: ${net_debit:.2f}/spread (${net_debit * contracts * 100:.0f} total)",
        "target": f"Target: {target_mult}x (${net_debit * target_mult:.2f})",
        "stop": f"Stop: {stop_mult}x (${net_debit * stop_mult:.2f})",
        "suggested": f"{contracts} contracts",
    })
    
    # =========================================================
    # VARIANT 4: Tighter Spread (1.5x target)
    # =========================================================
    long_dte_weeks = 13
    short_dte_weeks = 1
    otm_pts = 1.5
    target_mult = 1.5
    stop_mult = 0.6
    
    long_strike = round(uvxy_spot + otm_pts, 1)
    short_strike = round(uvxy_spot + otm_pts + 0.5, 1)
    T_long = long_dte_weeks / 52.0
    T_short = short_dte_weeks / 52.0
    
    long_price = bs_call_price(uvxy_spot, long_strike, r, sigma, T_long)
    short_price = bs_call_price(uvxy_spot, short_strike, r, sigma, T_short)
    net_debit = long_price - short_price
    contracts = max(1, int(capital / (net_debit * 100))) if net_debit > 0 else 1
    
    variants.append({
        "name": "Tighter Target (13w, 1.5x)",
        "desc": "Quarter LEAP, tighter profit target. Balanced risk/reward.",
        "long_strike": long_strike,
        "long_dte": long_dte_weeks * 7,
        "long_price": long_price,
        "short_strike": short_strike,
        "short_dte": short_dte_weeks * 7,
        "short_price": short_price,
        "net_debit": net_debit,
        "contracts": contracts,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
        "long_leg": f"BUY {contracts}x UVXY ${long_strike:.1f}C @ ${long_price:.2f} ({long_dte_weeks}w DTE)",
        "short_leg": f"SELL {contracts}x UVXY ${short_strike:.1f}C @ ${short_price:.2f} ({short_dte_weeks}w DTE)",
        "net_position": f"Net Debit: ${net_debit:.2f}/spread (${net_debit * contracts * 100:.0f} total)",
        "target": f"Target: {target_mult}x (${net_debit * target_mult:.2f})",
        "stop": f"Stop: {stop_mult}x (${net_debit * stop_mult:.2f})",
        "suggested": f"{contracts} contracts",
    })
    
    # =========================================================
    # VARIANT 5: Static Entry (Ignore Percentile)
    # =========================================================
    long_dte_weeks = 8
    short_dte_weeks = 1
    otm_pts = 2.0
    target_mult = 1.8
    stop_mult = 0.5
    
    long_strike = round(uvxy_spot + otm_pts, 1)
    short_strike = round(uvxy_spot + otm_pts + 1.0, 1)
    T_long = long_dte_weeks / 52.0
    T_short = short_dte_weeks / 52.0
    
    long_price = bs_call_price(uvxy_spot, long_strike, r, sigma, T_long)
    short_price = bs_call_price(uvxy_spot, short_strike, r, sigma, T_short)
    net_debit = long_price - short_price
    contracts = max(1, int(capital / (net_debit * 100))) if net_debit > 0 else 1
    
    variants.append({
        "name": "Static Entry (8w)",
        "desc": "2-month diagonal. Ignores percentile, always available.",
        "long_strike": long_strike,
        "long_dte": long_dte_weeks * 7,
        "long_price": long_price,
        "short_strike": short_strike,
        "short_dte": short_dte_weeks * 7,
        "short_price": short_price,
        "net_debit": net_debit,
        "contracts": contracts,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
        "long_leg": f"BUY {contracts}x UVXY ${long_strike:.1f}C @ ${long_price:.2f} ({long_dte_weeks}w DTE)",
        "short_leg": f"SELL {contracts}x UVXY ${short_strike:.1f}C @ ${short_price:.2f} ({short_dte_weeks}w DTE)",
        "net_position": f"Net Debit: ${net_debit:.2f}/spread (${net_debit * contracts * 100:.0f} total)",
        "target": f"Target: {target_mult}x (${net_debit * target_mult:.2f})",
        "stop": f"Stop: {stop_mult}x (${net_debit * stop_mult:.2f})",
        "suggested": f"{contracts} contracts",
    })
    
    return variants


# =====================================================================
# EMAIL FORMATTING & SENDING
# =====================================================================

def format_email_html(data: Dict[str, Any]) -> str:
    """Format the Thursday email with white background, large font."""
    today = dt.date.today().strftime("%B %d, %Y")
    pct = data['percentile'] * 100
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    signal_text = ">>> ENTRY SIGNAL ACTIVE <<<" if active else "No Signal (Above Threshold)"
    
    html = f"""
    <html>
    <body style="background:#ffffff;color:#333333;font-family:Arial,sans-serif;padding:20px;max-width:700px;margin:auto;line-height:1.6;font-size:18px;">
    
    <h1 style="color:#00aadd;text-align:center;margin-bottom:5px;">VIX 5% WEEKLY SUITE</h1>
    <h3 style="color:#666666;text-align:center;margin-top:0;margin-bottom:30px;">Thursday Signal Report — {today}</h3>
    
    <div style="padding:15px;border:1px solid #dddddd;margin-bottom:30px;background:#fafafa;">
        <strong style="font-size:20px;color:#00aadd;">📊 MARKET STATE</strong><br><br>
        <strong>VIX Close:</strong> ${data['vix_close']:.2f}<br>
        <strong>52w Percentile:</strong> {pct:.1f}%<br>
        <strong>Current Regime:</strong> {data['regime']}<br>
        <strong>UVXY Spot:</strong> ${data['uvxy_close']:.2f}
    </div>
    
    <div style="padding:20px;border:3px solid {'#00aa00' if active else '#aa0000'};background:{'#f0fff0' if active else '#fff0f0'};margin-bottom:40px;text-align:center;">
        <strong style="font-size:28px;color:{'#00aa00' if active else '#aa0000'};">{emoji} {signal_text}</strong><br>
        <span style="font-size:16px;color:#666;">Percentile ({pct:.1f}%) {'≤' if active else '>'} threshold (35%)</span>
    </div>
    
    <h2 style="color:#00aadd;font-size:24px;margin-bottom:20px;border-bottom:2px solid #00aadd;padding-bottom:10px;">
        📋 5 DIAGONAL VARIANTS
    </h2>
    """
    
    for i, v in enumerate(data['variants'], 1):
        html += f"""
        <div style="padding:15px;border:1px solid #cccccc;margin-bottom:20px;background:#f9f9f9;border-radius:5px;">
            <strong style="font-size:20px;color:#00aadd;">#{i} {v['name']}</strong><br>
            <em style="color:#666;">{v['desc']}</em><br><br>
            
            <div style="background:#e8f5e9;padding:10px;margin:5px 0;border-left:4px solid #4caf50;">
                <strong style="color:#2e7d32;">📈 LONG LEG (Buy):</strong><br>
                {v['long_leg']}
            </div>
            
            <div style="background:#ffebee;padding:10px;margin:5px 0;border-left:4px solid #f44336;">
                <strong style="color:#c62828;">📉 SHORT LEG (Sell):</strong><br>
                {v['short_leg']}
            </div>
            
            <div style="background:#e3f2fd;padding:10px;margin:5px 0;border-left:4px solid #2196f3;">
                <strong style="color:#1565c0;">💰 NET POSITION:</strong><br>
                {v['net_position']}<br>
                {v['target']}<br>
                {v['stop']}<br>
                <strong>Suggested: {v['suggested']}</strong>
            </div>
        </div>
        """
    
    html += """
    <div style="margin-top:40px;padding:15px;background:#fff3e0;border:1px solid #ffcc80;border-radius:5px;">
        <p style="color:#e65100;margin:0;font-size:14px;text-align:center;">
            ⚠️ <strong>DISCLAIMER:</strong> Research tool only — not financial advice.<br>
            Always verify quotes with your broker before trading.<br>
            Past performance does not guarantee future results.
        </p>
    </div>
    
    <p style="color:#999;font-size:12px;text-align:center;margin-top:30px;">
        Generated by VIX 5% Weekly Suite | LBR-Grade Research Platform
    </p>
    
    </body>
    </html>
    """
    return html


def send_thursday_email(data: Dict[str, Any], recipient: str = "onoshin333@gmail.com") -> tuple:
    """Send the Thursday email using SMTP env vars. Returns (success, message)."""
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    
    if not all([smtp_server, smtp_user, smtp_pass]):
        return False, "SMTP environment variables not set (SMTP_SERVER, SMTP_USER, SMTP_PASS)"
    
    try:
        msg = MIMEMultipart("alternative")
        pct = data['percentile'] * 100
        active = data['signal_active']
        emoji = "🟢" if active else "🔴"
        
        if active:
            subject = f"{emoji} [ENTRY SIGNAL] VIX {data['regime']} Regime ({pct:.1f}%)"
        else:
            subject = f"{emoji} [NO SIGNAL] VIX {data['regime']} Regime ({pct:.1f}%)"
        
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient
        
        html = format_email_html(data)
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        return True, f"Email sent successfully to {recipient}"
    except Exception as e:
        return False, f"Email failed: {str(e)}"


# =====================================================================
# PAGE: LIVE SIGNALS
# =====================================================================

def page_live_signals():
    """Live Signals page with 5 variants, quick jump, and email button."""
    
    st.title("📡 Live Signals — 5 Variant Comparison")
    
    # =========================================================
    # QUICK JUMP BUTTONS (at top)
    # =========================================================
    st.markdown("### ⚡ Quick Jump")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<a href="#market-state" style="text-decoration:none;"><button style="width:100%;padding:10px;font-size:16px;cursor:pointer;">📊 Market State</button></a>', unsafe_allow_html=True)
    with col2:
        st.markdown('<a href="#variant-signals" style="text-decoration:none;"><button style="width:100%;padding:10px;font-size:16px;cursor:pointer;">📋 5 Variant Signals</button></a>', unsafe_allow_html=True)
    with col3:
        st.markdown('<a href="#thursday-email" style="text-decoration:none;"><button style="width:100%;padding:10px;font-size:16px;cursor:pointer;">📧 Thursday Email</button></a>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================
    # LOAD LIVE DATA
    # =========================================================
    with st.spinner("Loading live market data..."):
        live_data = load_live_data()
    
    if "error" in live_data:
        st.error(f"Failed to load data: {live_data['error']}")
        st.info("Make sure yfinance is installed: `pip install yfinance`")
        return
    
    vix_close = live_data['vix_close']
    uvxy_close = live_data['uvxy_close']
    percentile = live_data['percentile']
    regime = live_data['regime']
    
    ENTRY_THRESHOLD = 0.35
    signal_active = percentile <= ENTRY_THRESHOLD
    
    # =========================================================
    # MARKET STATE SECTION
    # =========================================================
    st.markdown('<div id="market-state"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Current Market State")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VIX Close", f"${vix_close:.2f}")
    col2.metric("52w Percentile", f"{percentile*100:.1f}%")
    col3.metric("Regime", regime)
    col4.metric("UVXY Spot", f"${uvxy_close:.2f}")
    
    if signal_active:
        st.success(f"🟢 **ENTRY SIGNAL ACTIVE** — Percentile ({percentile*100:.1f}%) ≤ {ENTRY_THRESHOLD*100:.0f}% threshold")
    else:
        st.warning(f"🔴 **NO SIGNAL** — Percentile ({percentile*100:.1f}%) > {ENTRY_THRESHOLD*100:.0f}% threshold")
    
    st.markdown("---")
    
    # =========================================================
    # GENERATE 5 VARIANTS
    # =========================================================
    initial_capital = st.session_state.get('live_initial_capital', 250000.0)
    alloc_pct = st.session_state.get('live_alloc_pct', 0.01)
    
    variants = generate_5_variants(
        uvxy_spot=uvxy_close,
        percentile=percentile,
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
    )
    
    # Store for email use
    st.session_state['live_signal_data'] = {
        'date': dt.date.today().isoformat(),
        'vix_close': vix_close,
        'uvxy_close': uvxy_close,
        'percentile': percentile,
        'regime': regime,
        'signal_active': signal_active,
        'variants': variants,
    }
    
    # =========================================================
    # 5 VARIANT SIGNALS SECTION
    # =========================================================
    st.markdown('<div id="variant-signals"></div>', unsafe_allow_html=True)
    st.markdown("## 📋 5 Diagonal Variant Signals")
    
    # Expand/Collapse toggle — DEFAULT IS EXPANDED
    col1, col2 = st.columns([3, 1])
    with col2:
        if 'expand_all' not in st.session_state:
            st.session_state['expand_all'] = True  # DEFAULT EXPANDED
        
        btn_label = "🔽 Collapse All" if st.session_state['expand_all'] else "🔼 Expand All"
        if st.button(btn_label, use_container_width=True):
            st.session_state['expand_all'] = not st.session_state['expand_all']
            st.rerun()
    
    expand_all = st.session_state.get('expand_all', True)
    
    # Display each variant — EXPANDED BY DEFAULT
    for i, v in enumerate(variants, 1):
        with st.expander(f"**#{i} {v['name']}**", expanded=expand_all):
            st.markdown(f"*{v['desc']}*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 LONG LEG (Buy)")
                st.code(v['long_leg'], language=None)
                
            with col2:
                st.markdown("#### 📉 SHORT LEG (Sell)")
                st.code(v['short_leg'], language=None)
            
            st.markdown("#### 💰 Net Position")
            st.info(f"""
**{v['net_position']}**

{v['target']}  
{v['stop']}  

**Suggested:** {v['suggested']}
            """)
    
    st.markdown("---")
    
    # =========================================================
    # THURSDAY EMAIL SECTION
    # =========================================================
    st.markdown('<div id="thursday-email"></div>', unsafe_allow_html=True)
    st.markdown("## 📧 Thursday Email")
    
    st.markdown("""
    Send the signal report email with **exact data** shown above.  
    Uses SMTP env vars: `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        email_recipient = st.text_input("Recipient email", value="onoshin333@gmail.com", key="email_recipient")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_send = st.checkbox("Force send (even if no signal)", value=False)
    
    # Preview button
    if st.button("👁️ Preview Email HTML", use_container_width=True):
        data = st.session_state.get('live_signal_data', {})
        if data:
            html = format_email_html(data)
            st.components.v1.html(html, height=800, scrolling=True)
        else:
            st.error("No signal data available")
    
    # Send button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Send Thursday Email", type="primary", use_container_width=True):
            data = st.session_state.get('live_signal_data', {})
            
            if not data:
                st.error("No signal data available. Refresh the page.")
            elif not data.get('signal_active') and not force_send:
                st.warning("No active signal. Check 'Force send' to send anyway.")
            else:
                with st.spinner("Sending email..."):
                    success, message = send_thursday_email(data, email_recipient)
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
    
    with col2:
        if st.button("💾 Save HTML Preview", use_container_width=True):
            data = st.session_state.get('live_signal_data', {})
            if data:
                html = format_email_html(data)
                st.download_button(
                    label="Download HTML",
                    data=html,
                    file_name=f"vix_signal_{dt.date.today().isoformat()}.html",
                    mime="text/html"
                )
    
    # =========================================================
    # DISCLAIMER
    # =========================================================
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer:** This is a research tool only, not financial advice. 
    All prices are theoretical (Black-Scholes). Always verify quotes with your broker before trading.
    Past performance does not guarantee future results.
    """)


# =====================================================================
# SIMPLE SIDEBAR
# =====================================================================

def build_simple_sidebar() -> Dict[str, Any]:
    """Build sidebar with page selection and key params."""
    
    st.sidebar.title("VIX 5% Weekly Suite")
    
    page = st.sidebar.radio(
        "Page",
        ["Dashboard", "Backtester", "Live Signals", "Trade Explorer"],
        index=2,  # Default to Live Signals
        key="page_select"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    
    start_date = st.sidebar.date_input("Start date", value=dt.date(2015, 1, 1), key="sb_start_date")
    end_date = st.sidebar.date_input("End date", value=dt.date.today(), key="sb_end_date")
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", min_value=10000.0, max_value=10000000.0,
        value=250000.0, step=10000.0, key="sb_initial_capital"
    )
    
    alloc_pct_raw = st.sidebar.slider(
        "Allocation (%)", min_value=0.5, max_value=10.0,
        value=1.0, step=0.5, key="sb_alloc_pct"
    )
    alloc_pct = alloc_pct_raw / 100.0
    
    # Store values for Live Signals page (use different keys)
    st.session_state['live_initial_capital'] = initial_capital
    st.session_state['live_alloc_pct'] = alloc_pct
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy")
    
    mode = st.sidebar.selectbox("Position Structure", ["diagonal", "long_only"], index=0, key="sb_mode")
    entry_percentile = st.sidebar.slider("Entry Percentile", min_value=0.05, max_value=0.50, value=0.35, step=0.05, key="sb_entry_pct")
    
    return {
        "page": page,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "alloc_pct": alloc_pct,
        "mode": mode,
        "entry_percentile": entry_percentile,
        "pricing_source": "Synthetic (BS)",
        "underlying_symbol": "UVXY",
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        page_icon="📈",
        layout="wide",
    )
    
    params = build_simple_sidebar()
    page = params["page"]
    
    if page == "Live Signals":
        page_live_signals()
        return
    
    elif page == "Dashboard":
        st.title("📊 VIX 5% Weekly — Dashboard")
        st.info("Dashboard shows backtest equity curves and regime analysis.")
        
        if HAS_YFINANCE:
            with st.spinner("Loading VIX data..."):
                try:
                    df = yf.download("^VIX", start=params["start_date"], end=params["end_date"], progress=False)
                    if not df.empty:
                        vix_weekly = df["Close"].resample("W-FRI").last().dropna()
                        st.markdown("### VIX Weekly Close")
                        st.line_chart(vix_weekly)
                        st.markdown("### 52-Week Percentile")
                        pct = _compute_vix_percentile_local(vix_weekly, 52)
                        st.area_chart(pct)
                    else:
                        st.warning("No VIX data available for selected range")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
        else:
            st.warning("Install yfinance: `pip install yfinance`")
    
    elif page == "Backtester":
        st.title("🔬 VIX 5% Weekly — Backtester")
        st.info("Backtester with grid scan for parameter optimization.")
        st.markdown("*Full backtester functionality requires core modules.*")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Capital", _fmt_dollar(params["initial_capital"]))
        col2.metric("Allocation", f"{params['alloc_pct']*100:.1f}%")
        col3.metric("Entry Threshold", f"{params['entry_percentile']*100:.0f}%")
    
    elif page == "Trade Explorer":
        st.title("🔍 VIX 5% Weekly — Trade Explorer")
        st.info("Trade Explorer for analyzing individual trade entries and exits.")
        st.markdown("*Coming soon: Detailed trade-by-trade analysis*")


if __name__ == "__main__":
    main()
