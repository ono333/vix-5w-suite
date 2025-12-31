#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Thursday Signal Emailer

Generates PNG "screenshot" of Live Signals and emails it every Thursday.
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
from typing import Dict, Any, Optional
from io import BytesIO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.param_history import get_best_for_regime, apply_regime_params

# =============================================================================
# CONFIGURATION - Edit these for your setup
# =============================================================================
DEFAULT_EMAIL = "shin.takahama@gmail.com"  # Your email address
ENTRY_THRESHOLD = 0.35  # Entry signal when percentile <= this (35%)

# SMTP defaults (override with environment variables)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================
REGIME_THRESHOLDS = [
    ("ULTRA_LOW", 0.00, 0.10),
    ("LOW", 0.10, 0.25),
    ("MEDIUM", 0.25, 0.50),
    ("HIGH", 0.50, 0.75),
    ("EXTREME", 0.75, 1.00),
]

REGIME_COLORS = {
    "ULTRA_LOW": "#00ff00",  # Bright green
    "LOW": "#90EE90",        # Light green
    "MEDIUM": "#FFD700",     # Gold
    "HIGH": "#FFA500",       # Orange
    "EXTREME": "#FF4500",    # Red-orange
}


def get_regime(percentile: float) -> str:
    """Map percentile to regime name."""
    for name, pct_min, pct_max in REGIME_THRESHOLDS:
        if pct_min <= percentile < pct_max:
            return name
    return "EXTREME"


# =============================================================================
# DATA FETCHING
# =============================================================================
def get_vix_percentile(lookback_weeks: int = 52) -> Dict[str, Any]:
    """Get current VIX price and percentile."""
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(weeks=lookback_weeks + 10)
        
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        if vix.empty:
            return {"error": "No VIX data"}
        
        # Handle multi-level columns
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        col = "Adj Close" if "Adj Close" in vix.columns else "Close"
        weekly = vix[col].resample("W-FRI").last().dropna()
        
        if len(weekly) < 10:
            return {"error": "Insufficient data"}
        
        prices = weekly.iloc[-lookback_weeks:].values.astype(float).ravel()
        current = float(prices[-1])
        
        # Compute percentile
        below = np.sum(prices[:-1] < current)
        pct = below / (len(prices) - 1)
        
        regime = get_regime(pct)
        
        return {
            "current_price": current,
            "percentile": pct,
            "percentile_pct": pct * 100,
            "regime": regime,
            "lookback_weeks": lookback_weeks,
            "timestamp": dt.datetime.now().isoformat(),
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_uvxy_diagonal_quote(regime: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get current UVXY diagonal spread quotes."""
    try:
        uvxy = yf.Ticker("UVXY")
        
        # Get spot price
        hist = uvxy.history(period="5d")
        if hist.empty:
            return {"error": "No UVXY price data"}
        
        spot = float(hist['Close'].iloc[-1])
        
        # Get available expirations
        exps = uvxy.options
        if not exps:
            return {"error": "No options available"}
        
        today = dt.date.today()
        
        # Parse expirations
        future_exps = []
        for exp_str in exps:
            try:
                exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date > today + dt.timedelta(days=2):
                    future_exps.append(exp_date)
            except:
                continue
        
        if not future_exps:
            return {"error": "No future expirations"}
        
        future_exps.sort()
        
        # Get regime-specific parameters
        if params is None:
            params = {"mode": "diagonal"}
        
        regime_params = apply_regime_params(params, regime)
        otm_pts = float(regime_params.get("otm_pts", 10.0))
        long_dte_weeks = int(regime_params.get("long_dte_weeks", 26))
        entry_percentile = float(regime_params.get("entry_percentile", 0.15))
        
        # Find short expiration (~1 week)
        target_short = today + dt.timedelta(weeks=1)
        short_exp = min(future_exps, key=lambda e: abs((e - target_short).days))
        
        # Find long expiration
        target_long = today + dt.timedelta(weeks=long_dte_weeks)
        long_exp = min(future_exps, key=lambda e: abs((e - target_long).days))
        
        # Get chains
        short_chain = uvxy.option_chain(short_exp.strftime("%Y-%m-%d")).calls
        long_chain = uvxy.option_chain(long_exp.strftime("%Y-%m-%d")).calls
        
        if short_chain.empty or long_chain.empty:
            return {"error": "Empty option chains"}
        
        # Find strikes
        long_strike_target = round(spot + otm_pts + 2)
        short_strike_target = round(spot + otm_pts)
        
        # Long leg
        long_otm = long_chain[long_chain['strike'] >= long_strike_target]
        if long_otm.empty:
            long_otm = long_chain[long_chain['strike'] >= spot]
        
        if long_otm.empty:
            return {"error": "No long strikes"}
        
        long_row = long_otm.iloc[0]
        long_strike = float(long_row['strike'])
        long_bid = float(long_row['bid']) if pd.notna(long_row['bid']) else 0.0
        long_ask = float(long_row['ask']) if pd.notna(long_row['ask']) else 0.0
        long_mid = (long_bid + long_ask) / 2 if long_ask > 0 else long_bid
        
        # Short leg
        short_otm = short_chain[
            (short_chain['strike'] >= short_strike_target) & 
            (short_chain['strike'] <= spot + otm_pts + 5)
        ]
        if short_otm.empty:
            short_otm = short_chain[short_chain['strike'] >= spot]
        
        if short_otm.empty:
            return {"error": "No short strikes"}
        
        short_row = short_otm.iloc[0]
        short_strike = float(short_row['strike'])
        short_bid = float(short_row['bid']) if pd.notna(short_row['bid']) else 0.0
        short_ask = float(short_row['ask']) if pd.notna(short_row['ask']) else 0.0
        short_mid = (short_bid + short_ask) / 2 if short_ask > 0 else short_bid
        
        net_debit = long_ask - short_bid if (long_ask > 0 and short_bid > 0) else long_mid - short_mid
        net_debit_mid = long_mid - short_mid
        
        # Sizing calculation (1% of $250k = $2,500 risk budget)
        capital = 250000
        risk_pct = 0.01
        risk_budget = capital * risk_pct
        max_loss_per_contract = net_debit * 100  # max loss = net debit paid
        suggested_contracts = int(risk_budget / max_loss_per_contract) if max_loss_per_contract > 0 else 1
        suggested_contracts = max(1, min(suggested_contracts, 50))  # 1-50 range
        
        return {
            "spot": round(spot, 2),
            "regime": regime,
            
            "long_exp": long_exp.strftime("%Y-%m-%d"),
            "long_dte": (long_exp - today).days,
            "long_strike": long_strike,
            "long_bid": round(long_bid, 2),
            "long_ask": round(long_ask, 2),
            "long_mid": round(long_mid, 2),
            "long_leg": f"UVXY {long_exp.strftime('%b %d')} ${long_strike:.0f}C",
            
            "short_exp": short_exp.strftime("%Y-%m-%d"),
            "short_dte": (short_exp - today).days,
            "short_strike": short_strike,
            "short_bid": round(short_bid, 2),
            "short_ask": round(short_ask, 2),
            "short_mid": round(short_mid, 2),
            "short_leg": f"UVXY {short_exp.strftime('%b %d')} ${short_strike:.0f}C",
            
            "net_debit": round(net_debit, 2),
            "net_debit_mid": round(net_debit_mid, 2),
            
            "suggested_contracts": suggested_contracts,
            "risk_per_contract": round(net_debit * 100, 0),
            "total_risk": round(suggested_contracts * net_debit * 100, 0),
            
            "otm_pts_used": otm_pts,
            "entry_percentile_regime": entry_percentile,
            "timestamp": dt.datetime.now().isoformat(),
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# =============================================================================
# PNG GENERATION - Live Signals "Screenshot"
# =============================================================================
def generate_signal_png(vix_data: Dict, quote_data: Dict, threshold: float = 0.35) -> BytesIO:
    """
    Generate a PNG "screenshot" of the Live Signals tab.
    
    Returns BytesIO buffer containing PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    # Colors
    text_color = '#ffffff'
    accent_color = '#00d4ff'
    green_color = '#00ff00'
    red_color = '#ff4444'
    gold_color = '#ffd700'
    
    regime = vix_data.get('regime', 'UNKNOWN')
    regime_color = REGIME_COLORS.get(regime, '#888888')
    percentile = vix_data.get('percentile', 0) * 100
    vix_price = vix_data.get('current_price', 0)
    signal_active = vix_data.get('percentile', 1.0) <= threshold
    
    y = 0.95
    line_height = 0.045
    
    # Title
    ax.text(0.5, y, "VIX 5% WEEKLY SUITE", transform=ax.transAxes, fontsize=20, 
            fontweight='bold', color=accent_color, ha='center', fontfamily='monospace')
    y -= line_height * 1.5
    
    ax.text(0.5, y, f"Thursday Signal Report - {dt.date.today().strftime('%B %d, %Y')}", 
            transform=ax.transAxes, fontsize=12, color='#888888', ha='center', fontfamily='monospace')
    y -= line_height * 2
    
    # Divider
    ax.axhline(y=y + 0.01, xmin=0.1, xmax=0.9, color='#333355', linewidth=2, transform=ax.transAxes)
    y -= line_height
    
    # Market State Box
    ax.text(0.05, y, "📊 MARKET STATE", transform=ax.transAxes, fontsize=14, 
            fontweight='bold', color=gold_color, fontfamily='monospace')
    y -= line_height
    
    ax.text(0.08, y, f"VIX Close:       ${vix_price:.2f}", transform=ax.transAxes, 
            fontsize=13, color=text_color, fontfamily='monospace')
    y -= line_height
    
    ax.text(0.08, y, f"52w Percentile:  {percentile:.1f}%", transform=ax.transAxes, 
            fontsize=13, color=text_color, fontfamily='monospace')
    y -= line_height
    
    ax.text(0.08, y, f"Current Regime:  ", transform=ax.transAxes, 
            fontsize=13, color=text_color, fontfamily='monospace')
    ax.text(0.30, y, f"{regime}", transform=ax.transAxes, 
            fontsize=13, fontweight='bold', color=regime_color, fontfamily='monospace')
    y -= line_height * 1.5
    
    # Signal Status
    if signal_active:
        signal_text = "🟢 ENTRY SIGNAL ACTIVE"
        signal_color = green_color
        signal_detail = f"Percentile ({percentile:.1f}%) ≤ threshold ({threshold*100:.0f}%)"
    else:
        signal_text = "🔴 HOLD - No Entry Signal"
        signal_color = red_color
        signal_detail = f"Percentile ({percentile:.1f}%) > threshold ({threshold*100:.0f}%)"
    
    ax.text(0.05, y, signal_text, transform=ax.transAxes, fontsize=16, 
            fontweight='bold', color=signal_color, fontfamily='monospace')
    y -= line_height
    
    ax.text(0.08, y, signal_detail, transform=ax.transAxes, fontsize=11, 
            color='#888888', fontfamily='monospace')
    y -= line_height * 2
    
    # Divider
    ax.axhline(y=y + 0.01, xmin=0.1, xmax=0.9, color='#333355', linewidth=2, transform=ax.transAxes)
    y -= line_height
    
    # Trade Details
    if "error" not in quote_data:
        ax.text(0.05, y, "📈 RECOMMENDED DIAGONAL SPREAD", transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color=gold_color, fontfamily='monospace')
        y -= line_height * 1.2
        
        ax.text(0.08, y, f"UVXY Spot: ${quote_data.get('spot', 0):.2f}", transform=ax.transAxes, 
                fontsize=12, color='#aaaaaa', fontfamily='monospace')
        y -= line_height * 1.5
        
        # Long Leg
        ax.text(0.08, y, "LONG LEG (BUY):", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color=green_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"{quote_data.get('long_leg', 'N/A')}", transform=ax.transAxes, 
                fontsize=13, color=text_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Bid: ${quote_data.get('long_bid', 0):.2f}  Ask: ${quote_data.get('long_ask', 0):.2f}  Mid: ${quote_data.get('long_mid', 0):.2f}", 
                transform=ax.transAxes, fontsize=11, color='#aaaaaa', fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"DTE: {quote_data.get('long_dte', 0)} days", transform=ax.transAxes, 
                fontsize=11, color='#888888', fontfamily='monospace')
        y -= line_height * 1.5
        
        # Short Leg
        ax.text(0.08, y, "SHORT LEG (SELL):", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color=red_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"{quote_data.get('short_leg', 'N/A')}", transform=ax.transAxes, 
                fontsize=13, color=text_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Bid: ${quote_data.get('short_bid', 0):.2f}  Ask: ${quote_data.get('short_ask', 0):.2f}  Mid: ${quote_data.get('short_mid', 0):.2f}", 
                transform=ax.transAxes, fontsize=11, color='#aaaaaa', fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"DTE: {quote_data.get('short_dte', 0)} days", transform=ax.transAxes, 
                fontsize=11, color='#888888', fontfamily='monospace')
        y -= line_height * 1.5
        
        # Net Position
        ax.text(0.08, y, "NET POSITION:", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color=accent_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Net Debit (mid):  ${quote_data.get('net_debit_mid', 0):.2f}", transform=ax.transAxes, 
                fontsize=13, color=text_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Net Debit (cons): ${quote_data.get('net_debit', 0):.2f}", transform=ax.transAxes, 
                fontsize=12, color='#aaaaaa', fontfamily='monospace')
        y -= line_height * 1.5
        
        # Sizing
        ax.text(0.08, y, "POSITION SIZING:", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color=gold_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Suggested Contracts: {quote_data.get('suggested_contracts', 1)}", transform=ax.transAxes, 
                fontsize=13, color=text_color, fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Risk per Contract:   ${quote_data.get('risk_per_contract', 0):.0f}", transform=ax.transAxes, 
                fontsize=12, color='#aaaaaa', fontfamily='monospace')
        y -= line_height
        
        ax.text(0.10, y, f"Total Position Risk: ${quote_data.get('total_risk', 0):.0f}", transform=ax.transAxes, 
                fontsize=12, color='#aaaaaa', fontfamily='monospace')
        
    else:
        ax.text(0.08, y, f"⚠️ Quote Error: {quote_data.get('error', 'Unknown')}", transform=ax.transAxes, 
                fontsize=12, color=red_color, fontfamily='monospace')
    
    # Footer
    ax.text(0.5, 0.02, "⚠️ Research tool only — not financial advice. Verify quotes with broker.", 
            transform=ax.transAxes, fontsize=9, color='#666666', ha='center', fontfamily='monospace')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e', 
                edgecolor='none', dpi=150)
    buf.seek(0)
    plt.close()
    
    return buf


# =============================================================================
# EMAIL SENDING
# =============================================================================
def send_signal_email(
    to_email: str,
    vix_data: Dict,
    quote_data: Dict,
    png_buffer: BytesIO,
    threshold: float = 0.35,
) -> bool:
    """
    Send signal email with PNG attachment.
    
    Returns True if sent successfully.
    """
    if not SMTP_USER or not SMTP_PASS:
        print("⚠️ SMTP credentials not configured")
        print("   Set SMTP_USER and SMTP_PASS environment variables")
        print("   Or configure in daily_signal.py")
        return False
    
    try:
        regime = vix_data.get('regime', 'UNKNOWN')
        percentile = vix_data.get('percentile_pct', 0)
        signal_active = vix_data.get('percentile', 1.0) <= threshold
        
        # Subject line
        if signal_active:
            subject = f"🟢 VIX Entry Signal - {regime} Regime ({percentile:.1f}%)"
        else:
            subject = f"📊 VIX Weekly Report - {regime} Regime ({percentile:.1f}%)"
        
        # Create message
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        
        body = body.encode('ascii', 'replace').decode('ascii')
        
        # HTML body with embedded image
        html = f"""
        <html>
        <body style="background-color: #1a1a2e; color: #ffffff; font-family: monospace; padding: 20px;">
            <h2 style="color: #00d4ff;">VIX 5% Weekly Suite - Thursday Signal</h2>
            <p style="color: #888888;">{dt.date.today().strftime('%B %d, %Y')}</p>
            
            <p><strong>Regime:</strong> <span style="color: {REGIME_COLORS.get(regime, '#888888')};">{regime}</span></p>
            <p><strong>VIX Percentile:</strong> {percentile:.1f}%</p>
            <p><strong>Signal:</strong> {'🟢 ENTRY ACTIVE' if signal_active else '🔴 HOLD'}</p>
            
            <hr style="border-color: #333355;">
            
            <p>See attached PNG for full signal details.</p>
            
            <img src="cid:signal_image" style="max-width: 100%; border: 1px solid #333355;">
            
            <hr style="border-color: #333355;">
            <p style="color: #666666; font-size: 11px;">
                ⚠️ This is a research tool, not financial advice. Always verify quotes with your broker.
            </p>
        </body>
        </html>
        """
        
        html_part = MIMEText(html, 'html')
        msg.attach(html_part)
        
        # Attach PNG (inline)
        png_buffer.seek(0)
        img = MIMEImage(png_buffer.read())
        img.add_header('Content-ID', '<signal_image>')
        img.add_header('Content-Disposition', 'inline', filename='signal.png')
        msg.attach(img)
        
        # Also attach as downloadable file
        png_buffer.seek(0)
        img_attach = MIMEImage(png_buffer.read())
        img_attach.add_header('Content-Disposition', 'attachment', 
                              filename=f'VIX_Signal_{dt.date.today().strftime("%Y%m%d")}.png')
        msg.attach(img_attach)
        
        # Send
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        
        return True
        
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False


# =============================================================================
# TEXT REPORT (Fallback) - Bulletproof ASCII-only version
# =============================================================================
def format_text_report(vix_data: Dict, quote_data: Dict, threshold: float = 0.35) -> str:
    """Format signal data as plain text report - ASCII-only for SMTP compatibility."""
    
    signal_active = vix_data.get('percentile', 1.0) <= threshold
    
    # Extract all values upfront
    vix_close = vix_data.get('current_price', 0)
    percentile = vix_data.get('percentile_pct', 0)
    regime = vix_data.get('regime', 'N/A')
    threshold_pct = threshold * 100
    
    # Signal line
    if signal_active:
        signal_line = f">>> ENTRY SIGNAL ACTIVE <<<\n    Percentile ({percentile:.1f}%) <= threshold ({threshold_pct:.0f}%)"
    else:
        signal_line = f"HOLD - No entry signal\n    Percentile ({percentile:.1f}%) > threshold ({threshold_pct:.0f}%)"
    
    # Build report from clean strings - no copy-paste artifacts
    report = f"""============================================================
VIX 5% Weekly Suite - Thursday Signal Report
Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================

MARKET STATE
----------------------------------------
VIX Close:        ${vix_close:.2f}
52w Percentile:   {percentile:.1f}%
Current Regime:   {regime}

{signal_line}

"""
    
    # Trade details
    if "error" not in quote_data:
        spot = quote_data.get('spot', 0)
        long_leg = quote_data.get('long_leg', 'N/A')
        long_bid = quote_data.get('long_bid', 0)
        long_ask = quote_data.get('long_ask', 0)
        long_mid = quote_data.get('long_mid', 0)
        long_dte = quote_data.get('long_dte', 0)
        short_leg = quote_data.get('short_leg', 'N/A')
        short_bid = quote_data.get('short_bid', 0)
        short_ask = quote_data.get('short_ask', 0)
        short_mid = quote_data.get('short_mid', 0)
        short_dte = quote_data.get('short_dte', 0)
        net_debit_mid = quote_data.get('net_debit_mid', 0)
        net_debit = quote_data.get('net_debit', 0)
        suggested_contracts = quote_data.get('suggested_contracts', 1)
        risk_per_contract = quote_data.get('risk_per_contract', 0)
        total_risk = quote_data.get('total_risk', 0)
        
        report += f"""DIAGONAL SPREAD DETAILS
----------------------------------------
UVXY Spot: ${spot:.2f}

LONG LEG (Buy):
  {long_leg}
  Bid: ${long_bid:.2f}  Ask: ${long_ask:.2f}  Mid: ${long_mid:.2f}
  DTE: {long_dte} days

SHORT LEG (Sell):
  {short_leg}
  Bid: ${short_bid:.2f}  Ask: ${short_ask:.2f}  Mid: ${short_mid:.2f}
  DTE: {short_dte} days

NET POSITION:
  Net Debit (mid):         ${net_debit_mid:.2f}
  Net Debit (conservative): ${net_debit:.2f}

POSITION SIZING:
  Suggested Contracts: {suggested_contracts}
  Risk per Contract:   ${risk_per_contract:.0f}
  Total Position Risk: ${total_risk:.0f}

"""
    else:
        report += f"Quote Error: {quote_data.get('error')}\n\n"
    
    report += """============================================================
Warning: This is a research tool, not financial advice.
Always verify quotes with your broker before trading.
============================================================"""
    
    # Force ASCII - replace any rogue Unicode chars with ?
    report = report.encode('ascii', 'replace').decode('ascii')
    
    return report


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VIX 5% Weekly Thursday Signal Emailer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_signal.py                      # Print report
  python daily_signal.py --email              # Email to DEFAULT_EMAIL
  python daily_signal.py --email you@mail.com # Email to specific address
  python daily_signal.py --png signal.png     # Save PNG locally
  python daily_signal.py --json               # Output JSON

Cron for Thursday 4:30pm ET (20:30 UTC):
  30 20 * * 4 /path/to/venv/bin/python /path/to/daily_signal.py --email
        """
    )
    
    parser.add_argument(
        "--email",
        nargs='?',
        const=DEFAULT_EMAIL,
        default=None,
        help=f"Email address (default: {DEFAULT_EMAIL})"
    )
    
    parser.add_argument(
        "--png",
        type=str,
        default=None,
        help="Save PNG to file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=ENTRY_THRESHOLD,
        help=f"Entry threshold percentile (default: {ENTRY_THRESHOLD})"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output/email if signal is active"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send email even if signal is not active"
    )
    
    args = parser.parse_args()
    
    print(f"📊 VIX 5% Weekly - Thursday Signal Check")
    print(f"   {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get market data
    vix_data = get_vix_percentile()
    
    if "error" in vix_data:
        print(f"❌ Error fetching VIX data: {vix_data['error']}")
        sys.exit(1)
    
    print(f"✓ VIX: ${vix_data['current_price']:.2f} ({vix_data['percentile_pct']:.1f}% percentile)")
    print(f"✓ Regime: {vix_data['regime']}")
    
    # Get diagonal quote
    quote_data = get_uvxy_diagonal_quote(vix_data.get("regime", "MEDIUM"))
    
    if "error" in quote_data:
        print(f"⚠️ Quote warning: {quote_data['error']}")
    else:
        print(f"✓ UVXY: ${quote_data['spot']:.2f}")
    
    # Check if signal is active
    signal_active = vix_data.get("percentile", 1.0) <= args.threshold
    
    print()
    if signal_active:
        print(f"🟢 ENTRY SIGNAL ACTIVE (≤{args.threshold*100:.0f}%)")
    else:
        print(f"🔴 HOLD - No signal (>{args.threshold*100:.0f}%)")
    print()
    
    # Quiet mode
    if args.quiet and not signal_active:
        print("Quiet mode: No signal, exiting.")
        sys.exit(0)
    
    # Generate PNG
    png_buffer = generate_signal_png(vix_data, quote_data, args.threshold)
    
    # Save PNG if requested
    if args.png:
        png_buffer.seek(0)
        with open(args.png, 'wb') as f:
            f.write(png_buffer.read())
        print(f"✓ PNG saved: {args.png}")
    
    # Output
    if args.json:
        output = {
            "vix": vix_data,
            "quote": quote_data,
            "signal_active": signal_active,
            "threshold": args.threshold,
        }
        print(json.dumps(output, indent=2))
    else:
        report = format_text_report(vix_data, quote_data, args.threshold)
        print(report)
    
    # Send email
    if args.email and (signal_active or args.force):
        print()
        print(f"📧 Sending email to {args.email}...")
        png_buffer.seek(0)
        
        if send_signal_email(args.email, vix_data, quote_data, png_buffer, args.threshold):
            print(f"✓ Email sent successfully!")
        else:
            print(f"❌ Failed to send email")
            print("   Check SMTP_USER and SMTP_PASS environment variables")
    elif args.email and not signal_active:
        print()
        print(f"📧 No signal active - email skipped (use --force to override)")
    
    # Exit code
    sys.exit(0 if signal_active else 1)


if __name__ == "__main__":
    main()
