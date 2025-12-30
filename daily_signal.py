#!/usr/bin/env python3
"""
Daily Signal Check for VIX 5% Weekly Suite

This script checks current market conditions and generates signals.
Can be run via cron for automated daily notifications.

Run daily (e.g., 4:30 PM after market close):
    30 16 * * 1-5 /usr/bin/python3 /path/to/daily_signal.py >> /path/to/signal_log.txt 2>&1

Or manually:
    python daily_signal.py
    python daily_signal.py --email your@email.com
    python daily_signal.py --json  # Output as JSON

Features:
- Checks current VIX percentile and regime
- Generates diagonal trade signal if entry conditions met
- Optional email notification
- JSON output for integration with other systems
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.param_history import get_best_for_regime, apply_regime_params


def get_vix_percentile(lookback_weeks: int = 52) -> Dict[str, Any]:
    """
    Get current VIX price and percentile.
    
    Returns
    -------
    dict with:
        - current_price: float
        - percentile: float (0-1)
        - regime: str
        - prices_52w: list
    """
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
        
        # Map to regime
        if pct <= 0.10:
            regime = "ULTRA_LOW"
        elif pct <= 0.25:
            regime = "LOW"
        elif pct <= 0.50:
            regime = "MEDIUM"
        elif pct <= 0.75:
            regime = "HIGH"
        else:
            regime = "EXTREME"
        
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
    """
    Get current UVXY diagonal spread quotes.
    
    Parameters
    ----------
    regime : str
        Current regime for parameter selection
    params : dict, optional
        Base parameters (defaults used if None)
        
    Returns
    -------
    dict with trade details or error
    """
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
        
        return {
            "spot": round(spot, 2),
            "regime": regime,
            
            "long_exp": long_exp.strftime("%Y-%m-%d"),
            "long_dte": (long_exp - today).days,
            "long_strike": long_strike,
            "long_bid": round(long_bid, 2),
            "long_ask": round(long_ask, 2),
            "long_mid": round(long_mid, 2),
            
            "short_exp": short_exp.strftime("%Y-%m-%d"),
            "short_dte": (short_exp - today).days,
            "short_strike": short_strike,
            "short_bid": round(short_bid, 2),
            "short_ask": round(short_ask, 2),
            "short_mid": round(short_mid, 2),
            
            "net_debit_conservative": round(net_debit, 2),
            "net_debit_mid": round(long_mid - short_mid, 2),
            
            "otm_pts_used": otm_pts,
            "timestamp": dt.datetime.now().isoformat(),
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def send_email_notification(
    subject: str,
    body: str,
    to_email: str,
    smtp_server: str = None,
    smtp_port: int = 587,
    smtp_user: str = None,
    smtp_pass: str = None,
) -> bool:
    """
    Send email notification.
    
    Reads SMTP credentials from environment if not provided:
        SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS
        
    Returns True if sent successfully.
    """
    import smtplib
    from email.mime.text import MIMEText
    
    try:
        server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        port = smtp_port or int(os.environ.get("SMTP_PORT", 587))
        user = smtp_user or os.environ.get("SMTP_USER")
        password = smtp_pass or os.environ.get("SMTP_PASS")
        
        if not user or not password:
            print("Warning: SMTP credentials not configured")
            return False
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_email
        
        with smtplib.SMTP(server, port) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        
        return True
        
    except Exception as e:
        print(f"Email error: {e}")
        return False


def format_signal_report(vix_data: Dict, quote_data: Dict, entry_threshold: float = 0.25) -> str:
    """Format signal data as human-readable report."""
    
    lines = [
        "=" * 60,
        "VIX 5% Weekly Suite - Daily Signal Report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "MARKET STATE",
        "-" * 40,
        f"VIX Close:        {vix_data.get('current_price', 'N/A'):.2f}" if 'current_price' in vix_data else "VIX Close: N/A",
        f"52w Percentile:   {vix_data.get('percentile_pct', 0):.1f}%" if 'percentile_pct' in vix_data else "52w Percentile: N/A",
        f"Current Regime:   {vix_data.get('regime', 'N/A')}",
        "",
    ]
    
    # Signal determination
    pct = vix_data.get('percentile', 1.0)
    if pct <= entry_threshold:
        lines.append("🟢 ENTRY SIGNAL ACTIVE")
        lines.append(f"   Percentile ({pct*100:.1f}%) <= threshold ({entry_threshold*100:.0f}%)")
    else:
        lines.append("🔕 HOLD - No entry signal")
        lines.append(f"   Percentile ({pct*100:.1f}%) > threshold ({entry_threshold*100:.0f}%)")
    
    lines.append("")
    
    # Trade details if available
    if "error" not in quote_data:
        lines.extend([
            "DIAGONAL SPREAD DETAILS",
            "-" * 40,
            f"UVXY Spot: ${quote_data.get('spot', 0):.2f}",
            "",
            "LONG LEG (Buy):",
            f"  UVXY {quote_data.get('long_exp')} ${quote_data.get('long_strike')} Call",
            f"  Bid: ${quote_data.get('long_bid', 0):.2f}  Ask: ${quote_data.get('long_ask', 0):.2f}  Mid: ${quote_data.get('long_mid', 0):.2f}",
            f"  DTE: {quote_data.get('long_dte')} days",
            "",
            "SHORT LEG (Sell):",
            f"  UVXY {quote_data.get('short_exp')} ${quote_data.get('short_strike')} Call",
            f"  Bid: ${quote_data.get('short_bid', 0):.2f}  Ask: ${quote_data.get('short_ask', 0):.2f}  Mid: ${quote_data.get('short_mid', 0):.2f}",
            f"  DTE: {quote_data.get('short_dte')} days",
            "",
            "NET POSITION:",
            f"  Net Debit (mid):         ${quote_data.get('net_debit_mid', 0):.2f}",
            f"  Net Debit (conservative): ${quote_data.get('net_debit_conservative', 0):.2f}",
        ])
    else:
        lines.append(f"Quote Error: {quote_data.get('error')}")
    
    lines.extend([
        "",
        "=" * 60,
        "⚠️ This is a research tool, not financial advice.",
        "Always verify quotes with your broker before trading.",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Daily VIX signal check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email address for notification"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted report"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Entry threshold percentile (default: 0.25 = 25%%)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output if signal is active"
    )
    
    args = parser.parse_args()
    
    # Get market data
    vix_data = get_vix_percentile()
    
    if "error" in vix_data:
        print(f"Error fetching VIX data: {vix_data['error']}")
        sys.exit(1)
    
    # Get diagonal quote
    quote_data = get_uvxy_diagonal_quote(vix_data.get("regime", "MEDIUM"))
    
    # Check if signal is active
    signal_active = vix_data.get("percentile", 1.0) <= args.threshold
    
    # Quiet mode - only output if signal active
    if args.quiet and not signal_active:
        sys.exit(0)
    
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
        report = format_signal_report(vix_data, quote_data, args.threshold)
        print(report)
    
    # Send email if requested and signal is active
    if args.email and signal_active:
        subject = f"🟢 VIX Entry Signal - {vix_data.get('regime')} Regime ({vix_data.get('percentile_pct', 0):.1f}%)"
        body = format_signal_report(vix_data, quote_data, args.threshold)
        
        if send_email_notification(subject, body, args.email):
            print(f"\n✓ Email sent to {args.email}")
        else:
            print(f"\n⚠️ Failed to send email (check SMTP settings)")
    
    # Return exit code based on signal
    sys.exit(0 if signal_active else 1)


if __name__ == "__main__":
    main()
