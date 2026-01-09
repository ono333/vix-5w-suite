#!/usr/bin/env python3
"""
daily_signal.py - VIX 5% Weekly Suite Thursday Signal Generator

Run via cron every Thursday at 4:30 PM ET:
    30 16 * * 4 cd /path/to/01_vix_5w_suite && /path/to/python daily_signal.py

Or manually:
    python3 daily_signal.py
    python3 daily_signal.py --dry-run        # Preview without sending
    python3 daily_signal.py --to other@email.com

Environment variables required:
    SMTP_USER - Gmail address
    SMTP_PASS - Gmail app password (https://myaccount.google.com/apppasswords)
"""

import os
import sys
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import local modules
from enums import VolatilityRegime, VariantRole
from regime_detector import classify_regime, RegimeState
from variant_generator import generate_all_variants, SignalBatch, get_variant_display_name


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RECIPIENT = "onoshin333@gmail.com"
ACCOUNT_SIZE = 250_000  # For position sizing calculations
LOOKBACK_WEEKS = 52     # For percentile calculation


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_uvxy_data(lookback_days: int = 400) -> pd.DataFrame:
    """Fetch UVXY historical data for regime detection."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    print(f"📊 Fetching UVXY data from {start.date()} to {end.date()}...")
    
    df = yf.download("UVXY", start=start, end=end, progress=False)
    
    if df.empty:
        raise RuntimeError("Failed to fetch UVXY data from Yahoo Finance")
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"   ✅ Got {len(df)} days of data")
    return df


def calculate_percentile(prices: pd.Series, lookback_weeks: int = 52) -> float:
    """Calculate current price percentile over lookback period."""
    lookback_days = lookback_weeks * 5  # ~trading days
    
    if len(prices) < lookback_days:
        lookback_days = len(prices)
    
    recent = prices.tail(lookback_days)
    current = prices.iloc[-1]
    
    percentile = (recent < current).mean()
    return float(percentile)


def calculate_slope(prices: pd.Series, days: int = 5) -> float:
    """Calculate recent price slope (trend direction)."""
    if len(prices) < days:
        return 0.0
    
    recent = prices.tail(days).values
    x = np.arange(len(recent))
    
    # Simple linear regression slope
    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signal() -> tuple[SignalBatch, RegimeState]:
    """Generate signal batch and regime state from current market data."""
    
    # Fetch data
    df = fetch_uvxy_data()
    prices = df["Close"]
    
    current_price = float(prices.iloc[-1])
    percentile = calculate_percentile(prices, LOOKBACK_WEEKS)
    slope = calculate_slope(prices, 5)
    
    print(f"\n📈 Current Market State:")
    print(f"   UVXY: ${current_price:.2f}")
    print(f"   Percentile: {percentile:.1%}")
    print(f"   5-day slope: {slope:+.3f}")
    
    # Detect regime - correct signature: classify_regime(data, vix_percentile, lookback)
    regime = classify_regime(current_price, vix_percentile=percentile)
    
    print(f"\n🎯 Regime Detection:")
    print(f"   Regime: {regime.regime.value.upper()}")
    print(f"   Confidence: {regime.confidence:.0%}")
    
    # Generate all 5 variants (no filtering!)
    batch = generate_all_variants(regime)
    
    active_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    
    print(f"\n📋 Generated {len(batch.variants)} variants ({active_count} active for {regime.regime.value.upper()})")
    
    return batch, regime


# =============================================================================
# Email Generation
# =============================================================================

def build_email_html(batch: SignalBatch, regime: RegimeState) -> str:
    """Build HTML email showing ALL 5 variants for paper testing."""
    
    # Count active vs suppressed
    active_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    suppressed_count = len(batch.variants) - active_count
    
    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:14px;background:#fff;color:#333;padding:15px;max-width:800px;">
    
    <div style="text-align:center;border-bottom:2px solid #1f77b4;padding-bottom:10px;margin-bottom:15px;">
        <span style="font-size:22px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="font-size:14px;color:#666;">Paper Testing Signal • {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</span>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin-bottom:15px;">
        <tr>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Regime</b><br><span style="font-size:18px;">{regime.regime.value.upper()}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>UVXY</b><br><span style="font-size:18px;">${regime.vix_level:.2f}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Percentile</b><br><span style="font-size:18px;">{regime.vix_percentile:.0%}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Confidence</b><br><span style="font-size:18px;">{regime.confidence:.0%}</span>
            </td>
        </tr>
    </table>
    
    <div style="font-size:16px;font-weight:bold;color:#1f77b4;margin-bottom:5px;">
        📊 ALL VARIANTS (Paper Testing Mode)
    </div>
    <div style="font-size:12px;color:#666;margin-bottom:15px;">
        {active_count} active for {regime.regime.value.upper()} regime | {suppressed_count} suppressed (shown for comparison)
    </div>
    """
    
    # Sort: active first, then by role
    sorted_variants = sorted(
        batch.variants, 
        key=lambda v: (regime.regime not in v.active_in_regimes, v.role.value)
    )
    
    for variant in sorted_variants:
        is_active = regime.regime in variant.active_in_regimes
        
        # Calculate position sizing
        alloc_dollars = ACCOUNT_SIZE * variant.alloc_pct
        est_risk_per_contract = variant.long_strike_offset * 100
        if est_risk_per_contract > 0:
            suggested_contracts = max(1, min(50, int(alloc_dollars / est_risk_per_contract)))
        else:
            suggested_contracts = 1
        total_risk = suggested_contracts * est_risk_per_contract
        
        if is_active:
            header_bg = "#4CAF50"
            header_text = "#fff"
            status_icon = "✅"
            status_text = "ACTIVE"
            body_bg = "#f8fff8"
            border_color = "#4CAF50"
        else:
            header_bg = "#9e9e9e"
            header_text = "#fff"
            status_icon = "⏸️"
            status_text = f"SUPPRESSED (not for {regime.regime.value.upper()})"
            body_bg = "#f5f5f5"
            border_color = "#ccc"
        
        # Get active regimes for this variant
        active_regimes = ", ".join([r.value.upper() for r in variant.active_in_regimes])
        
        # Get roll_dte_days with fallback
        roll_dte = getattr(variant, 'roll_dte_days', 3)
        
        html += f"""
        <div style="border:2px solid {border_color};margin-bottom:12px;border-radius:6px;overflow:hidden;">
            <div style="background:{header_bg};color:{header_text};padding:8px 12px;font-weight:bold;">
                <span>{status_icon} {get_variant_display_name(variant.role)}</span>
                <span style="float:right;font-size:12px;font-weight:normal;">{status_text}</span>
            </div>
            <div style="padding:12px;background:{body_bg};">
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <tr>
                        <td style="padding:4px;width:50%;"><b>Entry Threshold:</b> ≤{variant.entry_percentile:.0%} percentile</td>
                        <td style="padding:4px;width:50%;"><b>Long Strike:</b> UVXY +{variant.long_strike_offset}pts OTM</td>
                    </tr>
                    <tr>
                        <td style="padding:4px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                        <td style="padding:4px;"><b>Short Strike:</b> UVXY +{variant.short_strike_offset}pts OTM</td>
                    </tr>
                    <tr>
                        <td style="padding:4px;"><b>Short DTE:</b> {variant.short_dte_weeks}w</td>
                        <td style="padding:4px;"><b>Roll Threshold:</b> {roll_dte}d</td>
                    </tr>
                    <tr style="background:{'#e8f5e9' if is_active else '#eeeeee'};">
                        <td style="padding:6px;"><b>Allocation:</b> {variant.alloc_pct:.1%} (${alloc_dollars:,.0f})</td>
                        <td style="padding:6px;"><b>Suggested:</b> {suggested_contracts} contracts</td>
                    </tr>
                    <tr style="background:{'#e8f5e9' if is_active else '#eeeeee'};">
                        <td style="padding:6px;"><b>Target:</b> +{variant.tp_pct:.0%}</td>
                        <td style="padding:6px;"><b>Stop:</b> -{variant.sl_pct:.0%}</td>
                    </tr>
                    <tr>
                        <td colspan="2" style="padding:6px;color:#666;font-size:12px;">
                            <b>Est. Max Risk:</b> ${total_risk:,.0f} ({total_risk/ACCOUNT_SIZE:.1%}) | 
                            <b>Active in:</b> {active_regimes}
                        </td>
                    </tr>
                </table>
            </div>
        </div>
        """
    
    html += f"""
    <div style="margin-top:20px;padding:12px;background:#e3f2fd;border-radius:4px;font-size:12px;">
        <b>📋 Paper Testing Notes:</b><br>
        • All 5 variants shown for observation and comparison<br>
        • Suppressed variants would not trade in live mode for this regime<br>
        • Track all variants to validate regime-based filtering logic<br>
        • Account basis: ${ACCOUNT_SIZE:,}
    </div>
    
    <div style="margin-top:15px;padding-top:10px;border-top:1px solid #ddd;color:#888;font-size:11px;text-align:center;">
        VIX 5% Weekly Suite - Paper Testing Signal | {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}
    </div>
    </body>
    </html>
    """
    
    return html


def send_email(batch: SignalBatch, regime: RegimeState, recipient: str) -> tuple[bool, str]:
    """Send signal email via SMTP."""
    
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials not configured. Set SMTP_USER and SMTP_PASS."
    
    # Count active
    active_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    suppressed_count = len(batch.variants) - active_count
    
    # Emoji based on regime
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    subject = f"{regime_emoji} VIX Signal: {regime.regime.value.upper()} ({regime.vix_percentile:.0%}) - {active_count} Active / {suppressed_count} Suppressed"
    
    html = build_email_html(batch, regime)
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))
    
    try:
        print(f"\n📧 Sending email to {recipient}...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        return True, f"Email sent to {recipient}"
    except Exception as e:
        return False, f"Email failed: {str(e)}"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VIX 5% Weekly Suite - Thursday Signal Generator"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Generate signal but don't send email"
    )
    parser.add_argument(
        "--to",
        type=str,
        default=DEFAULT_RECIPIENT,
        help=f"Email recipient (default: {DEFAULT_RECIPIENT})"
    )
    parser.add_argument(
        "--save-html",
        type=str,
        help="Save HTML to file instead of sending"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 VIX 5% Weekly Suite - Signal Generator")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Generate signal
        batch, regime = generate_signal()
        
        # Build email HTML
        html = build_email_html(batch, regime)
        
        if args.save_html:
            # Save to file
            Path(args.save_html).write_text(html)
            print(f"\n✅ HTML saved to {args.save_html}")
            
        elif args.dry_run:
            # Just show summary
            print("\n" + "=" * 60)
            print("🔍 DRY RUN - Email would contain:")
            print("=" * 60)
            active_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
            print(f"   Subject: VIX Signal: {regime.regime.value.upper()} ({regime.vix_percentile:.0%})")
            print(f"   To: {args.to}")
            print(f"   Active variants: {active_count}")
            print(f"   Suppressed variants: {len(batch.variants) - active_count}")
            print(f"\n   All {len(batch.variants)} variants:")
            for v in batch.variants:
                is_active = regime.regime in v.active_in_regimes
                status = "✅ ACTIVE" if is_active else "⏸️ SUPPRESSED"
                active_in = ", ".join([r.value.upper() for r in v.active_in_regimes])
                print(f"      {get_variant_display_name(v.role)}: {status}")
                print(f"         Entry ≤{v.entry_percentile:.0%} | +{v.long_strike_offset}pts OTM | {v.long_dte_weeks}w DTE")
                print(f"         Alloc: {v.alloc_pct:.1%} | TP: +{v.tp_pct:.0%} | SL: -{v.sl_pct:.0%}")
                print(f"         Active in: {active_in}")
            
        else:
            # Send email
            success, message = send_email(batch, regime, args.to)
            if success:
                print(f"\n✅ {message}")
            else:
                print(f"\n❌ {message}")
                sys.exit(1)
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
