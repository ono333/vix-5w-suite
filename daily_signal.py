#!/usr/bin/env python3
"""
daily_signal.py - VIX 5% Weekly Suite Thursday Signal Generator

PAPER TESTING MODE:
- Always generates and shows ALL 5 variants
- Marks 2-3 as "RECOMMENDED" (would trade live)
- Marks others as "GENERATED" (paper test only)
- NO variant is hidden or suppressed

Run via cron every Thursday at 4:30 PM ET:
    30 16 * * 4 cd /path/to/01_vix_5w_suite && /path/to/python daily_signal.py

Or manually:
    python3 daily_signal.py
    python3 daily_signal.py --dry-run
    python3 daily_signal.py --to other@email.com

Environment variables:
    SMTP_USER - Gmail address
    SMTP_PASS - Gmail app password
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

from enums import VolatilityRegime, VariantRole
from regime_detector import classify_regime, RegimeState
from variant_generator import generate_all_variants, SignalBatch, get_variant_display_name


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RECIPIENT = "onoshin333@gmail.com"
ACCOUNT_SIZE = 250_000
LOOKBACK_WEEKS = 52


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_uvxy_data(lookback_days: int = 400) -> pd.DataFrame:
    """Fetch UVXY historical data."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    print(f"📊 Fetching UVXY data from {start.date()} to {end.date()}...")
    
    df = yf.download("UVXY", start=start, end=end, progress=False)
    
    if df.empty:
        raise RuntimeError("Failed to fetch UVXY data")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"   ✅ Got {len(df)} days of data")
    return df


def calculate_percentile(prices: pd.Series, lookback_weeks: int = 52) -> float:
    """Calculate current price percentile."""
    lookback_days = lookback_weeks * 5
    if len(prices) < lookback_days:
        lookback_days = len(prices)
    
    recent = prices.tail(lookback_days)
    current = prices.iloc[-1]
    return float((recent < current).mean())


def calculate_slope(prices: pd.Series, days: int = 5) -> float:
    """Calculate recent price slope."""
    if len(prices) < days:
        return 0.0
    recent = prices.tail(days).values
    x = np.arange(len(recent))
    return float(np.polyfit(x, recent, 1)[0])


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signal() -> tuple[SignalBatch, RegimeState]:
    """Generate signal batch with ALL 5 variants."""
    
    df = fetch_uvxy_data()
    prices = df["Close"]
    
    current_price = float(prices.iloc[-1])
    percentile = calculate_percentile(prices, LOOKBACK_WEEKS)
    slope = calculate_slope(prices, 5)
    
    print(f"\n📈 Current Market State:")
    print(f"   UVXY: ${current_price:.2f}")
    print(f"   Percentile: {percentile:.1%}")
    print(f"   5-day slope: {slope:+.3f}")
    
    regime = classify_regime(current_price, vix_percentile=percentile)
    
    print(f"\n🎯 Regime Detection:")
    print(f"   Regime: {regime.regime.value.upper()}")
    print(f"   Confidence: {regime.confidence:.0%}")
    
    # Generate ALL 5 variants (no filtering)
    batch = generate_all_variants(regime)
    
    recommended = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    paper_only = len(batch.variants) - recommended
    
    print(f"\n📋 Generated {len(batch.variants)} variants:")
    print(f"   🟢 {recommended} RECOMMENDED (would trade live)")
    print(f"   🔵 {paper_only} PAPER TEST ONLY (observe behavior)")
    
    return batch, regime


# =============================================================================
# Email Generation - PAPER TESTING FORMAT
# =============================================================================

def build_email_html(batch: SignalBatch, regime: RegimeState) -> str:
    """
    Build HTML email for PAPER TESTING mode.
    
    Shows ALL 5 variants with clear distinction:
    - 🟢 RECOMMENDED: Would trade in live mode
    - 🔵 PAPER TEST: Generated for observation only
    """
    
    recommended_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    paper_only_count = len(batch.variants) - recommended_count
    
    # Regime emoji
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:14px;background:#fff;color:#333;padding:20px;max-width:850px;margin:0 auto;">
    
    <!-- Header -->
    <div style="text-align:center;border-bottom:3px solid #1f77b4;padding-bottom:15px;margin-bottom:20px;">
        <span style="font-size:24px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="font-size:16px;color:#666;background:#fff3cd;padding:4px 12px;border-radius:4px;display:inline-block;margin-top:8px;">
            📋 PAPER TESTING MODE
        </span>
    </div>
    
    <!-- Market State -->
    <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;color:#495057;margin-bottom:10px;">📈 Market State</div>
        <table style="width:100%;border-collapse:collapse;">
            <tr>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">Regime</div>
                    <div style="font-size:20px;font-weight:bold;">{regime_emoji} {regime.regime.value.upper()}</div>
                </td>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">UVXY Price</div>
                    <div style="font-size:20px;font-weight:bold;">${regime.vix_level:.2f}</div>
                </td>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">52w Percentile</div>
                    <div style="font-size:20px;font-weight:bold;">{regime.vix_percentile:.0%}</div>
                </td>
                <td style="padding:8px;text-align:center;">
                    <div style="font-size:12px;color:#6c757d;">Confidence</div>
                    <div style="font-size:20px;font-weight:bold;">{regime.confidence:.0%}</div>
                </td>
            </tr>
        </table>
    </div>
    
    <!-- Variant Summary -->
    <div style="display:flex;gap:15px;margin-bottom:20px;">
        <div style="flex:1;background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#155724;">{recommended_count}</div>
            <div style="font-size:12px;color:#155724;">🟢 RECOMMENDED<br>(Live-Ready)</div>
        </div>
        <div style="flex:1;background:#cce5ff;border:1px solid #b8daff;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#004085;">{paper_only_count}</div>
            <div style="font-size:12px;color:#004085;">🔵 PAPER TEST<br>(Observe Only)</div>
        </div>
        <div style="flex:1;background:#e2e3e5;border:1px solid #d6d8db;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#383d41;">5</div>
            <div style="font-size:12px;color:#383d41;">📊 TOTAL<br>(All Generated)</div>
        </div>
    </div>
    
    <!-- Section: RECOMMENDED Variants -->
    <div style="margin-bottom:25px;">
        <div style="font-size:16px;font-weight:bold;color:#155724;background:#d4edda;padding:10px 15px;border-radius:6px 6px 0 0;border:1px solid #c3e6cb;border-bottom:none;">
            🟢 RECOMMENDED VARIANTS (Would Execute in Live Mode)
        </div>
        <div style="border:1px solid #c3e6cb;border-radius:0 0 6px 6px;padding:10px;">
    """
    
    # RECOMMENDED variants first
    for variant in batch.variants:
        is_recommended = regime.regime in variant.active_in_regimes
        if not is_recommended:
            continue
            
        alloc_dollars = ACCOUNT_SIZE * variant.alloc_pct
        est_risk = variant.long_strike_offset * 100
        contracts = max(1, min(50, int(alloc_dollars / est_risk))) if est_risk > 0 else 1
        total_risk = contracts * est_risk
        roll_dte = getattr(variant, 'roll_dte_days', 3)
        
        html += f"""
            <div style="border:2px solid #28a745;margin-bottom:10px;border-radius:6px;overflow:hidden;">
                <div style="background:#28a745;color:#fff;padding:10px 15px;font-weight:bold;font-size:15px;">
                    ✅ {get_variant_display_name(variant.role)}
                </div>
                <div style="padding:12px;background:#f8fff8;">
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr>
                            <td style="padding:5px;width:50%;"><b>Entry:</b> ≤{variant.entry_percentile:.0%} percentile</td>
                            <td style="padding:5px;"><b>Long Strike:</b> UVXY +{variant.long_strike_offset}pts</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Short Strike:</b> UVXY +{variant.short_strike_offset}pts</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Short DTE:</b> {variant.short_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Roll:</b> {roll_dte}d before exp</td>
                        </tr>
                        <tr style="background:#d4edda;">
                            <td style="padding:8px;font-size:14px;"><b>💰 Allocation:</b> {variant.alloc_pct:.1%} (${alloc_dollars:,.0f})</td>
                            <td style="padding:8px;font-size:14px;"><b>📦 Contracts:</b> {contracts}</td>
                        </tr>
                        <tr style="background:#d4edda;">
                            <td style="padding:8px;"><b>🎯 Target:</b> +{variant.tp_pct:.0%}</td>
                            <td style="padding:8px;"><b>🛑 Stop:</b> -{variant.sl_pct:.0%}</td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding:8px;color:#666;font-size:12px;">
                                <b>Max Risk:</b> ${total_risk:,.0f} ({total_risk/ACCOUNT_SIZE:.1%} of ${ACCOUNT_SIZE:,})
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    
    html += """
        </div>
    </div>
    
    <!-- Section: PAPER TEST Variants -->
    <div style="margin-bottom:25px;">
        <div style="font-size:16px;font-weight:bold;color:#004085;background:#cce5ff;padding:10px 15px;border-radius:6px 6px 0 0;border:1px solid #b8daff;border-bottom:none;">
            🔵 PAPER TEST VARIANTS (Observe & Compare — Not Live-Ready)
        </div>
        <div style="border:1px solid #b8daff;border-radius:0 0 6px 6px;padding:10px;">
    """
    
    # PAPER TEST variants
    for variant in batch.variants:
        is_recommended = regime.regime in variant.active_in_regimes
        if is_recommended:
            continue
            
        alloc_dollars = ACCOUNT_SIZE * variant.alloc_pct
        est_risk = variant.long_strike_offset * 100
        contracts = max(1, min(50, int(alloc_dollars / est_risk))) if est_risk > 0 else 1
        total_risk = contracts * est_risk
        roll_dte = getattr(variant, 'roll_dte_days', 3)
        active_regimes = ", ".join([r.value.upper() for r in variant.active_in_regimes])
        
        html += f"""
            <div style="border:2px solid #6c757d;margin-bottom:10px;border-radius:6px;overflow:hidden;">
                <div style="background:#6c757d;color:#fff;padding:10px 15px;font-weight:bold;font-size:15px;">
                    🔬 {get_variant_display_name(variant.role)}
                    <span style="float:right;font-size:12px;font-weight:normal;background:#495057;padding:2px 8px;border-radius:3px;">
                        Paper Test Only
                    </span>
                </div>
                <div style="padding:12px;background:#f8f9fa;">
                    <div style="background:#fff3cd;border:1px solid #ffeeba;border-radius:4px;padding:8px;margin-bottom:10px;font-size:12px;color:#856404;">
                        ⚠️ <b>Why not recommended:</b> This variant activates in <b>{active_regimes}</b> regime(s), not {regime.regime.value.upper()}.
                        <br>Track it to validate this filtering logic.
                    </div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr>
                            <td style="padding:5px;width:50%;"><b>Entry:</b> ≤{variant.entry_percentile:.0%} percentile</td>
                            <td style="padding:5px;"><b>Long Strike:</b> UVXY +{variant.long_strike_offset}pts</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Short Strike:</b> UVXY +{variant.short_strike_offset}pts</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Short DTE:</b> {variant.short_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Roll:</b> {roll_dte}d before exp</td>
                        </tr>
                        <tr style="background:#e9ecef;">
                            <td style="padding:8px;"><b>💰 Allocation:</b> {variant.alloc_pct:.1%} (${alloc_dollars:,.0f})</td>
                            <td style="padding:8px;"><b>📦 Contracts:</b> {contracts}</td>
                        </tr>
                        <tr style="background:#e9ecef;">
                            <td style="padding:8px;"><b>🎯 Target:</b> +{variant.tp_pct:.0%}</td>
                            <td style="padding:8px;"><b>🛑 Stop:</b> -{variant.sl_pct:.0%}</td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding:8px;color:#666;font-size:12px;">
                                <b>Max Risk:</b> ${total_risk:,.0f} ({total_risk/ACCOUNT_SIZE:.1%}) | <b>Activates in:</b> {active_regimes}
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    
    html += f"""
        </div>
    </div>
    
    <!-- Paper Testing Notes -->
    <div style="background:#e7f3ff;border:1px solid #b6d4fe;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;color:#084298;margin-bottom:8px;">📋 Paper Testing Protocol</div>
        <ul style="margin:0;padding-left:20px;color:#084298;font-size:13px;">
            <li><b>Execute ALL 5 variants</b> in paper trading to collect data</li>
            <li><b>RECOMMENDED variants</b> = what live mode would trade</li>
            <li><b>PAPER TEST variants</b> = observe to validate regime filtering</li>
            <li>Track performance to confirm regime logic is correct</li>
            <li>After 8-12 weeks, compare results to decide graduation</li>
        </ul>
    </div>
    
    <!-- Footer -->
    <div style="text-align:center;padding:15px;border-top:1px solid #dee2e6;color:#6c757d;font-size:12px;">
        VIX 5% Weekly Suite — Paper Testing Signal<br>
        Generated: {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | Account Basis: ${ACCOUNT_SIZE:,}
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
    
    recommended = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    paper_only = len(batch.variants) - recommended
    
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    subject = f"{regime_emoji} [PAPER TEST] {regime.regime.value.upper()} ({regime.vix_percentile:.0%}) — {recommended} Recommended / {paper_only} Observe"
    
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VIX 5% Weekly Suite - Paper Testing Signal Generator")
    parser.add_argument("--dry-run", action="store_true", help="Generate without sending")
    parser.add_argument("--to", type=str, default=DEFAULT_RECIPIENT, help="Email recipient")
    parser.add_argument("--save-html", type=str, help="Save HTML to file")
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("🚀 VIX 5% Weekly Suite - PAPER TESTING Signal Generator")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    try:
        batch, regime = generate_signal()
        html = build_email_html(batch, regime)
        
        if args.save_html:
            Path(args.save_html).write_text(html)
            print(f"\n✅ HTML saved to {args.save_html}")
            
        elif args.dry_run:
            print("\n" + "=" * 65)
            print("🔍 DRY RUN — Email Preview")
            print("=" * 65)
            
            recommended = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
            print(f"\n   To: {args.to}")
            print(f"   Regime: {regime.regime.value.upper()} ({regime.vix_percentile:.0%})")
            print(f"\n   🟢 RECOMMENDED ({recommended}):")
            for v in batch.variants:
                if regime.regime in v.active_in_regimes:
                    print(f"      ✅ {get_variant_display_name(v.role)}")
                    print(f"         Entry ≤{v.entry_percentile:.0%} | +{v.long_strike_offset}pts | {v.long_dte_weeks}w")
                    print(f"         Alloc: {v.alloc_pct:.1%} | TP: +{v.tp_pct:.0%} | SL: -{v.sl_pct:.0%}")
            
            print(f"\n   🔵 PAPER TEST ({len(batch.variants) - recommended}):")
            for v in batch.variants:
                if regime.regime not in v.active_in_regimes:
                    active_in = ", ".join([r.value.upper() for r in v.active_in_regimes])
                    print(f"      🔬 {get_variant_display_name(v.role)}")
                    print(f"         Entry ≤{v.entry_percentile:.0%} | +{v.long_strike_offset}pts | {v.long_dte_weeks}w")
                    print(f"         Activates in: {active_in}")
            
        else:
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
