#!/usr/bin/env python3
"""
VIX 5% Weekly Suite - Position-Aware Signal Generator

This script generates weekly trading signals that are POSITION-AWARE:
- Reads from trade log to detect open positions
- Shows MANAGEMENT mode for variants with positions (P&L, DTE, exits)
- Shows ENTRY mode for variants without positions
- Computes and displays target/stop prices

Run: python3 daily_signal.py [--dry-run] [--to EMAIL]

Cron setup for Thursday 4:30 PM ET:
30 16 * * 4 cd /path/to/01_vix_5w_suite && /path/to/python daily_signal.py
"""

import os
import sys
import argparse
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enums import VolatilityRegime, VariantRole
from regime_detector import classify_regime, RegimeState
from variant_generator import generate_all_variants, get_variant_display_name, SignalBatch, VariantParams
from trade_log import get_trade_log, TradeLog, Position


# ============================================================
# Market Data
# ============================================================

def fetch_uvxy_data(lookback_days: int = 365) -> Tuple[float, float, float]:
    """
    Fetch UVXY data and compute current price, percentile, and slope.
    
    Returns: (current_price, percentile, slope_5d)
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    print(f"📊 Fetching UVXY data from {start.date()} to {end.date()}...")
    
    df = yf.download("UVXY", start=start, end=end, progress=False)
    
    if df.empty:
        raise ValueError("No UVXY data returned from Yahoo Finance")
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    prices = df[close_col].dropna()
    
    print(f"   ✅ Got {len(prices)} days of data")
    
    current_price = float(prices.iloc[-1])
    
    # 52-week percentile
    window = min(252, len(prices))
    rolling_min = prices.rolling(window=window).min()
    rolling_max = prices.rolling(window=window).max()
    
    percentile = (current_price - rolling_min.iloc[-1]) / (rolling_max.iloc[-1] - rolling_min.iloc[-1] + 1e-10)
    percentile = max(0, min(1, percentile))
    
    # 5-day slope (simple linear regression approximation)
    if len(prices) >= 5:
        recent = prices.iloc[-5:].values
        slope = (recent[-1] - recent[0]) / recent[0]
    else:
        slope = 0.0
    
    return current_price, percentile, slope


# ============================================================
# Position-Aware Classification
# ============================================================

@dataclass
class VariantState:
    """State of a variant including position status."""
    variant: VariantParams
    has_position: bool
    position: Optional[Position]
    is_recommended: bool  # Would trade in live mode
    
    # Entry mode fields (when no position)
    suggested_entry_credit: Optional[float] = None
    suggested_target_price: Optional[float] = None
    suggested_stop_price: Optional[float] = None
    
    # Management mode fields (when position exists)
    current_pnl: Optional[float] = None
    current_pnl_pct: Optional[float] = None
    dte_remaining: Optional[int] = None
    action_suggestion: Optional[str] = None  # hold, take_profit, stop_loss, roll, close


def classify_variants(
    batch: SignalBatch,
    trade_log: TradeLog,
    current_regime: VolatilityRegime,
) -> List[VariantState]:
    """
    Classify each variant into management or entry mode based on position state.
    """
    states = []
    
    for variant in batch.variants:
        variant_id = variant.role.value if hasattr(variant.role, 'value') else str(variant.role)
        
        # Check for open position
        position = trade_log.get_open_position(variant_id)
        has_position = position is not None
        
        # Is this variant recommended for current regime?
        is_recommended = current_regime in variant.active_in_regimes
        
        state = VariantState(
            variant=variant,
            has_position=has_position,
            position=position,
            is_recommended=is_recommended,
        )
        
        if has_position and position:
            # MANAGEMENT MODE
            state.current_pnl = position.current_pnl
            state.current_pnl_pct = position.current_pnl_pct
            state.dte_remaining = position.days_to_expiry()
            
            # Suggest action based on current state
            if position.current_pnl_pct >= variant.target_pct:
                state.action_suggestion = "🎯 TAKE PROFIT - Target reached"
            elif position.current_pnl_pct <= -variant.stop_pct:
                state.action_suggestion = "🛑 STOP LOSS - Stop level hit"
            elif state.dte_remaining <= 5:
                state.action_suggestion = "📅 ROLL or CLOSE - Low DTE"
            elif not is_recommended:
                state.action_suggestion = "⚠️ REGIME DRIFT - Consider closing"
            else:
                state.action_suggestion = "✋ HOLD - On track"
        else:
            # ENTRY MODE
            # Estimate entry credit (this would come from real option chains in production)
            # For now, use allocation as basis for suggested entry
            vix_level = batch.regime_state.vix_level if batch.regime_state else 20.0
            
            # Rough estimate: entry credit scales with VIX level
            state.suggested_entry_credit = round(1.0 + (vix_level - 15) * 0.1, 2)
            
            # Compute targets from estimated entry
            if state.suggested_entry_credit > 0:
                state.suggested_target_price = round(
                    state.suggested_entry_credit * (1 - variant.target_pct), 2
                )
                state.suggested_stop_price = round(
                    state.suggested_entry_credit * (1 + variant.stop_pct), 2
                )
        
        states.append(state)
    
    return states


# ============================================================
# Email Generation - POSITION-AWARE FORMAT
# ============================================================

def build_position_aware_email(
    batch: SignalBatch,
    variant_states: List[VariantState],
    account_size: float = 250_000.0,
) -> str:
    """
    Build HTML email that reflects trading state, not just signals.
    
    Two main sections:
    - OPEN POSITIONS (management mode)
    - ENTRY CANDIDATES (entry mode)
    """
    regime_state = batch.regime_state
    regime_name = regime_state.regime.value.upper() if regime_state else "UNKNOWN"
    vix_level = regime_state.vix_level if regime_state else 0
    vix_pct = regime_state.vix_percentile if regime_state else 0
    
    # Separate into management vs entry
    management_variants = [s for s in variant_states if s.has_position]
    entry_variants = [s for s in variant_states if not s.has_position]
    
    # Further split entry variants
    recommended_entries = [s for s in entry_variants if s.is_recommended]
    paper_test_entries = [s for s in entry_variants if not s.is_recommended]
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        .container {{ max-width: 700px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header .subtitle {{ opacity: 0.9; margin-top: 8px; }}
        .stats {{ display: flex; gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #252542; padding: 15px 20px; border-radius: 8px; text-align: center; flex: 1; }}
        .stat-box .value {{ font-size: 28px; font-weight: bold; }}
        .stat-box .label {{ font-size: 11px; opacity: 0.7; margin-top: 4px; }}
        .stat-green {{ border-left: 4px solid #4CAF50; }}
        .stat-blue {{ border-left: 4px solid #2196F3; }}
        .stat-orange {{ border-left: 4px solid #FF9800; }}
        .section {{ background: #252542; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        .section-header {{ font-size: 16px; font-weight: 600; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #444; }}
        .position-card {{ background: #1a1a2e; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .position-card.profit {{ border-left: 4px solid #4CAF50; }}
        .position-card.loss {{ border-left: 4px solid #f44336; }}
        .position-card.neutral {{ border-left: 4px solid #2196F3; }}
        .variant-name {{ font-weight: 600; font-size: 15px; margin-bottom: 8px; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 15px; font-size: 13px; }}
        .metric {{ }}
        .metric-label {{ opacity: 0.6; font-size: 11px; }}
        .metric-value {{ font-weight: 500; }}
        .action {{ margin-top: 12px; padding: 8px 12px; background: #333; border-radius: 6px; font-size: 13px; }}
        .entry-card {{ background: #1a1a2e; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .entry-card.recommended {{ border-left: 4px solid #4CAF50; }}
        .entry-card.paper-test {{ border-left: 4px solid #9E9E9E; opacity: 0.8; }}
        .targets {{ display: flex; gap: 20px; margin-top: 10px; }}
        .target {{ padding: 8px 12px; background: #333; border-radius: 6px; }}
        .target-label {{ font-size: 11px; opacity: 0.6; }}
        .target-value {{ font-weight: 600; }}
        .target.profit {{ color: #4CAF50; }}
        .target.stop {{ color: #f44336; }}
        .allocation {{ margin-top: 10px; font-size: 12px; opacity: 0.8; }}
        .regime-warning {{ background: #442; border-left: 4px solid #FF9800; padding: 12px; margin: 10px 0; border-radius: 6px; font-size: 13px; }}
        .footer {{ text-align: center; opacity: 0.5; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    
    <div class="header">
        <h1>📈 VIX 5% Weekly — Position Report</h1>
        <div class="subtitle">
            {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')} ET
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-box stat-orange">
            <div class="value">{regime_name}</div>
            <div class="label">Current Regime</div>
        </div>
        <div class="stat-box">
            <div class="value">${vix_level:.2f}</div>
            <div class="label">UVXY Level</div>
        </div>
        <div class="stat-box">
            <div class="value">{vix_pct:.0%}</div>
            <div class="label">52w Percentile</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-box stat-green">
            <div class="value">{len(management_variants)}</div>
            <div class="label">🔄 Open Positions</div>
        </div>
        <div class="stat-box stat-blue">
            <div class="value">{len(recommended_entries)}</div>
            <div class="label">🎯 Entry Candidates</div>
        </div>
        <div class="stat-box">
            <div class="value">{len(paper_test_entries)}</div>
            <div class="label">🔬 Paper Test Only</div>
        </div>
    </div>
"""
    
    # ================================================================
    # SECTION 1: OPEN POSITIONS (Management Mode)
    # ================================================================
    if management_variants:
        html += """
    <div class="section">
        <div class="section-header">🔄 OPEN POSITIONS — Management Mode</div>
"""
        for state in management_variants:
            pos = state.position
            variant = state.variant
            name = get_variant_display_name(variant.role)
            
            # Determine card class based on P&L
            if state.current_pnl_pct and state.current_pnl_pct > 0.05:
                card_class = "profit"
            elif state.current_pnl_pct and state.current_pnl_pct < -0.05:
                card_class = "loss"
            else:
                card_class = "neutral"
            
            pnl_dollars = state.current_pnl or 0
            pnl_pct = (state.current_pnl_pct or 0) * 100
            dte = state.dte_remaining or 0
            
            html += f"""
        <div class="position-card {card_class}">
            <div class="variant-name">{name}</div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Current P&L</div>
                    <div class="metric-value">${pnl_dollars:+,.0f} ({pnl_pct:+.1f}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-label">DTE Remaining</div>
                    <div class="metric-value">{dte} days</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Entry Price</div>
                    <div class="metric-value">${pos.entry_price:.2f}</div>
                </div>
            </div>
            <div class="targets">
                <div class="target profit">
                    <div class="target-label">Target Exit</div>
                    <div class="target-value">${pos.target_price:.2f}</div>
                </div>
                <div class="target stop">
                    <div class="target-label">Stop Loss</div>
                    <div class="target-value">${pos.stop_price:.2f}</div>
                </div>
            </div>
            <div class="action">{state.action_suggestion}</div>
        </div>
"""
        html += "    </div>\n"
    
    # ================================================================
    # SECTION 2: ENTRY CANDIDATES (Recommended)
    # ================================================================
    if recommended_entries:
        html += """
    <div class="section">
        <div class="section-header">🎯 ENTRY CANDIDATES — Would Execute in Live Mode</div>
"""
        for state in recommended_entries:
            variant = state.variant
            name = get_variant_display_name(variant.role)
            
            alloc_dollars = account_size * (variant.allocation_pct / 100)
            contracts = max(1, min(50, int(alloc_dollars / 500)))  # Rough estimate
            
            html += f"""
        <div class="entry-card recommended">
            <div class="variant-name">✅ {name}</div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Entry Trigger</div>
                    <div class="metric-value">≤{variant.entry_percentile:.0%} percentile</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Strike Offset</div>
                    <div class="metric-value">+{variant.long_strike_offset:.0f} pts OTM</div>
                </div>
                <div class="metric">
                    <div class="metric-label">DTE</div>
                    <div class="metric-value">{variant.long_dte_weeks}w</div>
                </div>
            </div>
            <div class="targets">
                <div class="target">
                    <div class="target-label">Suggested Entry Credit</div>
                    <div class="target-value">≥${state.suggested_entry_credit:.2f}</div>
                </div>
                <div class="target profit">
                    <div class="target-label">Target Exit ({variant.target_pct:.0%} gain)</div>
                    <div class="target-value">${state.suggested_target_price:.2f}</div>
                </div>
                <div class="target stop">
                    <div class="target-label">Stop Loss ({variant.stop_pct:.0%} loss)</div>
                    <div class="target-value">${state.suggested_stop_price:.2f}</div>
                </div>
            </div>
            <div class="allocation">
                💰 Allocation: {variant.allocation_pct:.1f}% (${alloc_dollars:,.0f}) → ~{contracts} contracts
            </div>
        </div>
"""
        html += "    </div>\n"
    
    # ================================================================
    # SECTION 3: PAPER TEST VARIANTS (Not Recommended, but track)
    # ================================================================
    if paper_test_entries:
        html += """
    <div class="section" style="opacity: 0.75;">
        <div class="section-header">🔬 PAPER TEST ONLY — Not Recommended for {regime_name}</div>
""".format(regime_name=regime_name)
        
        for state in paper_test_entries:
            variant = state.variant
            name = get_variant_display_name(variant.role)
            active_in = ", ".join([r.value.upper() for r in variant.active_in_regimes])
            
            html += f"""
        <div class="entry-card paper-test">
            <div class="variant-name">🔬 {name}</div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Entry</div>
                    <div class="metric-value">≤{variant.entry_percentile:.0%} | +{variant.long_strike_offset:.0f}pts | {variant.long_dte_weeks}w</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Activates In</div>
                    <div class="metric-value">{active_in}</div>
                </div>
            </div>
            <div class="regime-warning">
                ⚠️ Not recommended for {regime_name} regime. Track in paper trading for validation.
            </div>
        </div>
"""
        html += "    </div>\n"
    
    # ================================================================
    # REGIME WARNING (if any positions have regime drift)
    # ================================================================
    regime_drift = [s for s in management_variants if not s.is_recommended]
    if regime_drift:
        html += """
    <div class="section" style="background: #442;">
        <div class="section-header">⚠️ REGIME DRIFT WARNING</div>
        <p style="font-size: 14px; margin: 0;">
            The following positions were opened in a different regime and may need attention:
        </p>
        <ul style="margin: 10px 0;">
"""
        for state in regime_drift:
            name = get_variant_display_name(state.variant.role)
            entry_regime = state.position.entry_regime if state.position else "unknown"
            html += f"            <li>{name} — Opened in {entry_regime}, now in {regime_name}</li>\n"
        html += """        </ul>
    </div>
"""
    
    # Footer
    html += f"""
    <div class="footer">
        VIX 5% Weekly Suite — Paper Trading Mode<br>
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET<br>
        Email is a VIEW of trading state, not a signal generator.
    </div>
    
</div>
</body>
</html>
"""
    
    return html


# ============================================================
# Email Sending
# ============================================================

def send_email(
    html_body: str,
    to_email: str,
    subject: str,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
) -> bool:
    """Send HTML email via SMTP (Gmail)."""
    smtp_user = smtp_user or os.environ.get("SMTP_USER", "")
    smtp_pass = smtp_pass or os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        print("❌ SMTP credentials not set. Export SMTP_USER and SMTP_PASS.")
        return False
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    
    msg.attach(MIMEText(html_body, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Position-Aware VIX Signal Generator")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    parser.add_argument("--to", type=str, default="onoshin333@gmail.com", help="Recipient email")
    parser.add_argument("--save-html", type=str, help="Save HTML to file")
    args = parser.parse_args()
    
    print("=" * 65)
    print("🚀 VIX 5% Weekly Suite - POSITION-AWARE Signal Generator")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # 1. Fetch market data
    try:
        current_price, percentile, slope = fetch_uvxy_data()
    except Exception as e:
        print(f"❌ Failed to fetch UVXY data: {e}")
        sys.exit(1)
    
    print(f"\n📈 Current Market State:")
    print(f"   UVXY: ${current_price:.2f}")
    print(f"   Percentile: {percentile:.1%}")
    print(f"   5-day slope: {slope:+.3f}")
    
    # 2. Detect regime
    regime_state = RegimeState(
        regime=classify_regime(current_price, vix_percentile=percentile),
        vix_level=current_price,
        vix_percentile=percentile,
        confidence=0.5 + abs(percentile - 0.5),
        vix_slope=slope,
    )
    
    print(f"\n🎯 Regime Detection:")
    print(f"   Regime: {regime_state.regime.value.upper()}")
    print(f"   Confidence: {regime_state.confidence:.0%}")
    
    # 3. Generate all 5 variants
    batch = generate_all_variants(regime_state)
    print(f"\n📋 Generated {len(batch.variants)} variants")
    
    # 4. Load trade log and classify variants
    trade_log = get_trade_log()
    variant_states = classify_variants(batch, trade_log, regime_state.regime)
    
    # Count by category
    management = [s for s in variant_states if s.has_position]
    recommended = [s for s in variant_states if not s.has_position and s.is_recommended]
    paper_test = [s for s in variant_states if not s.has_position and not s.is_recommended]
    
    print(f"   🔄 {len(management)} with OPEN POSITIONS (management mode)")
    print(f"   🎯 {len(recommended)} ENTRY CANDIDATES (would trade)")
    print(f"   🔬 {len(paper_test)} PAPER TEST ONLY (observe)")
    
    # 5. Build email
    html = build_position_aware_email(batch, variant_states)
    
    # 6. Determine subject
    subject = f"[VIX 5%] {regime_state.regime.value.upper()} ({percentile:.0%}) — "
    subject += f"{len(management)} Open, {len(recommended)} Entry, {len(paper_test)} Observe"
    
    # 7. Send or preview
    if args.dry_run:
        print("\n" + "=" * 65)
        print("🔍 DRY RUN — Email Preview")
        print("=" * 65)
        print(f"\n   To: {args.to}")
        print(f"   Subject: {subject}")
        print(f"\n   🔄 OPEN POSITIONS ({len(management)}):")
        for s in management:
            name = get_variant_display_name(s.variant.role)
            pnl = s.current_pnl or 0
            dte = s.dte_remaining or 0
            print(f"      • {name}: ${pnl:+,.0f} P&L, {dte} DTE")
            print(f"        → {s.action_suggestion}")
        
        print(f"\n   🎯 ENTRY CANDIDATES ({len(recommended)}):")
        for s in recommended:
            name = get_variant_display_name(s.variant.role)
            v = s.variant
            print(f"      ✅ {name}")
            print(f"         Entry ≤{v.entry_percentile:.0%} | +{v.long_strike_offset}pts | {v.long_dte_weeks}w")
            print(f"         Target: ${s.suggested_target_price} | Stop: ${s.suggested_stop_price}")
        
        print(f"\n   🔬 PAPER TEST ({len(paper_test)}):")
        for s in paper_test:
            name = get_variant_display_name(s.variant.role)
            active_in = ", ".join([r.value.upper() for r in s.variant.active_in_regimes])
            print(f"      🔬 {name} (Activates in: {active_in})")
    else:
        send_email(html, args.to, subject)
    
    # 8. Save HTML if requested
    if args.save_html:
        with open(args.save_html, 'w') as f:
            f.write(html)
        print(f"\n📄 HTML saved to {args.save_html}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
