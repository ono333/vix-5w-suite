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
from market_calendar import format_calendar_warning, is_market_open, get_next_trading_day

# ============================================================
# Helper Functions
# ============================================================

def estimate_entry_credit(vix_level: float, strike_offset: float, dte_weeks: int) -> float:
    """
    Estimate option credit based on VIX level and position parameters.
    This is a rough approximation for display purposes.
    """
    # Base credit scales with VIX level
    base_credit = vix_level * 0.02
    
    # Adjust for strike distance (closer = more credit)
    strike_factor = max(0.5, 1.0 - (strike_offset / 20.0))
    
    # Adjust for time (more DTE = more credit)
    time_factor = min(2.0, 0.5 + (dte_weeks / 26.0))
    
    credit = base_credit * strike_factor * time_factor
    return round(max(0.25, min(5.00, credit)), 2)


def compute_price_targets(entry_credit: float, tp_pct: float, sl_pct: float) -> dict:
    """Compute target and stop prices from entry credit."""
    return {
        "target": round(entry_credit * (1 - tp_pct), 2),
        "stop": round(entry_credit * (1 + sl_pct), 2),
    }



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
    Always calculates fresh entry signals for all variants (for paper testing comparison).
    """
    states = []
    vix_level = batch.regime_state.vix_level if batch.regime_state else 20.0
    
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
        
        # ALWAYS calculate fresh entry signals (for paper testing comparison)
        # Estimate entry credit based on VIX level and variant parameters
        base_credit = 1.0 + (vix_level - 15) * 0.1
        # Adjust based on strike offset (further OTM = less credit)
        offset_factor = max(0.5, 1.0 - abs(variant.long_strike_offset) * 0.02)
        state.suggested_entry_credit = round(base_credit * offset_factor, 2)
        
        if state.suggested_entry_credit > 0:
            state.suggested_target_price = round(
                state.suggested_entry_credit * (1 - variant.tp_pct), 2
            )
            state.suggested_stop_price = round(
                state.suggested_entry_credit * (1 + variant.sl_pct), 2
            )
        
        # Add management info if has position
        if has_position and position:
            state.current_pnl = position.current_pnl
            state.current_pnl_pct = position.current_pnl_pct
            state.dte_remaining = position.days_to_expiry()
            
            # Get actual diagonal position for short leg analysis
            from trade_log import get_trade_log
            tl = get_trade_log()
            diag = None
            for pid, d in tl.diagonal_positions.items():
                if d.variant_id.upper() == variant_id.upper() and d.status == "open":
                    diag = d
                    break
            
            # Suggest action based on SHORT LEG status (not total P&L)
            short = diag.current_short_leg if diag else None
            
            if short:
                short_dte = short.days_to_expiry()
                short_current = short.current_price
                short_credit = short.entry_credit
                target_price = short_credit * 0.20  # 80% profit target
                stop_price = short_credit * 2.0     # Double = stop
                
                if short_dte <= 0:
                    state.action_suggestion = "ROLL_NOW"  # Expired, roll immediately
                elif short_dte <= 3:
                    state.action_suggestion = "ROLL"  # Roll soon
                elif short_current <= target_price:
                    state.action_suggestion = "TAKE_PROFIT"  # Can close early for profit
                elif short_current >= stop_price:
                    state.action_suggestion = "STOP_LOSS"  # Short went against us
                elif not is_recommended:
                    state.action_suggestion = "REGIME_DRIFT"
                else:
                    state.action_suggestion = "HOLD"
            else:
                # No active short - need to sell one
                state.action_suggestion = "SELL_SHORT"
        
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
    """Build HTML email showing both open positions AND fresh signals for all variants."""
    from datetime import datetime, timedelta
    regime_state = batch.regime_state
    regime_name = regime_state.regime.value.upper() if regime_state else "UNKNOWN"
    vix_level = regime_state.vix_level if regime_state else 20.0
    vix_pct = regime_state.vix_percentile if regime_state else 0
    
    management_variants = [s for s in variant_states if s.has_position]
    
    # All variants for fresh signals section
    all_variants = variant_states
    recommended_variants = [s for s in variant_states if s.is_recommended]
    paper_test_variants = [s for s in variant_states if not s.is_recommended]
    
    # Regime colors
    regime_colors = {
        "CALM": "#4CAF50", "DECLINING": "#FFC107", "RISING": "#FF9800",
        "STRESSED": "#f44336", "EXTREME": "#9C27B0"
    }
    regime_color = regime_colors.get(regime_name, "#607D8B")
    
    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, sans-serif; background: #f5f5f5; color: #333; padding: 20px; margin: 0;">
<div style="max-width: 650px; margin: 0 auto; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    
    <!-- Header -->
    <div style="background: linear-gradient(135deg, #1a73e8, #4285f4); color: #fff; padding: 25px; text-align: center;">
        <div style="font-size: 22px; font-weight: bold;">📈 VIX 5% Weekly Suite</div>
        <div style="font-size: 14px; opacity: 0.9; margin-top: 8px;">Position Report — {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')} ET</div>
    </div>
    
    <!-- Market State -->
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr>
            <td style="padding: 15px; text-align: center; background: {regime_color}; color: #fff; width: 33%;">
                <div style="font-size: 24px; font-weight: bold;">{regime_name}</div>
                <div style="font-size: 11px; opacity: 0.9;">Regime</div>
            </td>
            <td style="padding: 15px; text-align: center; background: #f8f9fa; width: 33%;">
                <div style="font-size: 24px; font-weight: bold; color: #333;">${vix_level:.2f}</div>
                <div style="font-size: 11px; color: #666;">UVXY Price</div>
            </td>
            <td style="padding: 15px; text-align: center; background: #f8f9fa; width: 33%;">
                <div style="font-size: 24px; font-weight: bold; color: #333;">{vix_pct:.0%}</div>
                <div style="font-size: 11px; color: #666;">52w Percentile</div>
            </td>
        </tr>
    </table>
    
    <!-- Stats -->
    <table style="width: 100%; border-collapse: collapse; margin: 0 0 20px 0;">
        <tr>
            <td style="padding: 12px; text-align: center; background: #e8f5e9; border-left: 4px solid #4CAF50; width: 33%;">
                <div style="font-size: 28px; font-weight: bold; color: #2e7d32;">{len(management_variants)}</div>
                <div style="font-size: 11px; color: #666;">🔄 Open Positions</div>
            </td>
            <td style="padding: 12px; text-align: center; background: #e3f2fd; border-left: 4px solid #2196F3; width: 33%;">
                <div style="font-size: 28px; font-weight: bold; color: #1565c0;">{len(recommended_variants)}</div>
                <div style="font-size: 11px; color: #666;">🎯 Recommended</div>
            </td>
            <td style="padding: 12px; text-align: center; background: #fafafa; border-left: 4px solid #9e9e9e; width: 33%;">
                <div style="font-size: 28px; font-weight: bold; color: #616161;">{len(paper_test_variants)}</div>
                <div style="font-size: 11px; color: #666;">🔬 Paper Test</div>
            </td>
        </tr>
    </table>
"""
    
    # Calendar warning
    calendar_warning = format_calendar_warning()
    if calendar_warning:
        html += f"""
    <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 0 15px 20px 15px; border-radius: 4px;">
        <div style="font-weight: bold; color: #e65100; margin-bottom: 8px;">📅 Calendar Alerts</div>
        <div style="font-size: 13px; color: #333; line-height: 1.6;">
"""
        for line in calendar_warning.split("\n"):
            if line.strip():
                html += f"            ⚠️ {line}<br>\n"
        html += """        </div>
    </div>
"""
    
    # ================================================================
    # SECTION 1: OPEN POSITIONS (Management Mode)
    # ================================================================
    if management_variants:
        html += """
    <div style="margin: 0 15px 20px 15px;">
        <div style="font-size: 16px; font-weight: bold; color: #1a73e8; padding-bottom: 10px; border-bottom: 2px solid #1a73e8; margin-bottom: 15px;">
            🔄 OPEN POSITIONS — Management Mode
        </div>
"""
        for state in management_variants:
            pos = state.position
            variant = state.variant
            name = get_variant_display_name(variant.role)
            
            pnl = state.current_pnl or 0
            pnl_pct = state.current_pnl_pct or 0
            dte = state.dte_remaining or 0
            action = state.action_suggestion or "HOLD"
            
            pnl_color = "#2e7d32" if pnl >= 0 else "#c62828"
            border_color = "#4CAF50" if pnl >= 0 else "#f44336"
            
            # Get long/short P&L separately from diagonal position
            long_pnl = 0
            short_pnl = 0
            short_target = 0
            short_stop = 0
            short_current = 0
            short_credit = 0
            short_dte = 0
            short_coverage = 0
            
            # Access the actual diagonal position for detailed info
            from trade_log import get_trade_log
            tl = get_trade_log()
            diag = None
            for pid, d in tl.diagonal_positions.items():
                if d.variant_id.upper() == variant.role.value.upper() and d.status == "open":
                    diag = d
                    break
            
            if diag:
                long_pnl = diag.long_pnl
                short_pnl = diag.short_pnl
                short_coverage = diag.short_coverage_pct
                short = diag.current_short_leg
                if short:
                    short_credit = short.entry_credit
                    short_current = short.current_price
                    short_target = short_credit * 0.20  # Buy back at 20% = 80% profit
                    short_stop = short_credit * 2.0     # Stop if doubled
                    short_dte = short.days_to_expiry()
            
            # Suggested roll strikes based on current UVXY price
            suggested_conservative = round(vix_level * 1.02, 0)
            suggested_moderate = round(vix_level * 1.05, 0)
            suggested_aggressive = round(vix_level * 1.10, 0)
            
            action_colors = {
                "TAKE_PROFIT": ("#4CAF50", "🎯 TAKE PROFIT - Short decayed, can close early"),
                "STOP_LOSS": ("#f44336", "🛑 STOP LOSS - Short doubled, manage risk"),
                "ROLL": ("#ff9800", "🔄 ROLL SOON - Short expiring in ≤3 days"),
                "ROLL_NOW": ("#f44336", "⚠️ ROLL NOW - Short expired!"),
                "SELL_SHORT": ("#9c27b0", "📝 SELL SHORT - No active short leg"),
                "CLOSE": ("#9c27b0", "⚠️ CLOSE - Exit recommended"),
                "HOLD": ("#2196F3", "✋ HOLD - Short on track"),
                "REGIME_DRIFT": ("#ff9800", "⚠️ REGIME DRIFT - Consider closing")
            }
            action_color, action_text = action_colors.get(action.upper(), ("#607D8B", f"ℹ️ {action}"))
            
            long_pnl_color = "#2e7d32" if long_pnl >= 0 else "#c62828"
            short_pnl_color = "#2e7d32" if short_pnl >= 0 else "#c62828"
            
            html += f"""
        <div style="background: #fafafa; border-left: 4px solid {border_color}; border-radius: 4px; padding: 15px; margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 15px; color: #333; margin-bottom: 10px;">{name}</div>
            <table style="width: 100%; font-size: 13px;">
                <tr>
                    <td style="padding: 4px 0; color: #666;">Long P&L:</td>
                    <td style="padding: 4px 0; font-weight: bold; color: {long_pnl_color};">${long_pnl:+,.0f}</td>
                    <td style="padding: 4px 0; color: #666;">Short P&L:</td>
                    <td style="padding: 4px 0; font-weight: bold; color: {short_pnl_color};">${short_pnl:+,.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0; color: #666;">Total P&L:</td>
                    <td style="padding: 4px 0; font-weight: bold; color: {pnl_color};">${pnl:+,.0f} ({pnl_pct:+.1%})</td>
                    <td style="padding: 4px 0; color: #666;">Long DTE:</td>
                    <td style="padding: 4px 0; font-weight: 500;">{dte} days</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0; color: #666;">Short Coverage:</td>
                    <td style="padding: 4px 0; font-weight: 500;">{short_coverage:.0f}%</td>
                    <td colspan="2"></td>
                </tr>
            </table>
            <div style="margin-top: 10px; font-size: 11px; color: #666; margin-bottom: 4px;">Short Leg Management (DTE: {short_dte}d):</div>
            <table style="width: 100%; font-size: 12px; margin-bottom: 8px;">
                <tr>
                    <td style="color: #666;">Sold at:</td>
                    <td style="font-weight: 500;">${short_credit:.2f}</td>
                    <td style="color: #666;">Current:</td>
                    <td style="font-weight: 500;">${short_current:.2f}</td>
                    <td style="color: #2e7d32;">Target:</td>
                    <td style="font-weight: 500; color: #2e7d32;">${short_target:.2f}</td>
                    <td style="color: #c62828;">Stop:</td>
                    <td style="font-weight: 500; color: #c62828;">${short_stop:.2f}</td>
                </tr>
            </table>
            <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Suggested Roll Strikes:</div>
            <div style="margin-top: 4px;">
                <span style="background: #e3f2fd; color: #1565c0; padding: 6px 12px; border-radius: 4px; font-size: 12px; display: inline-block; margin-right: 8px;">
                    🟢 ${suggested_conservative:.0f}
                </span>
                <span style="background: #fff3e0; color: #e65100; padding: 6px 12px; border-radius: 4px; font-size: 12px; display: inline-block; margin-right: 8px;">🟡 ${suggested_moderate:.0f}</span>
                <span style="background: #ffebee; color: #c62828; padding: 6px 12px; border-radius: 4px; font-size: 12px; display: inline-block;">
                    🔴 ${suggested_aggressive:.0f}
                </span>
            </div>
            <div style="margin-top: 12px; background: {action_color}; color: #fff; padding: 10px 15px; border-radius: 4px; font-size: 13px; font-weight: 500;">
                {action_text}
            </div>
        </div>
"""
        html += """    </div>
"""
    
    # ================================================================
    # SECTION 2: FRESH SIGNALS — All Variants (for paper testing comparison)
    # ================================================================
    html += """
    <div style="margin: 0 15px 20px 15px;">
        <div style="font-size: 16px; font-weight: bold; color: #2e7d32; padding-bottom: 10px; border-bottom: 2px solid #4CAF50; margin-bottom: 15px;">
            📊 TODAY'S FRESH SIGNALS — All Variants
        </div>
        <div style="font-size: 12px; color: #666; margin-bottom: 15px;">
            Compare current market signals with your open positions. Fixed 5 contracts per variant for paper testing.
        </div>
"""
    
    # Fixed contract size for paper testing
    PAPER_CONTRACTS = 5
    
    for state in all_variants:
        variant = state.variant
        name = get_variant_display_name(variant.role)
        is_recommended = state.is_recommended
        has_position = state.has_position
        
        entry_credit = state.suggested_entry_credit or 0
        
        # Calculate suggested strike based on UVXY price and variant offset
        short_offset = getattr(variant, 'short_strike_offset', 2)
        target = round(vix_level + short_offset, 0)  # Suggested short strike
        stop = round(target * 1.3, 0)  # Stop level (30% above strike)
        
        # Calculate actual expiry dates (datetime/timedelta imported at top)
        today = datetime.now()
        long_expiry = today + timedelta(weeks=variant.long_dte_weeks)
        short_expiry = today + timedelta(weeks=variant.short_dte_weeks)
        # Find next Friday for short expiry
        days_to_friday = (4 - short_expiry.weekday()) % 7
        if days_to_friday == 0 and variant.short_dte_weeks > 0:
            days_to_friday = 7
        short_expiry = short_expiry + timedelta(days=days_to_friday)
        # Format dates
        long_exp_str = long_expiry.strftime("%b %d")
        short_exp_str = short_expiry.strftime("%b %d")
        
        # Border and background based on recommendation status
        if is_recommended:
            border_color = "#4CAF50"
            bg_color = "#f1f8e9"
            status_badge = '<span style="background: #4CAF50; color: #fff; padding: 3px 8px; border-radius: 3px; font-size: 10px; font-weight: bold;">RECOMMENDED</span>'
        else:
            border_color = "#9e9e9e"
            bg_color = "#fafafa"
            active_regimes = ", ".join([r.value.upper() for r in variant.active_in_regimes])
            status_badge = f'<span style="background: #9e9e9e; color: #fff; padding: 3px 8px; border-radius: 3px; font-size: 10px;">PAPER TEST (activates in {active_regimes})</span>'
        
        # Position indicator
        position_indicator = ""
        if has_position:
            position_indicator = '<span style="background: #2196F3; color: #fff; padding: 3px 8px; border-radius: 3px; font-size: 10px; margin-left: 5px;">HAS POSITION</span>'
        
        html += f"""
        <div style="background: {bg_color}; border-left: 4px solid {border_color}; border-radius: 4px; padding: 15px; margin-bottom: 12px;">
            <div style="margin-bottom: 10px;">
                <span style="font-weight: bold; font-size: 15px; color: #333;">{name}</span>
                {position_indicator}
            </div>
            <div style="margin-bottom: 8px;">{status_badge}</div>
            <table style="width: 100%; font-size: 13px;">
                <tr>
                    <td style="padding: 4px 0; color: #666;">Long Strike:</td>
                    <td style="padding: 4px 0; font-weight: 500;">${variant.long_strike:.0f}</td>
                    <td style="padding: 4px 0; color: #666;">Short Strike:</td>
                    <td style="padding: 4px 0; font-weight: 500;">{target:.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0; color: #666;">Long Exp:</td>
                    <td style="padding: 4px 0; font-weight: 500;">{long_exp_str} ({variant.long_dte_weeks}w)</td>
                    <td style="padding: 4px 0; color: #666;">Short Exp:</td>
                    <td style="padding: 4px 0; font-weight: 500;">{short_exp_str} (roll {variant.roll_dte_days}d)</td>
                </tr>
                <tr style="background: #e8f5e9;">
                    <td style="padding: 6px 0; color: #666;">Est. Credit:</td>
                    <td style="padding: 6px 0; font-weight: bold; color: #2e7d32;">${entry_credit:.2f}/contract</td>
                    <td style="padding: 6px 0; color: #666;">Contracts:</td>
                    <td style="padding: 6px 0; font-weight: 500;">{PAPER_CONTRACTS}</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0; color: #666;">Target:</td>
                    <td style="padding: 4px 0; color: #2e7d32;">${target:.2f}</td>
                    <td style="padding: 4px 0; color: #666;">Stop:</td>
                    <td style="padding: 4px 0; color: #c62828;">${stop:.2f}</td>
                </tr>
            </table>
        </div>
"""
    
    html += """    </div>
"""
    
    # Footer
    html += """
    <div style="text-align: center; padding: 20px; background: #f5f5f5; color: #999; font-size: 11px;">
        VIX 5% Weekly Suite — Paper Testing Mode<br>
        Generated automatically. Do not reply.
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
    regime_state = classify_regime(current_price, vix_percentile=percentile)
    # Update with additional data
    regime_state = RegimeState(
        regime=regime_state.regime,
        vix_level=current_price,
        vix_percentile=percentile,
        confidence=0.5 + abs(percentile - 0.5),
        vix_slope=slope,
    )
    
    print(f"\n🎯 Regime Detection:")
    print(f"   Regime: {regime_state.regime.value.upper()}")
    print(f"   Confidence: {regime_state.confidence:.0%}")
    
    # Check market calendar
    from datetime import date
    today = date.today()
    if not is_market_open(today):
        next_open = get_next_trading_day(today)
        print(f"\n⛔ MARKET CLOSED TODAY - Next trading day: {next_open.strftime('%A, %b %d')}")
    
    calendar_warning = format_calendar_warning()
    if calendar_warning:
        print(f"\n📅 Calendar Alerts:")
        for line in calendar_warning.split("\n"):
            print(f"   {line}")
    
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
            est_cr = estimate_entry_credit(regime_state.vix_level, v.long_strike_offset, v.long_dte_weeks)
            targets = compute_price_targets(est_cr, v.tp_pct, v.sl_pct)
            print(f"         Credit: ${est_cr:.2f} | Target: ${targets['target']:.2f} | Stop: ${targets['stop']:.2f}")
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
