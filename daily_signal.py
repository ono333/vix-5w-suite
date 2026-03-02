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
from real_trade_log import get_real_trade_log
from market_calendar import format_calendar_warning, is_market_open, get_next_trading_day
from roll_manager import evaluate_roll

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
            short_current = 0.0
            short_target = 0.0
            short_stop = 0.0
            short_credit = 0.0
            
            if short:
                # Fix: calculate DTE directly, with null guard
                from datetime import date as _date
                if short is None:
                    short_dte = -1
                else:
                    try:
                        _exp = _date.fromisoformat(short.expiration_date)
                        short_dte = max(0, (_exp - _date.today()).days)
                    except:
                        short_dte = 0
                short_credit = short.entry_credit if short else 0.0
                # Re-evaluate action using correct DTE (overrides state.action_suggestion)
                from roll_manager import evaluate_roll as _er
                _roll = _er(
                    dte_remaining    = max(short_dte, 0),
                    short_delta      = getattr(short, "delta", None) if short else None,
                    uvxy_price       = batch.uvxy_price if hasattr(batch, 'uvxy_price') else uvxy_price if 'uvxy_price' in dir() else 38.0,
                    short_strike     = short.strike if short else 0.0,
                    variant_params   = variant.__dict__ if hasattr(variant, "__dict__") else {},
                    last_spike_date  = None,
                    original_premium = short_credit if short_credit else None,
                )
                if short is None:
                    action = "SELL_SHORT"
                elif _roll.action == "roll_now":
                    action = "ROLL_NOW"
                elif _roll.action in ("roll_early_delta", "roll_early_itm"):
                    action = "ROLL"
                elif _roll.action == "spike_guard_hold":
                    action = "HOLD"
                elif short_current <= short_target and short_target > 0:
                    action = "TAKE_PROFIT"
                elif short_current >= short_stop and short_stop > 0:
                    action = "STOP_LOSS"
                else:
                    action = "HOLD"
                short_current = short.current_price
                short_credit = short.entry_credit
                target_price = short_credit * 0.20  # 80% profit target
                stop_price = short_credit * 2.0     # Double = stop
                
                _roll = evaluate_roll(
                    dte_remaining    = max(short_dte, 0),
                    short_delta      = getattr(short, "delta", None),
                    uvxy_price       = getattr(short, "underlying_price", short.strike),
                    short_strike     = short.strike,
                    variant_params   = variant.__dict__ if hasattr(variant, "__dict__") else {},
                    last_spike_date  = None,
                    original_premium = short_credit,
                )
                if _roll.action == "roll_now":
                    state.action_suggestion = "ROLL_NOW"
                elif _roll.action in ("roll_early_delta", "roll_early_itm"):
                    state.action_suggestion = "ROLL"
                elif _roll.action == "spike_guard_hold":
                    state.action_suggestion = "HOLD"
                elif short_current <= target_price:
                    state.action_suggestion = "TAKE_PROFIT"
                elif short_current >= stop_price:
                    state.action_suggestion = "STOP_LOSS"
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


# ─────────────────────────────────────────────────────────────────────
# EMAIL HELPERS — Risk computations
# ─────────────────────────────────────────────────────────────────────

def _bs_delta(S: float, K: float, T: float, sigma: float = 0.85,
              r: float = 0.0, option_type: str = "call") -> float:
    """Synthetic Black-Scholes delta for short leg monitoring."""
    import math
    if T <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        from scipy.stats import norm
        return float(norm.cdf(d1)) if option_type == "call" else float(norm.cdf(d1) - 1)
    except Exception:
        # Fallback: moneyness approximation
        moneyness = K / S - 1.0
        if moneyness > 0.12: return 0.15
        if moneyness > 0.08: return 0.25
        if moneyness > 0.04: return 0.35
        if moneyness > 0.00: return 0.45
        return 0.60


def _fetch_iv_term_structure() -> dict:
    """Fetch VIX term structure for front/back IV ratio."""
    try:
        import yfinance as yf
        vix9d  = yf.Ticker("^VIX9D").history(period="2d")["Close"].iloc[-1]
        vix1m  = yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1]
        vix3m  = yf.Ticker("^VIX3M").history(period="2d")["Close"].iloc[-1]
        ratio  = round(float(vix1m) / float(vix3m), 3) if vix3m > 0 else 1.0
        if   ratio < 0.90: term = "Strong Contango"
        elif ratio < 1.00: term = "Mild Contango"
        elif ratio < 1.05: term = "Flat"
        elif ratio < 1.15: term = "Mild Backwardation"
        else:              term = "Strong Backwardation"
        return {
            "front_iv":  round(float(vix1m), 1),
            "back_iv":   round(float(vix3m), 1),
            "ratio":     ratio,
            "term":      term,
            "vix9d":     round(float(vix9d), 1),
        }
    except Exception:
        return {"front_iv": 0, "back_iv": 0, "ratio": 1.0,
                "term": "N/A", "vix9d": 0}


def _short_strike_band(vix_level: float) -> str:
    """Dynamic short strike OTM band from ChatGPT spec."""
    if vix_level < 17:   return "9–11% OTM"
    if vix_level <= 22:  return "6–8% OTM"
    if vix_level <= 25:  return "4–6% OTM"
    return "PAUSE — VIX > 25"


def _estimate_roll_debit(uvxy_price: float, current_strike: float,
                         new_strike: float, dte_weeks: float = 1.0,
                         sigma: float = 0.85) -> float:
    """Estimate roll net debit/credit to a new strike."""
    try:
        import math
        from scipy.stats import norm
        def bs_call(S, K, T, sig):
            if T <= 0: return max(0, S - K)
            d1 = (math.log(S/K) + 0.5*sig**2*T) / (sig*math.sqrt(T))
            d2 = d1 - sig*math.sqrt(T)
            return S*norm.cdf(d1) - K*norm.cdf(d2)
        T = dte_weeks / 52.0
        bb   = bs_call(uvxy_price, current_strike, 0, sigma)   # buy back (0 DTE ≈ intrinsic)
        new  = bs_call(uvxy_price, new_strike, T, sigma)
        return round(new - bb, 2)   # positive = net credit
    except Exception:
        # Rough approximation
        otm_pct = (new_strike - uvxy_price) / uvxy_price
        base = max(0.05, 2.0 - otm_pct * 15)
        return round(base, 2)


def _stress_test(positions: list, uvxy_scenarios: list,
                 uvxy_current: float) -> list:
    """Simple linear stress test — estimates P&L at each UVXY level."""
    results = []
    for target in uvxy_scenarios:
        move_pct = (target - uvxy_current) / uvxy_current
        total_impact = 0.0
        for pos in positions:
            try:
                # Long delta gain
                long_delta = _bs_delta(uvxy_current, pos.long_strike,
                                       pos.days_to_long_expiry() / 365)
                long_gain = long_delta * move_pct * uvxy_current * pos.contracts * 100
                # Short delta loss
                short = pos.current_short_leg
                if short:
                    short_delta = _bs_delta(uvxy_current, short.strike,
                                            pos.days_to_expiry() / 365)
                    short_loss = -short_delta * move_pct * uvxy_current * pos.contracts * 100
                else:
                    short_loss = 0
                total_impact += long_gain + short_loss
            except Exception:
                pass
        results.append({"uvxy": target, "impact": round(total_impact, 0)})
    return results


def _scaling_eligible(pos) -> tuple:
    """Return (eligible: bool, reason: str) for scaling check."""
    rh = getattr(pos, "roll_history", [])
    if len(rh) < 2:
        return False, "Insufficient history"
    last2 = rh[-2:]
    # Check debit rolls
    for r in last2:
        if getattr(r, "roll_credit", 0) < 0:
            return False, "Recent debit roll"
    # Check profitability — net credits > 0 for last 2
    credits = [getattr(r, "roll_credit", 0) for r in last2]
    if any(c <= 0 for c in credits):
        return False, "Recent unprofitable roll"
    return True, "+1 contract allowed"


def _data_age_warning(fetch_time) -> str:
    """Return warning if data is stale (> 2 min)."""
    from datetime import datetime
    try:
        age = (datetime.now() - fetch_time).total_seconds()
        if age > 120:
            mins = int(age // 60)
            return f"⚠️ DATA STALE: {mins}min old — disable new entries"
        return ""
    except Exception:
        return ""


def _roll_mode(short_delta: float, short_dte: int, short_strike: float,
               uvxy_price: float, debit_cap: float = 1.50) -> dict:
    """
    Three-mode roll classifier per ChatGPT spec.
    Returns dict with mode, color, emoji, action_label, reason, target_otm_pct.
    """
    itm = short_strike <= uvxy_price

    if short_dte <= 0 or (short_dte <= 2 and itm):
        return dict(
            mode        = "EMERGENCY",
            color       = "#cc0000",
            emoji       = "🚨",
            badge       = "🚨 EMERGENCY ROLL",
            action      = "Roll immediately — position ITM or expired",
            reason      = "Short expired or ITM — gamma risk extreme",
            target_otm  = "6–9% OTM minimum",
            target_note = "Accept debit if necessary to restore structure",
            priority    = 3,
        )
    elif short_delta >= 0.50 or itm:
        return dict(
            mode        = "EMERGENCY",
            color       = "#cc0000",
            emoji       = "🚨",
            badge       = "🚨 EMERGENCY ROLL",
            action      = "Roll now — delta ≥ 0.50, gamma accelerating",
            reason      = f"Short delta {short_delta:.2f} — credit window closing",
            target_otm  = "6–9% OTM",
            target_note = "Roll before debit exceeds cap",
            priority    = 3,
        )
    elif short_delta >= 0.40 or (short_dte <= 4 and short_delta >= 0.35):
        return dict(
            mode        = "DEFENSIVE",
            color       = "#ff6600",
            emoji       = "🛡️",
            badge       = "🛡️ DEFENSIVE ROLL SUGGESTED",
            action      = "Roll forward this week — credit still available",
            reason      = f"Short delta {short_delta:.2f} approaching 0.50 — act before gamma spikes",
            target_otm  = "4–6% OTM",
            target_note = "Target strike that keeps delta < 0.40 after roll",
            priority    = 2,
        )
    elif short_dte <= 2:
        return dict(
            mode        = "ROUTINE",
            color       = "#ff9800",
            emoji       = "🔄",
            badge       = "🔄 ROUTINE ROLL",
            action      = "Roll as scheduled — DTE ≤ 2",
            reason      = f"DTE {short_dte}d — standard weekly harvest roll",
            target_otm  = "4–6% OTM",
            target_note = "Maintain net credit above debit cap",
            priority    = 1,
        )
    else:
        return dict(
            mode        = "HOLD",
            color       = "#2196F3",
            emoji       = "✅",
            badge       = "✅ On Track",
            action      = "Hold — structure healthy",
            reason      = f"Short delta {short_delta:.2f}, DTE {short_dte}d — no action needed",
            target_otm  = "—",
            target_note = "Monitor daily",
            priority    = 0,
        )


def _convexity_status(short_delta: float) -> tuple:
    """Return (label, color, emoji) for convexity status."""
    if short_delta < 0.35:  return "Stable",    "#4CAF50", "🟢"
    if short_delta < 0.50:  return "Active",     "#FF9800", "🟡"
    return                          "Defensive",  "#f44336", "🔴"


def build_position_aware_email(
    batch,
    variant_states,
) -> str:
    """
    Upgraded email — Risk & Structure Dashboard.
    Paper trades: blue theme, PAPER watermark.
    """
    from datetime import datetime, timedelta
    import math

    fetch_time   = datetime.now()
    regime_state = batch.regime_state
    regime_name  = regime_state.regime.value.upper() if regime_state else "UNKNOWN"
    vix_level    = regime_state.vix_level if regime_state else 20.0
    vix_pct      = regime_state.vix_percentile if regime_state else 0

    management_variants  = [s for s in variant_states if s.has_position]
    all_variants         = variant_states
    recommended_variants = [s for s in variant_states if s.is_recommended]
    paper_test_variants  = [s for s in variant_states if not s.is_recommended]

    # ── Market data
    ts_data  = _fetch_iv_term_structure()
    stale_warn = _data_age_warning(fetch_time)

    regime_colors = {
        "CALM": "#2196F3", "DECLINING": "#FFC107",
        "RISING": "#FF9800", "STRESSED": "#f44336", "EXTREME": "#9C27B0"
    }
    regime_color = regime_colors.get(regime_name, "#607D8B")

    # ── Collect diagonal positions
    from trade_log import get_trade_log
    tl = get_trade_log()

    open_diags = [d for d in tl.diagonal_positions.values() if d.status == "open"]
    total_long_cost  = sum(d.long_cost if hasattr(d, "long_cost") else
                           d.long_entry_price * d.contracts * 100
                           for d in open_diags)
    total_net_credits = sum(d.short_pnl for d in open_diags)

    # ── Stress test
    stress = _stress_test(open_diags, [45, 50, 55], vix_level)

    # ── HTML start — PAPER blue theme
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;background:#f0f4ff;color:#333;padding:20px;margin:0;">
<div style="max-width:680px;margin:0 auto;background:#fff;border-radius:10px;
            overflow:hidden;box-shadow:0 2px 12px rgba(0,0,80,0.12);">

  <!-- PAPER HEADER -->
  <div style="background:linear-gradient(135deg,#1a73e8,#1557b0);color:#fff;
              padding:22px 25px 16px;text-align:center;">
    <div style="font-size:11px;letter-spacing:3px;opacity:0.7;margin-bottom:4px;">
      📋 PAPER TRADING MODE
    </div>
    <div style="font-size:21px;font-weight:800;">📈 VIX 5% Weekly Suite</div>
    <div style="font-size:12px;opacity:0.85;margin-top:6px;">
      Live Position Risk Report — {fetch_time.strftime('%A, %b %d, %Y  %I:%M %p')} ET
    </div>
  </div>
"""

    # ── Data staleness warning
    if stale_warn:
        html += f"""
  <div style="background:#fff3e0;border-left:5px solid #ff9800;padding:12px 20px;
              font-size:13px;color:#e65100;font-weight:600;">
    ⏱️ {stale_warn}
  </div>
"""

    # ── SECTION A: MARKET STATE
    html += f"""
  <!-- MARKET STATE -->
  <div style="background:#f8faff;border-bottom:1px solid #e0e8ff;padding:16px 20px;">
    <div style="font-size:13px;font-weight:700;color:#1a73e8;margin-bottom:10px;
                letter-spacing:1px;">🧭 MARKET STATE</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr>
        <td style="padding:5px 8px;color:#555;width:40%;">UVXY</td>
        <td style="padding:5px 8px;font-weight:700;color:#1a1a1a;">${vix_level:.2f}</td>
        <td style="padding:5px 8px;color:#555;width:30%;">Regime</td>
        <td style="padding:5px 8px;"><span style="background:{regime_color};color:#fff;
            padding:2px 8px;border-radius:3px;font-size:12px;font-weight:700;">
            {regime_name}</span></td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#555;">52w Percentile</td>
        <td style="padding:5px 8px;font-weight:700;">{vix_pct:.0%}</td>
        <td style="padding:5px 8px;color:#555;">Short Band</td>
        <td style="padding:5px 8px;font-weight:600;color:#1a73e8;">
            {_short_strike_band(vix_level)}</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#555;">Front IV (VIX)</td>
        <td style="padding:5px 8px;font-weight:700;">{ts_data['front_iv']:.1f}%</td>
        <td style="padding:5px 8px;color:#555;">Back IV (VIX3M)</td>
        <td style="padding:5px 8px;font-weight:700;">{ts_data['back_iv']:.1f}%</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#555;">IV Ratio</td>
        <td style="padding:5px 8px;font-weight:700;color:{'#f44336' if ts_data['ratio']>1.1 else '#FF9800' if ts_data['ratio']>1.0 else '#4CAF50'};">
            {ts_data['ratio']:.3f}</td>
        <td style="padding:5px 8px;color:#555;">Term Structure</td>
        <td style="padding:5px 8px;font-weight:600;">{ts_data['term']}</td>
      </tr>
    </table>
  </div>
"""

    # ── SECTION B: SYSTEM RISK SUMMARY
    html += f"""
  <!-- SYSTEM RISK SUMMARY -->
  <div style="background:#fff;border-bottom:1px solid #e8e8e8;padding:16px 20px;">
    <div style="font-size:13px;font-weight:700;color:#1a73e8;margin-bottom:10px;
                letter-spacing:1px;">⚙️ SYSTEM RISK SUMMARY</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr>
        <td style="padding:5px 8px;color:#555;">Open Positions</td>
        <td style="padding:5px 8px;font-weight:700;">{len(management_variants)}</td>
        <td style="padding:5px 8px;color:#555;">Long Capital</td>
        <td style="padding:5px 8px;font-weight:700;">${total_long_cost:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#555;">Net Credits</td>
        <td style="padding:5px 8px;font-weight:700;color:#2e7d32;">${total_net_credits:+,.0f}</td>
        <td style="padding:5px 8px;color:#555;">Mode</td>
        <td style="padding:5px 8px;"><span style="background:#1a73e8;color:#fff;
            padding:2px 8px;border-radius:3px;font-size:11px;">📋 PAPER</span></td>
      </tr>
    </table>
  </div>
"""

    # ── Calendar warning
    calendar_warning = format_calendar_warning()
    if calendar_warning:
        html += f"""
  <div style="background:#fff8e1;border-left:5px solid #ff9800;padding:12px 20px;
              margin:0;font-size:13px;">
    <div style="font-weight:700;color:#e65100;margin-bottom:6px;">📅 Calendar Alerts</div>
"""
        for line in calendar_warning.split("\n"):
            if line.strip():
                html += f'    <div>⚠️ {line}</div>\n'
        html += "  </div>\n"

    # ── SECTION C: OPEN POSITIONS — Risk-first
    if management_variants:
        html += """
  <!-- OPEN POSITIONS -->
  <div style="padding:16px 20px 0;">
    <div style="font-size:14px;font-weight:700;color:#1a73e8;padding-bottom:8px;
                border-bottom:2px solid #1a73e8;margin-bottom:14px;letter-spacing:1px;">
      🔄 OPEN POSITIONS — Risk &amp; Structure View
    </div>
"""
        for state in management_variants:
            pos = state.position
            variant = state.variant
            name = get_variant_display_name(variant.role)

            pnl     = state.current_pnl or 0
            pnl_pct = state.current_pnl_pct or 0
            dte     = state.dte_remaining or 0

            long_pnl = short_pnl = short_coverage = 0
            short_credit = short_current = short_dte = 0
            short_delta  = 0.30
            long_delta   = 0.35
            action = "HOLD"

            diag = None
            for pid, d in tl.diagonal_positions.items():
                if d.variant_id.upper() == variant.role.value.upper() and d.status == "open":
                    diag = d; break

            if diag:
                long_pnl      = diag.long_pnl
                short_pnl     = diag.short_pnl
                short_coverage = diag.short_coverage_pct
                short = diag.current_short_leg
                if short:
                    short_credit = short.entry_credit
                    short_current = short.current_price
                    # DTE
                    from datetime import date as _date
                    try:
                        short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days)
                    except Exception:
                        short_dte = 0
                    # Delta
                    long_T  = diag.days_to_long_expiry() / 365
                    short_T = short_dte / 365
                    short_delta = _bs_delta(vix_level, short.strike, short_T)
                    long_delta  = _bs_delta(vix_level, diag.long_strike, long_T)
                # Action
                from roll_manager import evaluate_roll as _er
                try:
                    _roll = _er(
                        dte_remaining    = max(short_dte, 0),
                        short_delta      = short_delta,
                        uvxy_price       = vix_level,
                        short_strike     = short.strike if short else 0,
                        variant_params   = variant.__dict__ if hasattr(variant, "__dict__") else {},
                        last_spike_date  = None,
                        original_premium = short_credit or None,
                    )
                    if short is None:          action = "SELL_SHORT"
                    elif _roll.action == "roll_now": action = "ROLL_NOW"
                    elif _roll.action in ("roll_early_delta","roll_early_itm"): action = "ROLL"
                    elif short_current <= short_credit * 0.20 and short_credit > 0: action = "TAKE_PROFIT"
                    elif short_current >= short_credit * 2.0  and short_credit > 0: action = "STOP_LOSS"
                    else: action = "HOLD"
                except Exception:
                    action = "ROLL_NOW" if short_dte <= 0 else "HOLD"

            # Convexity
            conv_label, conv_color, conv_emoji = _convexity_status(short_delta)

            # Roll debit estimates
            cur_strike = short.strike if short else vix_level
            roll_conservative = round(vix_level * 1.02)
            roll_moderate     = round(vix_level * 1.05)
            roll_aggressive   = round(vix_level * 1.10)
            rd_cons = _estimate_roll_debit(vix_level, cur_strike, roll_conservative)
            rd_mod  = _estimate_roll_debit(vix_level, cur_strike, roll_moderate)
            rd_agg  = _estimate_roll_debit(vix_level, cur_strike, roll_aggressive)

            # Scaling
            if diag:
                sc_ok, sc_reason = _scaling_eligible(diag)
            else:
                sc_ok, sc_reason = False, "No position"

            # Debit cap
            v_id = variant.role.value.upper()
            debit_cap = 1.50 if "V1" in v_id else 1.25 if "V5" in v_id else 1.50

            # Three-mode roll classifier
            _rm = _roll_mode(short_delta, short_dte,
                             float(diag.short_legs[-1].strike) if diag.short_legs else 999,
                             vix_level, debit_cap)
            urgency = _rm["badge"]
            ac      = _rm["color"]
            at      = _rm["action"]

            # Legacy action map for SELL_SHORT / TAKE_PROFIT overrides
            if action == "SELL_SHORT":
                urgency = "📝 SELL SHORT — No active short"
                ac = "#9c27b0"
            elif action == "TAKE_PROFIT":
                urgency = "🎯 TAKE PROFIT"
                ac = "#4CAF50"

            pnl_c = "#2e7d32" if pnl >= 0 else "#c62828"
            lpnl_c = "#2e7d32" if long_pnl >= 0 else "#c62828"
            spnl_c = "#2e7d32" if short_pnl >= 0 else "#c62828"

            html += f"""
    <div style="border:1px solid #e0e8ff;border-left:4px solid {ac};border-radius:6px;
                padding:14px;margin-bottom:14px;background:#fafcff;">

      <!-- Position header -->
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px;">
        <div style="font-weight:700;font-size:15px;color:#1a1a1a;">{name}</div>
        <div style="font-size:12px;background:{ac};color:#fff;padding:3px 10px;
                    border-radius:4px;font-weight:600;">{urgency}</div>
      </div>

      <!-- Structure Health -->
      <div style="font-size:11px;font-weight:700;color:#666;letter-spacing:1px;
                  margin-bottom:6px;text-transform:uppercase;">Structure Health</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px;
                    background:#f0f4ff;border-radius:4px;">
        <tr>
          <td style="padding:4px 8px;color:#555;">Short Delta</td>
          <td style="padding:4px 8px;font-weight:700;color:{'#f44336' if short_delta>=0.5 else '#FF9800' if short_delta>=0.35 else '#4CAF50'};">
              {short_delta:.3f}</td>
          <td style="padding:4px 8px;color:#555;">Convexity</td>
          <td style="padding:4px 8px;font-weight:700;">{conv_emoji} {conv_label}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#555;">Long Delta</td>
          <td style="padding:4px 8px;font-weight:700;">{long_delta:.3f}</td>
          <td style="padding:4px 8px;color:#555;">Short DTE</td>
          <td style="padding:4px 8px;font-weight:700;color:{'#f44336' if short_dte<=0 else '#FF9800' if short_dte<=2 else '#333'};">
              {short_dte}d</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#555;">Roll Trigger</td>
          <td colspan="3" style="padding:4px 8px;color:#555;font-size:11px;">
              Delta ≥ 0.50 &nbsp;|&nbsp; Debit &gt; ${debit_cap:.2f} &nbsp;|&nbsp; DTE ≤ 2 and ITM</td>
        </tr>
      </table>

      <!-- P&L -->
      <div style="font-size:11px;font-weight:700;color:#666;letter-spacing:1px;
                  margin-bottom:6px;text-transform:uppercase;">P&amp;L</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px;">
        <tr>
          <td style="padding:3px 8px;color:#555;">Long P&amp;L</td>
          <td style="padding:3px 8px;font-weight:700;color:{lpnl_c};">${long_pnl:+,.0f}</td>
          <td style="padding:3px 8px;color:#555;">Short P&amp;L</td>
          <td style="padding:3px 8px;font-weight:700;color:{spnl_c};">${short_pnl:+,.0f}</td>
        </tr>
        <tr>
          <td style="padding:3px 8px;color:#555;">Total P&amp;L</td>
          <td style="padding:3px 8px;font-weight:700;font-size:14px;color:{pnl_c};">${pnl:+,.0f} ({pnl_pct:+.1%})</td>
          <td style="padding:3px 8px;color:#555;">Coverage</td>
          <td style="padding:3px 8px;font-weight:700;">{short_coverage:.0f}%</td>
        </tr>
      </table>

      <!-- Roll Planning -->
      <div style="font-size:11px;font-weight:700;color:#666;letter-spacing:1px;
                  margin-bottom:6px;text-transform:uppercase;">Roll Planning — Est. Net Credit</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px;
                    background:#f8fff8;border-radius:4px;">
        <tr style="background:#e8f5e9;">
          <th style="padding:4px 8px;text-align:left;color:#555;">Strike</th>
          <th style="padding:4px 8px;text-align:left;color:#555;">OTM%</th>
          <th style="padding:4px 8px;text-align:left;color:#555;">Est. Credit</th>
          <th style="padding:4px 8px;text-align:left;color:#555;">vs Cap</th>
        </tr>
        <tr>
          <td style="padding:4px 8px;">🟢 ${roll_conservative:.0f}</td>
          <td style="padding:4px 8px;">{(roll_conservative/vix_level-1)*100:.0f}%</td>
          <td style="padding:4px 8px;font-weight:700;">${rd_cons:+.2f}</td>
          <td style="padding:4px 8px;color:{'#4CAF50' if rd_cons>=-debit_cap else '#f44336'};">
              {'✅' if rd_cons>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;">🟡 ${roll_moderate:.0f}</td>
          <td style="padding:4px 8px;">{(roll_moderate/vix_level-1)*100:.0f}%</td>
          <td style="padding:4px 8px;font-weight:700;">${rd_mod:+.2f}</td>
          <td style="padding:4px 8px;color:{'#4CAF50' if rd_mod>=-debit_cap else '#f44336'};">
              {'✅' if rd_mod>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;">🔴 ${roll_aggressive:.0f}</td>
          <td style="padding:4px 8px;">{(roll_aggressive/vix_level-1)*100:.0f}%</td>
          <td style="padding:4px 8px;font-weight:700;">${rd_agg:+.2f}</td>
          <td style="padding:4px 8px;color:{'#4CAF50' if rd_agg>=-debit_cap else '#f44336'};">
              {'✅' if rd_agg>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td>
        </tr>
      </table>

      <!-- Scaling -->
      <div style="font-size:11px;font-weight:700;color:#666;letter-spacing:1px;
                  margin-bottom:4px;text-transform:uppercase;">Scaling</div>
      <div style="font-size:12px;background:{'#e8f5e9' if sc_ok else '#fafafa'};
                  border:1px solid {'#4CAF50' if sc_ok else '#ccc'};border-radius:4px;
                  padding:6px 10px;margin-bottom:10px;">
        {'✅' if sc_ok else '⛔'} {sc_reason}
      </div>

      <!-- Action banner -->
      <div style="background:{ac};color:#fff;padding:10px 14px;border-radius:4px;
                  font-size:13px;font-weight:600;">{at}</div>
    </div>
"""
        html += "  </div>\n"

    # ── SECTION D: STRESS TEST
    if open_diags and stress:
        html += """
  <!-- STRESS TEST -->
  <div style="padding:16px 20px;background:#fff8f8;border-top:1px solid #ffe0e0;">
    <div style="font-size:13px;font-weight:700;color:#c62828;margin-bottom:10px;
                letter-spacing:1px;">📊 STRESS TEST SNAPSHOT</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#ffebee;">
        <th style="padding:6px 10px;text-align:left;color:#555;">UVXY Scenario</th>
        <th style="padding:6px 10px;text-align:left;color:#555;">Est. Net Impact</th>
        <th style="padding:6px 10px;text-align:left;color:#555;">Assessment</th>
      </tr>
"""
        for s in stress:
            imp = s["impact"]
            color = "#4CAF50" if imp >= 0 else "#c62828"
            assess = "✅ Gain" if imp > 500 else "⚠️ Moderate" if imp > -1000 else "🔴 Significant"
            html += f"""
      <tr>
        <td style="padding:5px 10px;">UVXY → ${s['uvxy']}</td>
        <td style="padding:5px 10px;font-weight:700;color:{color};">${imp:+,.0f}</td>
        <td style="padding:5px 10px;">{assess}</td>
      </tr>
"""
        html += "    </table>\n  </div>\n"

    # ── SECTION E: SCALING PERMISSION BLOCK
    html += f"""
  <!-- SCALING BLOCK -->
  <div style="padding:14px 20px;border-top:1px solid #e8e8e8;">
    <div style="font-size:13px;font-weight:700;color:#1a73e8;margin-bottom:10px;
                letter-spacing:1px;">📏 SCALING PERMISSION</div>
    <div style="font-size:11px;color:#666;margin-bottom:8px;">
      Rules: Last 2 cycles profitable · No debit roll · VIX ≤ 25
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
"""
    for state in management_variants:
        variant = state.variant
        name = get_variant_display_name(variant.role)
        diag = None
        for pid, d in tl.diagonal_positions.items():
            if d.variant_id.upper() == variant.role.value.upper() and d.status == "open":
                diag = d; break
        if diag:
            sc_ok2, sc_reason2 = _scaling_eligible(diag)
        else:
            sc_ok2, sc_reason2 = False, "No position"
        html += f"""
      <tr>
        <td style="padding:4px 8px;font-weight:600;">{name}</td>
        <td style="padding:4px 8px;">{'✅' if sc_ok2 else '⛔'} {sc_reason2}</td>
      </tr>
"""
    html += f"""
    </table>
    <div style="font-size:11px;color:#c62828;margin-top:6px;">
      Max drawdown freeze: 3% of account. Current VIX: {vix_level:.1f}
      {'⚠️ SCALING PAUSED — VIX > 25' if vix_level > 25 else ''}
    </div>
  </div>
"""

    # ── SECTION F: FRESH SIGNALS
    html += f"""
  <!-- FRESH SIGNALS -->
  <div style="padding:14px 20px;border-top:2px solid #4CAF50;">
    <div style="font-size:13px;font-weight:700;color:#2e7d32;margin-bottom:4px;
                letter-spacing:1px;">🆕 TODAY'S FRESH SIGNALS — Reference Only</div>
    <div style="font-size:11px;color:#666;margin-bottom:12px;">
      Paper testing only. Use only if scaling permitted. {_short_strike_band(vix_level)} OTM band active.
    </div>
"""
    PAPER_CONTRACTS = 5
    for state in all_variants:
        variant = state.variant
        name = get_variant_display_name(variant.role)
        is_rec     = state.is_recommended
        has_pos    = state.has_position
        entry_cred = state.suggested_entry_credit or 0
        short_offset = getattr(variant, 'short_strike_offset', 2)
        target = round(vix_level + short_offset, 0)
        today  = datetime.now()
        long_expiry  = snap_to_uvxy_expiry((today + timedelta(weeks=variant.long_dte_weeks)).date())
        short_expiry = today + timedelta(weeks=variant.short_dte_weeks)
        days_fri = (4 - short_expiry.weekday()) % 7 or 7
        short_expiry = short_expiry + timedelta(days=days_fri)
        long_exp_str  = long_expiry.strftime("%b %d")
        short_exp_str = short_expiry.strftime("%b %d")
        bc = "#4CAF50" if is_rec else "#9e9e9e"
        bg = "#f1f8e9" if is_rec else "#fafafa"
        badge = ('<span style="background:#4CAF50;color:#fff;padding:2px 7px;'
                 'border-radius:3px;font-size:10px;font-weight:700;">RECOMMENDED</span>'
                 if is_rec else
                 f'<span style="background:#9e9e9e;color:#fff;padding:2px 7px;'
                 f'border-radius:3px;font-size:10px;">PAPER TEST</span>')
        pos_badge = ('<span style="background:#2196F3;color:#fff;padding:2px 7px;'
                     'border-radius:3px;font-size:10px;margin-left:5px;">HAS POSITION</span>'
                     if has_pos else "")
        html += f"""
    <div style="background:{bg};border-left:4px solid {bc};border-radius:4px;
                padding:12px;margin-bottom:10px;">
      <div style="margin-bottom:6px;">
        <span style="font-weight:700;font-size:14px;">{name}</span>
        {pos_badge}
      </div>
      <div style="margin-bottom:8px;">{badge}</div>
      <table style="width:100%;font-size:12px;">
        <tr>
          <td style="padding:3px 0;color:#555;">Long:</td>
          <td>${variant.long_strike:.0f} exp {long_exp_str} ({variant.long_dte_weeks}w)</td>
          <td style="padding:3px 0;color:#555;">Short:</td>
          <td>${target:.0f} exp {short_exp_str}</td>
        </tr>
        <tr style="background:rgba(76,175,80,0.1);">
          <td style="padding:3px 0;color:#555;">Est. Credit:</td>
          <td style="font-weight:700;color:#2e7d32;">${entry_cred:.2f}/c</td>
          <td style="padding:3px 0;color:#555;">Contracts:</td>
          <td>{PAPER_CONTRACTS} (paper)</td>
        </tr>
      </table>
    </div>
"""
    html += "  </div>\n"

    # ── SYSTEM PRINCIPLES footer
    html += """
  <div style="padding:14px 20px;border-top:1px solid #e8e8e8;background:#f8f9ff;">
    <div style="font-size:11px;font-weight:700;color:#666;margin-bottom:6px;">🧠 SYSTEM PRINCIPLES</div>
    <div style="font-size:11px;color:#888;line-height:1.7;">
      Long leg = convexity engine — do not trim &nbsp;·&nbsp;
      Manage diagonal as one structure &nbsp;·&nbsp;
      Use delta &amp; debit triggers, not price stops &nbsp;·&nbsp;
      Scale gradually &nbsp;·&nbsp; Preserve convexity through stress
    </div>
  </div>

  <div style="text-align:center;padding:16px;background:#f0f4ff;color:#9999bb;font-size:11px;">
    📋 VIX 5% Weekly Suite — Paper Trading Mode<br>
    Auto-generated. Do not reply.
  </div>

</div>
</body>
</html>
"""
    return html


def build_real_capital_email(
    batch,
    variant_states,
    real_trade_log=None,
) -> str:
    """
    Separate email for REAL MONEY positions.
    Orange/red theme. Risk-first. Unmistakably live capital.
    """
    from datetime import datetime, timedelta

    fetch_time   = datetime.now()
    regime_state = batch.regime_state
    regime_name  = regime_state.regime.value.upper() if regime_state else "UNKNOWN"
    vix_level    = regime_state.vix_level if regime_state else 20.0
    vix_pct      = regime_state.vix_percentile if regime_state else 0

    ts_data    = _fetch_iv_term_structure()
    stale_warn = _data_age_warning(fetch_time)

    regime_colors = {
        "CALM": "#e65100", "DECLINING": "#bf360c",
        "RISING": "#b71c1c", "STRESSED": "#7f0000", "EXTREME": "#4a0000"
    }
    regime_color = regime_colors.get(regime_name, "#c62828")

    try:
        from real_trade_log import get_real_trade_log
        rtl = real_trade_log or get_real_trade_log()
    except Exception:
        rtl = None

    if not rtl or not rtl.open_positions():
        return ""

    open_pos  = rtl.open_positions()
    all_pos   = list(open_pos.values())
    summary   = rtl.summary()

    total_long_cost   = sum(float(p.long_cost) for p in all_pos)
    total_net_credits = sum(float(p.net_short_credits) for p in all_pos)
    total_pnl         = summary["total_pnl"]
    total_comm        = summary["total_commissions"]
    total_slip        = summary["total_slippage"]

    stress = _stress_test(all_pos, [45, 50, 55], vix_level)

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;background:#1a0800;color:#e0d0c0;padding:20px;margin:0;">
<div style="max-width:680px;margin:0 auto;background:#1f0e00;border-radius:10px;
            overflow:hidden;box-shadow:0 4px 24px rgba(255,80,0,0.3);
            border:2px solid #ff6b35;">

  <!-- LIVE CAPITAL HEADER -->
  <div style="background:linear-gradient(135deg,#cc3300,#ff6b35);color:#fff;
              padding:20px 25px 14px;text-align:center;">
    <div style="font-size:11px;letter-spacing:4px;font-weight:800;
                background:rgba(255,255,0,0.2);display:inline-block;
                padding:3px 12px;border-radius:3px;margin-bottom:8px;">
      💵 LIVE CAPITAL AT RISK
    </div>
    <div style="font-size:22px;font-weight:800;">🔴 VIX 5% Weekly Suite</div>
    <div style="font-size:12px;opacity:0.9;margin-top:6px;">
      Real Money Risk Report — {fetch_time.strftime('%A, %b %d, %Y  %I:%M %p')} ET
    </div>
  </div>
"""

    if stale_warn:
        html += f"""
  <div style="background:#7f0000;border-left:5px solid #ff0000;padding:12px 20px;
              font-size:13px;color:#ffcccc;font-weight:700;">
    ⏱️ {stale_warn}
  </div>
"""

    # Market state — orange palette
    html += f"""
  <!-- MARKET STATE -->
  <div style="background:#2a1200;border-bottom:1px solid #5a2800;padding:16px 20px;">
    <div style="font-size:13px;font-weight:700;color:#ff9944;margin-bottom:10px;
                letter-spacing:1px;">🧭 MARKET STATE</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0c8b0;">
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">UVXY</td>
        <td style="padding:5px 8px;font-weight:700;color:#ffcc88;">${vix_level:.2f}</td>
        <td style="padding:5px 8px;color:#aa8866;">Regime</td>
        <td style="padding:5px 8px;"><span style="background:{regime_color};color:#fff;
            padding:2px 8px;border-radius:3px;font-size:12px;font-weight:700;">
            {regime_name}</span></td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">Percentile</td>
        <td style="padding:5px 8px;font-weight:700;color:#ffcc88;">{vix_pct:.0%}</td>
        <td style="padding:5px 8px;color:#aa8866;">Short Band</td>
        <td style="padding:5px 8px;font-weight:600;color:#ff9944;">
            {_short_strike_band(vix_level)}</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">IV Ratio</td>
        <td style="padding:5px 8px;font-weight:700;color:#ffcc88;">{ts_data['ratio']:.3f}</td>
        <td style="padding:5px 8px;color:#aa8866;">Term</td>
        <td style="padding:5px 8px;color:#ffcc88;">{ts_data['term']}</td>
      </tr>
    </table>
  </div>

  <!-- SYSTEM RISK SUMMARY -->
  <div style="background:#2a1000;border-bottom:1px solid #5a2800;padding:16px 20px;">
    <div style="font-size:13px;font-weight:700;color:#ff9944;margin-bottom:10px;
                letter-spacing:1px;">⚙️ SYSTEM RISK SUMMARY</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0c8b0;">
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">Open Positions</td>
        <td style="padding:5px 8px;font-weight:700;color:#ffcc88;">{len(all_pos)}</td>
        <td style="padding:5px 8px;color:#aa8866;">Long Capital</td>
        <td style="padding:5px 8px;font-weight:700;color:#ffcc88;">${total_long_cost:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">Net Credits</td>
        <td style="padding:5px 8px;font-weight:700;color:#88ff88;">${total_net_credits:+,.0f}</td>
        <td style="padding:5px 8px;color:#aa8866;">Total P&amp;L</td>
        <td style="padding:5px 8px;font-weight:700;
                   color:{'#88ff88' if total_pnl>=0 else '#ff6666'};">${total_pnl:+,.0f}</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#aa8866;">Commissions</td>
        <td style="padding:5px 8px;color:#ffaa66;">${total_comm:.2f}</td>
        <td style="padding:5px 8px;color:#aa8866;">Slippage</td>
        <td style="padding:5px 8px;color:#{'ffaa66' if total_slip<0 else '88ff88'};">${total_slip:+.2f}</td>
      </tr>
    </table>
  </div>
"""

    # Open positions — real money cards
    html += """
  <!-- REAL POSITIONS -->
  <div style="padding:14px 20px 0;">
    <div style="font-size:14px;font-weight:700;color:#ff9944;padding-bottom:8px;
                border-bottom:2px solid #ff6b35;margin-bottom:14px;letter-spacing:1px;">
      💵 OPEN REAL POSITIONS — Risk &amp; Action View
    </div>
"""
    for pos in sorted(all_pos, key=lambda p: p.variant_id):
        short = pos.current_short_leg
        try:
            from datetime import date as _date
            short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days) if short else -1
        except Exception:
            short_dte = 0
        short_delta = _bs_delta(vix_level, short.strike, short_dte/365) if short and short_dte > 0 else (1.0 if short else 0.0)
        long_delta  = _bs_delta(vix_level, pos.long_strike, pos.days_to_long_expiry()/365)
        conv_label, conv_color, conv_emoji = _convexity_status(short_delta)

        v_id = pos.variant_id.upper()
        debit_cap = 1.50 if "V1" in v_id else 1.25 if "V5" in v_id else 1.50

        cur_k = short.strike if short else vix_level
        roll_cons = round(vix_level * 1.02)
        roll_mod  = round(vix_level * 1.05)
        roll_agg  = round(vix_level * 1.10)
        rd_c = _estimate_roll_debit(vix_level, cur_k, roll_cons)
        rd_m = _estimate_roll_debit(vix_level, cur_k, roll_mod)
        rd_a = _estimate_roll_debit(vix_level, cur_k, roll_agg)

        sc_ok, sc_reason = _scaling_eligible(pos)

        long_pnl  = float(pos.long_pnl)
        net_creds = float(pos.net_short_credits)
        tot_pnl   = float(pos.total_pnl)
        coverage  = float(pos.short_coverage_pct)

        _rm_real = _roll_mode(short_delta, short_dte,
                              float(short.strike) if short else 999,
                              vix_level, debit_cap)
        urgency = _rm_real["badge"]
        uc      = _rm_real["color"]

        lpnl_c = "#88ff88" if long_pnl >= 0 else "#ff6666"
        spnl_c = "#88ff88" if net_creds >= 0 else "#ff6666"
        tpnl_c = "#88ff88" if tot_pnl >= 0 else "#ff6666"

        html += f"""
    <div style="border:1px solid #5a2800;border-left:4px solid {uc};border-radius:6px;
                padding:14px;margin-bottom:14px;background:#261000;">

      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px;">
        <div>
          <span style="font-weight:700;font-size:15px;color:#ffcc88;">{pos.variant_name}</span>
          <span style="font-size:11px;color:#888;margin-left:8px;">{pos.broker} · {pos.account_id}</span>
        </div>
        <div style="font-size:12px;background:{uc};color:#fff;padding:3px 10px;
                    border-radius:4px;font-weight:600;">{urgency}</div>
      </div>

      <div style="font-size:11px;font-weight:700;color:#aa8866;letter-spacing:1px;
                  margin-bottom:5px;">STRUCTURE HEALTH</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;
                    background:#2e1500;border-radius:4px;margin-bottom:10px;color:#e0c8b0;">
        <tr>
          <td style="padding:4px 8px;color:#aa8866;">Short Delta</td>
          <td style="padding:4px 8px;font-weight:700;
                     color:{'#ff4444' if short_delta>=0.5 else '#ffaa44' if short_delta>=0.35 else '#88ff88'};">
              {short_delta:.3f}</td>
          <td style="padding:4px 8px;color:#aa8866;">Convexity</td>
          <td style="padding:4px 8px;font-weight:700;">{conv_emoji} {conv_label}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#aa8866;">Long Delta</td>
          <td style="padding:4px 8px;font-weight:700;">{long_delta:.3f}</td>
          <td style="padding:4px 8px;color:#aa8866;">Short DTE</td>
          <td style="padding:4px 8px;font-weight:700;
                     color:{'#ff4444' if short_dte<=0 else '#ffaa44' if short_dte<=2 else '#e0c8b0'};">
              {short_dte}d</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#aa8866;">Long</td>
          <td style="padding:4px 8px;">${pos.long_strike:.0f} exp {pos.long_expiration}</td>
          <td style="padding:4px 8px;color:#aa8866;">Short</td>
          <td style="padding:4px 8px;">${short.strike:.0f} exp {short.expiration_date if short else '—'}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#aa8866;">Roll Trigger</td>
          <td colspan="3" style="padding:4px 8px;font-size:11px;color:#aa8866;">
              Delta ≥ 0.50 &nbsp;|&nbsp; Debit &gt; ${debit_cap:.2f} &nbsp;|&nbsp; DTE ≤ 2 and ITM</td>
        </tr>
      </table>

      <div style="font-size:11px;font-weight:700;color:#aa8866;letter-spacing:1px;
                  margin-bottom:5px;">P&amp;L</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;
                    margin-bottom:10px;color:#e0c8b0;">
        <tr>
          <td style="padding:3px 8px;color:#aa8866;">Long P&amp;L</td>
          <td style="padding:3px 8px;font-weight:700;color:{lpnl_c};">${long_pnl:+,.0f}</td>
          <td style="padding:3px 8px;color:#aa8866;">Short Credits</td>
          <td style="padding:3px 8px;font-weight:700;color:{spnl_c};">${net_creds:+,.0f}</td>
        </tr>
        <tr>
          <td style="padding:3px 8px;color:#aa8866;">Total P&amp;L</td>
          <td style="padding:3px 8px;font-weight:700;font-size:14px;color:{tpnl_c};">${tot_pnl:+,.0f}</td>
          <td style="padding:3px 8px;color:#aa8866;">Coverage</td>
          <td style="padding:3px 8px;font-weight:700;color:#ffcc88;">{coverage:.0f}%</td>
        </tr>
      </table>

      <div style="font-size:11px;font-weight:700;color:#aa8866;letter-spacing:1px;
                  margin-bottom:5px;">ROLL PLANNING — Est. Net Credit</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;
                    background:#2e1500;border-radius:4px;margin-bottom:10px;color:#e0c8b0;">
        <tr style="background:#3a1800;">
          <th style="padding:4px 8px;text-align:left;color:#aa8866;">Strike</th>
          <th style="padding:4px 8px;text-align:left;color:#aa8866;">Est. Credit</th>
          <th style="padding:4px 8px;text-align:left;color:#aa8866;">vs Cap</th>
        </tr>
        <tr><td style="padding:4px 8px;">🟢 ${roll_cons:.0f}</td>
            <td style="padding:4px 8px;font-weight:700;">${rd_c:+.2f}</td>
            <td style="padding:4px 8px;">{'✅' if rd_c>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td></tr>
        <tr><td style="padding:4px 8px;">🟡 ${roll_mod:.0f}</td>
            <td style="padding:4px 8px;font-weight:700;">${rd_m:+.2f}</td>
            <td style="padding:4px 8px;">{'✅' if rd_m>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td></tr>
        <tr><td style="padding:4px 8px;">🔴 ${roll_agg:.0f}</td>
            <td style="padding:4px 8px;font-weight:700;">${rd_a:+.2f}</td>
            <td style="padding:4px 8px;">{'✅' if rd_a>=-debit_cap else '❌'} cap ${debit_cap:.2f}</td></tr>
      </table>

      <div style="font-size:12px;background:{'#1a4a1a' if sc_ok else '#2a1a00'};
                  border:1px solid {'#2e7d32' if sc_ok else '#5a3a00'};border-radius:4px;
                  padding:6px 10px;color:#e0c8b0;">
        Scaling: {'✅' if sc_ok else '⛔'} {sc_reason}
      </div>
    </div>
"""
    html += "  </div>\n"

    # Stress test
    if stress:
        html += """
  <div style="padding:14px 20px;background:#200800;border-top:1px solid #5a2000;">
    <div style="font-size:13px;font-weight:700;color:#ff9944;margin-bottom:10px;">
      📊 STRESS TEST</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0c8b0;">
      <tr style="background:#2e1000;">
        <th style="padding:6px 10px;text-align:left;color:#aa8866;">Scenario</th>
        <th style="padding:6px 10px;text-align:left;color:#aa8866;">Est. Impact</th>
        <th style="padding:6px 10px;text-align:left;color:#aa8866;">Assessment</th>
      </tr>
"""
        for s in stress:
            imp = s["impact"]
            col = "#88ff88" if imp >= 0 else "#ff6666"
            assess = "✅ Gain" if imp > 0 else "⚠️ Moderate" if imp > -1000 else "🔴 Significant loss"
            html += f"""
      <tr>
        <td style="padding:5px 10px;">UVXY → ${s['uvxy']}</td>
        <td style="padding:5px 10px;font-weight:700;color:{col};">${imp:+,.0f}</td>
        <td style="padding:5px 10px;">{assess}</td>
      </tr>
"""
        html += "    </table>\n  </div>\n"

    # Footer
    html += f"""
  <div style="text-align:center;padding:16px;background:#0f0500;
              color:#664422;font-size:11px;border-top:1px solid #3a1800;">
    🔴 VIX 5% Weekly Suite — LIVE CAPITAL MODE<br>
    {fetch_time.strftime('%Y-%m-%d %H:%M ET')} · Auto-generated · Do not reply
  </div>

</div>
</body>
</html>
"""
    return html


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

def snap_to_uvxy_expiry(target_date):
    """
    Snap a theoretical expiration date to the nearest real UVXY option expiry.
    UVXY lists: 3rd Friday monthly + some quarterlies.
    Try to fetch from yfinance, fall back to computed 3rd Fridays.
    """
    import yfinance as yf
    from datetime import date, timedelta
    try:
        tk = yf.Ticker("UVXY")
        exps = [date.fromisoformat(e) for e in tk.options]
        # Find nearest expiry >= target_date
        future = [e for e in exps if e >= target_date]
        if future:
            return min(future, key=lambda e: abs((e - target_date).days))
        return max(exps)  # fallback to furthest available
    except:
        # Fallback: compute 3rd Friday of target month
        d = target_date.replace(day=1)
        fridays = []
        while d.month == target_date.month:
            if d.weekday() == 4:
                fridays.append(d)
            d += timedelta(days=1)
        return fridays[2] if len(fridays) >= 3 else target_date



def build_alert_email(positions_needing_action: list, is_real: bool = False) -> str:
    """
    Build a concise alert email for DEFENSIVE or EMERGENCY roll positions.
    positions_needing_action: list of dicts with position info + roll_mode dict.
    """
    from datetime import datetime
    fetch_time = datetime.now()
    theme_bg   = "#1a0800" if is_real else "#f0f4ff"
    theme_txt  = "#ffcc88" if is_real else "#1a1a2e"
    header_bg  = "#cc3300" if is_real else "#1565c0"
    tag        = "💵 LIVE CAPITAL" if is_real else "📋 PAPER"

    rows = ""
    for p in positions_needing_action:
        rm    = p["roll_mode"]
        rows += f"""
    <tr style="border-bottom:1px solid #333;">
      <td style="padding:8px;font-weight:700;">{p['name']}</td>
      <td style="padding:8px;color:{rm['color']};font-weight:700;">{rm['badge']}</td>
      <td style="padding:8px;">{rm['reason']}</td>
      <td style="padding:8px;font-weight:600;">{rm['action']}</td>
      <td style="padding:8px;">{rm['target_otm']} OTM</td>
      <td style="padding:8px;font-size:11px;color:#888;">{rm['target_note']}</td>
    </tr>"""

    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:{theme_bg};font-family:Arial,sans-serif;">
<div style="max-width:700px;margin:0 auto;padding:20px;">

  <div style="background:{header_bg};color:#fff;padding:16px 20px;border-radius:8px 8px 0 0;">
    <div style="font-size:18px;font-weight:800;">{tag} — ⚡ ROLL ALERT</div>
    <div style="font-size:12px;opacity:0.85;margin-top:4px;">
      {len(positions_needing_action)} position(s) require attention · 
      {fetch_time.strftime('%Y-%m-%d %H:%M ET')}
    </div>
  </div>

  <div style="background:#fff;border:1px solid #ddd;padding:16px;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#f5f5f5;font-weight:700;">
        <th style="padding:8px;text-align:left;">Variant</th>
        <th style="padding:8px;text-align:left;">Mode</th>
        <th style="padding:8px;text-align:left;">Reason</th>
        <th style="padding:8px;text-align:left;">Action</th>
        <th style="padding:8px;text-align:left;">Target</th>
        <th style="padding:8px;text-align:left;">Note</th>
      </tr>
      {rows}
    </table>
  </div>

  <div style="background:#fff3cd;border:1px solid #ffc107;padding:12px;margin-top:12px;
              border-radius:4px;font-size:12px;">
    <strong>Roll Mode Definitions:</strong><br>
    🔄 <b>Routine Roll</b> — DTE ≤ 2, delta &lt; 0.40. Scheduled harvest. Target 4–6% OTM.<br>
    🛡️ <b>Defensive Roll</b> — Delta 0.40–0.49 or DTE ≤ 4 rising. Proactive, credit available. Target 4–6% OTM.<br>
    🚨 <b>Emergency Roll</b> — Delta ≥ 0.50 or ITM. Roll immediately. Target 6–9% OTM. Accept debit if needed.
  </div>

  <div style="text-align:center;padding:12px;color:#888;font-size:11px;margin-top:8px;">
    VIX 5% Weekly Suite · Auto-generated · Do not reply
  </div>

</div></body></html>"""
    return html


def check_and_send_alerts(batch, variant_states, vix_level: float,
                           to_email: str = "") -> list:
    """
    Check all positions for DEFENSIVE or EMERGENCY roll mode.
    Sends alert email if any found. Returns list of alert dicts.
    """
    import os
    from datetime import date as _date

    to_email = to_email or os.environ.get("SMTP_TO", os.environ.get("SMTP_USER", ""))

    paper_alerts = []
    real_alerts  = []

    # ── Check paper positions ──
    try:
        from trade_log import get_trade_log
        tl = get_trade_log()
        for pos in tl.get_open_diagonals():
            short = pos.current_short_leg
            if not short:
                continue
            try:
                short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days)
            except Exception:
                short_dte = 0
            sd = _bs_delta(vix_level, float(short.strike), short_dte/365 if short_dte > 0 else 0.001)
            debit_cap = 1.50
            rm = _roll_mode(sd, short_dte, float(short.strike), vix_level, debit_cap)
            if rm["priority"] >= 2:  # DEFENSIVE or EMERGENCY
                paper_alerts.append({
                    "name":      getattr(pos, "variant_name", pos.variant_id),
                    "roll_mode": rm,
                })
    except Exception as _e:
        print(f"⚠️ Paper alert check: {_e}")

    # ── Check real positions ──
    try:
        from real_trade_log import get_real_trade_log
        rtl = get_real_trade_log()
        for pos in rtl.open_positions().values():
            short = pos.current_short_leg
            if not short:
                continue
            try:
                short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days)
            except Exception:
                short_dte = 0
            sd = _bs_delta(vix_level, float(short.strike), short_dte/365 if short_dte > 0 else 0.001)
            debit_cap = 1.50
            rm = _roll_mode(sd, short_dte, float(short.strike), vix_level, debit_cap)
            if rm["priority"] >= 2:
                real_alerts.append({
                    "name":      pos.variant_name,
                    "roll_mode": rm,
                })
    except Exception as _e:
        print(f"⚠️ Real alert check: {_e}")

    # ── Send alert emails ──
    if paper_alerts:
        try:
            html = build_alert_email(paper_alerts, is_real=False)
            subj = f"⚡ ROLL ALERT [PAPER] — {len(paper_alerts)} position(s) need attention"
            send_email(html, to_email, subj)
            print(f"✅ Paper alert email sent: {len(paper_alerts)} positions")
        except Exception as _e:
            print(f"⚠️ Paper alert send: {_e}")

    if real_alerts:
        try:
            html = build_alert_email(real_alerts, is_real=True)
            subj = f"🚨 ROLL ALERT [LIVE 💵] — {len(real_alerts)} position(s) need attention"
            send_email(html, to_email, subj)
            print(f"✅ Real alert email sent: {len(real_alerts)} positions")
        except Exception as _e:
            print(f"⚠️ Real alert send: {_e}")

    return paper_alerts + real_alerts


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

    # 5b. Build and send real capital email (separate, orange theme)
    try:
        # Fetch live long prices before building real email
        try:
            from real_trade_log import fetch_real_long_prices
            _rtl_live = __import__('real_trade_log').get_real_trade_log()
            _fetched = fetch_real_long_prices(_rtl_live)
            if _fetched:
                print(f"   💰 Live long prices fetched: {_fetched}")
        except Exception as _pxe:
            print(f"   ⚠️ Live price fetch: {_pxe}")
        real_html = build_real_capital_email(batch, variant_states)
        if real_html:
            real_subject = (f"[LIVE 💵] {regime_state.regime.value.upper()} "
                            f"({percentile:.0%}) — Real Capital Risk Report")
            to_addr = os.environ.get("SMTP_TO", os.environ.get("SMTP_USER", ""))
            ok_r = send_email(real_html, to_addr, real_subject)
            print(f"   {'✅' if ok_r else '⚠️'} Real capital email {'sent' if ok_r else 'failed'}")
    except Exception as _re:
        print(f"   ⚠️ Real capital email error: {_re}")

    # 6. Determine subject
    subject = f"[PAPER 📋] {regime_state.regime.value.upper()} ({percentile:.0%}) — "
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
