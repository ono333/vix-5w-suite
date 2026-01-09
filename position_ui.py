#!/usr/bin/env python3
"""
Position Management UI Components for VIX 5% Weekly Suite

Add these to your Streamlit app to enable:
- Trade input (record actual fills)
- Open positions panel (P&L, DTE, actions)
- Performance tracking (win rate, P&L by variant)
- Signal suppression (hide entries when position exists)

Usage in app.py:
    from position_ui import (
        render_trade_input,
        render_open_positions,
        render_performance_summary,
        get_entry_suppressed_variants,
    )
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

# Import trade log (adjust path as needed)
try:
    from trade_log import get_trade_log, TradeLog, Position
except ImportError:
    st.error("trade_log.py not found. Please install the position-aware system first.")
    raise

try:
    from enums import VolatilityRegime, VariantRole
    from variant_generator import get_variant_display_name
except ImportError:
    # Fallback if enums not available
    VolatilityRegime = None
    VariantRole = None
    def get_variant_display_name(role): return str(role)


# ============================================================
# Constants
# ============================================================

VARIANT_OPTIONS = [
    ("V1_INCOME_HARVESTER", "V1 Income Harvester"),
    ("V2_MEAN_REVERSION", "V2 Mean Reversion"),
    ("V3_SHOCK_ABSORBER", "V3 Shock Absorber"),
    ("V4_TAIL_HUNTER", "V4 Tail Hunter"),
    ("V5_REGIME_ALLOCATOR", "V5 Regime Allocator"),
]

REGIME_OPTIONS = ["CALM", "DECLINING", "RISING", "STRESSED", "EXTREME"]

EXIT_REASONS = [
    "target_hit",
    "stop_hit", 
    "manual_close",
    "rolled",
    "expired_worthless",
    "expired_itm",
]


# ============================================================
# Trade Input UI
# ============================================================

def render_trade_input(
    current_regime: str = "CALM",
    current_vix: float = 20.0,
    current_percentile: float = 0.5,
) -> Optional[Position]:
    """
    Render trade input form for recording actual fills.
    
    Returns the newly created Position if successful, None otherwise.
    """
    st.markdown("### 📝 Record New Trade")
    
    trade_log = get_trade_log()
    
    # Check which variants already have positions
    open_variants = set(trade_log.get_variants_with_open_positions())
    
    with st.expander("Enter Trade Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Variant selection (filter out those with open positions)
            available_variants = [
                (vid, name) for vid, name in VARIANT_OPTIONS 
                if vid not in open_variants
            ]
            
            if not available_variants:
                st.warning("⚠️ All variants have open positions. Close one first.")
                return None
            
            variant_id = st.selectbox(
                "Variant",
                options=[v[0] for v in available_variants],
                format_func=lambda x: dict(VARIANT_OPTIONS).get(x, x),
                key="trade_input_variant",
            )
            
            entry_credit = st.number_input(
                "Entry Credit Received ($)",
                min_value=0.01,
                max_value=50.0,
                value=1.50,
                step=0.05,
                help="The credit you received when opening the position",
                key="trade_input_credit",
            )
            
            contracts = st.number_input(
                "Contracts",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                key="trade_input_contracts",
            )
            
            strike = st.number_input(
                "Strike Price",
                min_value=1.0,
                max_value=200.0,
                value=round(current_vix + 5, 0),
                step=0.5,
                key="trade_input_strike",
            )
        
        with col2:
            entry_date = st.date_input(
                "Entry Date",
                value=date.today(),
                key="trade_input_date",
            )
            
            # Calculate default expiration (Friday of next month)
            default_exp = date.today() + timedelta(days=30)
            while default_exp.weekday() != 4:  # Find Friday
                default_exp += timedelta(days=1)
            
            expiration = st.date_input(
                "Expiration Date",
                value=default_exp,
                key="trade_input_expiration",
            )
            
            target_pct = st.slider(
                "Target % (profit to close)",
                min_value=10,
                max_value=80,
                value=40,
                step=5,
                key="trade_input_target",
            ) / 100.0
            
            stop_pct = st.slider(
                "Stop % (loss to close)",
                min_value=20,
                max_value=100,
                value=60,
                step=5,
                key="trade_input_stop",
            ) / 100.0
        
        # Calculated values
        target_price = round(entry_credit * (1 - target_pct), 2)
        stop_price = round(entry_credit * (1 + stop_pct), 2)
        dte = (expiration - date.today()).days
        
        st.markdown("#### Calculated Values")
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        calc_col1.metric("Target Exit", f"${target_price:.2f}", f"{target_pct:.0%} gain")
        calc_col2.metric("Stop Loss", f"${stop_price:.2f}", f"{stop_pct:.0%} loss")
        calc_col3.metric("DTE", f"{dte} days")
        
        notes = st.text_input(
            "Notes (optional)",
            key="trade_input_notes",
        )
        
        # Submit button
        if st.button("📥 Record Trade", type="primary", key="trade_input_submit"):
            try:
                variant_name = dict(VARIANT_OPTIONS).get(variant_id, variant_id)
                
                position = trade_log.open_position(
                    variant_id=variant_id,
                    variant_name=variant_name,
                    entry_price=entry_credit,
                    entry_regime=current_regime,
                    entry_vix_level=current_vix,
                    entry_percentile=current_percentile,
                    strike=strike,
                    expiration_date=expiration.isoformat(),
                    contracts=contracts,
                    target_pct=target_pct,
                    stop_pct=stop_pct,
                    notes=notes,
                )
                
                st.success(f"✅ Recorded {variant_name} position!")
                st.rerun()
                return position
                
            except ValueError as e:
                st.error(f"❌ Error: {e}")
                return None
    
    return None


# ============================================================
# Open Positions Panel
# ============================================================

def render_open_positions(
    current_regime: str = "CALM",
) -> None:
    """
    Render panel showing all open positions with P&L, DTE, and actions.
    """
    trade_log = get_trade_log()
    positions = trade_log.get_all_open_positions()
    
    st.markdown("### 🔄 Open Positions")
    
    if not positions:
        st.info("No open positions. Use the form above to record trades.")
        return
    
    for pos in positions:
        dte = pos.days_to_expiry()
        
        # Determine card style based on P&L
        if pos.current_pnl_pct > 0.05:
            border_color = "#4CAF50"  # Green
            status_emoji = "🟢"
        elif pos.current_pnl_pct < -0.05:
            border_color = "#f44336"  # Red
            status_emoji = "🔴"
        else:
            border_color = "#2196F3"  # Blue
            status_emoji = "🔵"
        
        # Check for regime drift
        regime_drift = pos.entry_regime != current_regime
        
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: #1e1e2e;
                    border-left: 4px solid {border_color};
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                ">
                    <div style="font-weight: 600; font-size: 16px; margin-bottom: 10px;">
                        {status_emoji} {pos.variant_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Entry Credit",
                    f"${pos.entry_price:.2f}",
                    f"{pos.contracts} contracts",
                )
            
            with col2:
                pnl_delta = f"{pos.current_pnl_pct:+.1%}" if pos.current_pnl_pct else None
                st.metric(
                    "Current P&L",
                    f"${pos.current_pnl:+,.0f}",
                    pnl_delta,
                )
            
            with col3:
                dte_warning = "⚠️" if dte <= 7 else ""
                st.metric(
                    "DTE Remaining",
                    f"{dte} days {dte_warning}",
                )
            
            with col4:
                st.metric(
                    "Targets",
                    f"TP: ${pos.target_price:.2f}",
                    f"SL: ${pos.stop_price:.2f}",
                )
            
            # Action suggestions
            action_col1, action_col2 = st.columns([3, 1])
            
            with action_col1:
                # Determine suggested action
                if pos.current_pnl_pct >= pos.target_pct:
                    st.success("🎯 **TAKE PROFIT** — Target reached!")
                elif pos.current_pnl_pct <= -pos.stop_pct:
                    st.error("🛑 **STOP LOSS** — Stop level hit!")
                elif dte <= 5:
                    st.warning("📅 **ROLL or CLOSE** — Low DTE")
                elif regime_drift:
                    st.warning(f"⚠️ **REGIME DRIFT** — Opened in {pos.entry_regime}, now {current_regime}")
                else:
                    st.info("✋ **HOLD** — On track")
            
            with action_col2:
                if st.button("Close Position", key=f"close_{pos.position_id}"):
                    st.session_state[f"closing_{pos.variant_id}"] = True
            
            # Close position form
            if st.session_state.get(f"closing_{pos.variant_id}"):
                with st.form(key=f"close_form_{pos.variant_id}"):
                    st.markdown("#### Close Position")
                    
                    close_col1, close_col2 = st.columns(2)
                    with close_col1:
                        exit_price = st.number_input(
                            "Exit Price ($)",
                            min_value=0.0,
                            max_value=50.0,
                            value=pos.current_price if pos.current_price > 0 else pos.entry_price * 0.6,
                            step=0.05,
                        )
                    
                    with close_col2:
                        exit_reason = st.selectbox(
                            "Exit Reason",
                            options=EXIT_REASONS,
                        )
                    
                    submit_col1, submit_col2 = st.columns(2)
                    with submit_col1:
                        if st.form_submit_button("✅ Confirm Close", type="primary"):
                            record = trade_log.close_position(
                                variant_id=pos.variant_id,
                                exit_price=exit_price,
                                exit_reason=exit_reason,
                                exit_regime=current_regime,
                            )
                            if record:
                                st.success(f"Position closed! P&L: ${record.pnl_dollars:+,.0f}")
                                del st.session_state[f"closing_{pos.variant_id}"]
                                st.rerun()
                    
                    with submit_col2:
                        if st.form_submit_button("❌ Cancel"):
                            del st.session_state[f"closing_{pos.variant_id}"]
                            st.rerun()
            
            st.markdown("---")


def render_update_prices() -> None:
    """
    Render form to update current prices for open positions.
    """
    trade_log = get_trade_log()
    positions = trade_log.get_all_open_positions()
    
    if not positions:
        return
    
    with st.expander("📊 Update Current Prices"):
        st.markdown("Enter current market prices to update P&L calculations.")
        
        for pos in positions:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text(f"{pos.variant_name} (Entry: ${pos.entry_price:.2f})")
            
            with col2:
                new_price = st.number_input(
                    "Current Price",
                    min_value=0.0,
                    max_value=50.0,
                    value=pos.current_price if pos.current_price > 0 else pos.entry_price,
                    step=0.05,
                    key=f"price_{pos.variant_id}",
                    label_visibility="collapsed",
                )
                
                if new_price != pos.current_price:
                    trade_log.update_position_price(pos.variant_id, new_price)


# ============================================================
# Performance Summary
# ============================================================

def render_performance_summary() -> None:
    """
    Render performance tracking with win rate, P&L by variant, etc.
    """
    trade_log = get_trade_log()
    summary = trade_log.get_summary()
    history = trade_log.history
    
    st.markdown("### 📈 Performance Summary")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Total Trades",
        summary["total_trades"],
    )
    
    col2.metric(
        "Win Rate",
        f"{summary['win_rate']:.0%}",
        f"{summary['wins']}W / {summary['losses']}L",
    )
    
    col3.metric(
        "Realized P&L",
        f"${summary['total_realized_pnl']:+,.0f}",
    )
    
    col4.metric(
        "Open P&L",
        f"${summary['open_pnl']:+,.0f}",
        f"{summary['open_positions']} positions",
    )
    
    # Per-variant breakdown
    if history:
        st.markdown("#### By Variant")
        
        variant_stats = {}
        for record in history:
            vid = record.variant_id
            if vid not in variant_stats:
                variant_stats[vid] = {
                    "name": record.variant_name,
                    "trades": 0,
                    "wins": 0,
                    "pnl": 0.0,
                }
            variant_stats[vid]["trades"] += 1
            variant_stats[vid]["pnl"] += record.pnl_dollars
            if record.pnl_dollars > 0:
                variant_stats[vid]["wins"] += 1
        
        # Display as dataframe
        df_data = []
        for vid, stats in variant_stats.items():
            win_rate = stats["wins"] / max(1, stats["trades"])
            df_data.append({
                "Variant": stats["name"],
                "Trades": stats["trades"],
                "Win Rate": f"{win_rate:.0%}",
                "Total P&L": f"${stats['pnl']:+,.0f}",
                "Avg P&L": f"${stats['pnl'] / max(1, stats['trades']):+,.0f}",
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Trade history table
    if history:
        with st.expander("📋 Trade History"):
            history_data = []
            for record in reversed(history):  # Most recent first
                history_data.append({
                    "Date": record.exit_date[:10],
                    "Variant": record.variant_name,
                    "Entry": f"${record.entry_price:.2f}",
                    "Exit": f"${record.exit_price:.2f}",
                    "P&L": f"${record.pnl_dollars:+,.0f}",
                    "Duration": f"{record.duration_days}d",
                    "Reason": record.exit_reason,
                    "Regime": f"{record.entry_regime}→{record.exit_regime}",
                })
            
            if history_data:
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Signal Suppression Helper
# ============================================================

def get_entry_suppressed_variants() -> List[str]:
    """
    Get list of variant IDs that should NOT show entry signals
    because they already have open positions.
    
    Use this to filter variant displays in your signal UI.
    """
    trade_log = get_trade_log()
    return trade_log.get_variants_with_open_positions()


def should_suppress_entry(variant_id: str) -> bool:
    """
    Check if a specific variant should have its entry signal suppressed.
    """
    return variant_id in get_entry_suppressed_variants()


# ============================================================
# Full Position Management Page
# ============================================================

def render_position_management_page(
    current_regime: str = "CALM",
    current_vix: float = 20.0,
    current_percentile: float = 0.5,
) -> None:
    """
    Render complete position management page with all components.
    
    Add this to your app.py page routing:
    
        if page == "Position Manager":
            render_position_management_page(
                current_regime=regime_state.regime.value,
                current_vix=current_price,
                current_percentile=percentile,
            )
    """
    st.title("📊 Position Manager")
    
    st.markdown(f"""
    **Current Market:** {current_regime} regime | UVXY ${current_vix:.2f} | {current_percentile:.0%} percentile
    """)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🔄 Open Positions", "📝 Record Trade", "📈 Performance"])
    
    with tab1:
        render_open_positions(current_regime)
        render_update_prices()
    
    with tab2:
        render_trade_input(current_regime, current_vix, current_percentile)
    
    with tab3:
        render_performance_summary()
