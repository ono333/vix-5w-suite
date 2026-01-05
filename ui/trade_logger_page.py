#!/usr/bin/env python3
"""
Trade Logger Page for VIX 5% Weekly Suite

Manual paper trade entry form for collecting data across 5 strategy variants.
Each variant's trades are logged separately for future analysis.

Variants:
    V1: Static Baseline
    V2: Regime Adaptive  
    V3: Aggressive Entry
    V4: Conservative
    V5: High VIX Contrarian

Usage:
    Import and call render_trade_logger_page() from your app.py
"""

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================

VARIANTS = {
    1: {"name": "Static Baseline", "color": "#4CAF50"},
    2: {"name": "Regime Adaptive", "color": "#2196F3"},
    3: {"name": "Aggressive Entry", "color": "#FF9800"},
    4: {"name": "Conservative", "color": "#9C27B0"},
    5: {"name": "High VIX Contrarian", "color": "#F44336"},
}

# Default log file path (in project root)
DEFAULT_LOG_PATH = Path(__file__).parent / "paper_trade_log.json"


# =============================================================================
# Data Storage Functions
# =============================================================================

def _load_trades(log_path: Path = DEFAULT_LOG_PATH) -> List[Dict]:
    """Load trades from JSON file."""
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        return data.get("trades", [])
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return []


def _save_trades(trades: List[Dict], log_path: Path = DEFAULT_LOG_PATH):
    """Save trades to JSON file."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"trades": trades, "updated": dt.datetime.now().isoformat()}, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving trades: {e}")


def _generate_trade_id(variant_id: int) -> str:
    """Generate unique trade ID."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"V{variant_id}_{timestamp}"


# =============================================================================
# Trade Logger UI
# =============================================================================

def render_trade_logger_page(log_path: Optional[Path] = None):
    """
    Render the Trade Logger page with manual entry form.
    
    Call this from your app.py when page == "Trade Logger"
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    
    st.title("📝 Paper Trade Logger")
    st.caption("Log paper trades for 5 strategy variants")
    
    # Load existing trades
    all_trades = _load_trades(log_path)
    
    # Tabs for different actions
    tab_entry, tab_view, tab_analysis, tab_manage = st.tabs([
        "➕ New Entry", "📋 View Trades", "📊 Analysis", "⚙️ Manage"
    ])
    
    # -----------------------------------------------------------------
    # TAB 1: New Trade Entry
    # -----------------------------------------------------------------
    with tab_entry:
        st.subheader("Log New Trade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variant selection
            variant_id = st.selectbox(
                "Strategy Variant",
                options=list(VARIANTS.keys()),
                format_func=lambda x: f"V{x}: {VARIANTS[x]['name']}",
                key="entry_variant",
            )
            
            # Trade status
            trade_status = st.radio(
                "Trade Status",
                options=["Open (Entry)", "Closed (Exit)"],
                horizontal=True,
                key="entry_status",
            )
            
            is_entry = trade_status == "Open (Entry)"
        
        with col2:
            # Date
            trade_date = st.date_input(
                "Trade Date",
                value=dt.date.today(),
                key="entry_date",
            )
            
            # Underlying price
            underlying_price = st.number_input(
                "Underlying Price (VIX/UVXY)",
                min_value=0.0,
                value=15.0,
                step=0.1,
                format="%.2f",
                key="entry_underlying",
            )
        
        st.markdown("---")
        
        if is_entry:
            # ENTRY FORM
            st.markdown("#### Entry Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                contracts = st.number_input(
                    "Contracts",
                    min_value=1,
                    value=1,
                    step=1,
                    key="entry_contracts",
                )
                
                long_strike = st.number_input(
                    "Long Strike",
                    min_value=0.0,
                    value=20.0,
                    step=0.5,
                    format="%.1f",
                    key="entry_long_strike",
                )
            
            with col2:
                long_premium = st.number_input(
                    "Long Premium (per contract)",
                    min_value=0.0,
                    value=2.50,
                    step=0.05,
                    format="%.2f",
                    key="entry_long_premium",
                )
                
                long_dte = st.number_input(
                    "Long DTE (days)",
                    min_value=1,
                    value=180,
                    step=1,
                    key="entry_long_dte",
                )
            
            with col3:
                short_strike = st.number_input(
                    "Short Strike (0 if long-only)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f",
                    key="entry_short_strike",
                )
                
                short_premium = st.number_input(
                    "Short Premium (per contract)",
                    min_value=0.0,
                    value=0.0,
                    step=0.05,
                    format="%.2f",
                    key="entry_short_premium",
                )
            
            # Calculate costs
            total_debit = (long_premium - short_premium) * contracts * 100
            st.metric("Net Debit", f"${total_debit:,.2f}")
            
            # VIX percentile at entry
            entry_percentile = st.slider(
                "VIX Percentile at Entry",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                key="entry_percentile",
                help="52-week percentile of VIX at time of entry",
            )
            
            # Notes
            notes = st.text_area("Notes", key="entry_notes", height=80)
            
            # Submit button
            if st.button("📥 Log Entry", type="primary", key="submit_entry"):
                trade = {
                    "trade_id": _generate_trade_id(variant_id),
                    "variant_id": variant_id,
                    "variant_name": VARIANTS[variant_id]["name"],
                    "status": "open",
                    "entry_date": str(trade_date),
                    "entry_underlying": underlying_price,
                    "contracts": contracts,
                    "long_strike": long_strike,
                    "long_premium": long_premium,
                    "long_dte": long_dte,
                    "short_strike": short_strike if short_strike > 0 else None,
                    "short_premium": short_premium if short_strike > 0 else None,
                    "net_debit": total_debit,
                    "entry_percentile": entry_percentile,
                    "notes": notes,
                    "exit_date": None,
                    "exit_underlying": None,
                    "exit_value": None,
                    "pnl": None,
                    "exit_reason": None,
                }
                
                all_trades.append(trade)
                _save_trades(all_trades, log_path)
                st.success(f"✅ Entry logged: {trade['trade_id']}")
                st.rerun()
        
        else:
            # EXIT FORM
            st.markdown("#### Exit Details")
            
            # Get open trades for this variant
            open_trades = [t for t in all_trades if t.get("status") == "open" and t.get("variant_id") == variant_id]
            
            if not open_trades:
                st.warning(f"No open trades for V{variant_id}: {VARIANTS[variant_id]['name']}")
            else:
                # Select trade to close
                trade_options = {t["trade_id"]: f"{t['trade_id']} | Entry: {t['entry_date']} | Strike: {t['long_strike']}" for t in open_trades}
                selected_trade_id = st.selectbox(
                    "Select Trade to Close",
                    options=list(trade_options.keys()),
                    format_func=lambda x: trade_options[x],
                    key="exit_select_trade",
                )
                
                selected_trade = next((t for t in open_trades if t["trade_id"] == selected_trade_id), None)
                
                if selected_trade:
                    st.info(f"Entry: {selected_trade['entry_date']} | Contracts: {selected_trade['contracts']} | Net Debit: ${selected_trade['net_debit']:,.2f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        exit_value = st.number_input(
                            "Exit Value (total $)",
                            min_value=0.0,
                            value=float(selected_trade["net_debit"]),
                            step=10.0,
                            format="%.2f",
                            key="exit_value",
                        )
                    
                    with col2:
                        exit_reason = st.selectbox(
                            "Exit Reason",
                            options=["Target Hit", "Stop Loss", "Expiry", "Manual", "Roll"],
                            key="exit_reason",
                        )
                    
                    # Calculate P&L
                    pnl = exit_value - selected_trade["net_debit"]
                    pnl_pct = (pnl / selected_trade["net_debit"] * 100) if selected_trade["net_debit"] > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("P&L ($)", f"${pnl:+,.2f}")
                    col2.metric("P&L (%)", f"{pnl_pct:+.1f}%")
                    
                    exit_notes = st.text_area("Exit Notes", key="exit_notes", height=80)
                    
                    if st.button("📤 Log Exit", type="primary", key="submit_exit"):
                        # Update the trade
                        for t in all_trades:
                            if t["trade_id"] == selected_trade_id:
                                t["status"] = "closed"
                                t["exit_date"] = str(trade_date)
                                t["exit_underlying"] = underlying_price
                                t["exit_value"] = exit_value
                                t["pnl"] = pnl
                                t["pnl_pct"] = pnl_pct
                                t["exit_reason"] = exit_reason
                                t["notes"] = f"{t.get('notes', '')} | Exit: {exit_notes}".strip(" |")
                                break
                        
                        _save_trades(all_trades, log_path)
                        st.success(f"✅ Exit logged for {selected_trade_id}")
                        st.rerun()
    
    # -----------------------------------------------------------------
    # TAB 2: View Trades
    # -----------------------------------------------------------------
    with tab_view:
        st.subheader("Trade Log")
        
        # Filter by variant
        filter_variant = st.selectbox(
            "Filter by Variant",
            options=[0] + list(VARIANTS.keys()),
            format_func=lambda x: "All Variants" if x == 0 else f"V{x}: {VARIANTS[x]['name']}",
            key="view_filter_variant",
        )
        
        # Filter by status
        filter_status = st.radio(
            "Status",
            options=["All", "Open", "Closed"],
            horizontal=True,
            key="view_filter_status",
        )
        
        # Apply filters
        filtered = all_trades
        if filter_variant > 0:
            filtered = [t for t in filtered if t.get("variant_id") == filter_variant]
        if filter_status == "Open":
            filtered = [t for t in filtered if t.get("status") == "open"]
        elif filter_status == "Closed":
            filtered = [t for t in filtered if t.get("status") == "closed"]
        
        if not filtered:
            st.info("No trades found.")
        else:
            # Display as table
            display_data = []
            for t in filtered:
                display_data.append({
                    "ID": t.get("trade_id", ""),
                    "Variant": f"V{t.get('variant_id')}",
                    "Status": t.get("status", "").upper(),
                    "Entry": t.get("entry_date", ""),
                    "Exit": t.get("exit_date", "-"),
                    "Strike": t.get("long_strike", 0),
                    "Contracts": t.get("contracts", 0),
                    "Net Debit": f"${t.get('net_debit', 0):,.0f}",
                    "P&L": f"${t.get('pnl', 0):+,.0f}" if t.get("pnl") is not None else "-",
                    "Reason": t.get("exit_reason", "-"),
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f"paper_trades_{dt.date.today()}.csv",
                mime="text/csv",
            )
    
    # -----------------------------------------------------------------
    # TAB 3: Analysis
    # -----------------------------------------------------------------
    with tab_analysis:
        st.subheader("Variant Comparison")
        
        closed_trades = [t for t in all_trades if t.get("status") == "closed"]
        
        if not closed_trades:
            st.info("No closed trades yet. Complete some trades to see analysis.")
        else:
            # Stats by variant
            stats = []
            for v_id, v_info in VARIANTS.items():
                v_trades = [t for t in closed_trades if t.get("variant_id") == v_id]
                if v_trades:
                    wins = [t for t in v_trades if t.get("pnl", 0) > 0]
                    total_pnl = sum(t.get("pnl", 0) for t in v_trades)
                    stats.append({
                        "Variant": f"V{v_id}: {v_info['name']}",
                        "Trades": len(v_trades),
                        "Wins": len(wins),
                        "Win Rate": f"{len(wins)/len(v_trades)*100:.0f}%",
                        "Total P&L": f"${total_pnl:+,.0f}",
                        "Avg P&L": f"${total_pnl/len(v_trades):+,.0f}",
                    })
            
            if stats:
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
                
                # P&L by variant chart
                st.markdown("#### Total P&L by Variant")
                pnl_data = {f"V{v_id}": sum(t.get("pnl", 0) for t in closed_trades if t.get("variant_id") == v_id) 
                           for v_id in VARIANTS}
                st.bar_chart(pnl_data)
    
    # -----------------------------------------------------------------
    # TAB 4: Manage
    # -----------------------------------------------------------------
    with tab_manage:
        st.subheader("Manage Trades")
        
        st.markdown(f"**Log file:** `{log_path}`")
        st.markdown(f"**Total trades:** {len(all_trades)}")
        
        # Delete a trade
        st.markdown("#### Delete Trade")
        if all_trades:
            trade_to_delete = st.selectbox(
                "Select trade to delete",
                options=[t["trade_id"] for t in all_trades],
                key="delete_select",
            )
            
            if st.button("🗑️ Delete Trade", type="secondary"):
                all_trades = [t for t in all_trades if t["trade_id"] != trade_to_delete]
                _save_trades(all_trades, log_path)
                st.success(f"Deleted {trade_to_delete}")
                st.rerun()
        
        st.markdown("---")
        
        # Clear all trades
        st.markdown("#### Danger Zone")
        if st.button("🗑️ Clear ALL Trades", type="secondary"):
            st.warning("Are you sure? This cannot be undone.")
            if st.button("Yes, clear all", key="confirm_clear"):
                _save_trades([], log_path)
                st.success("All trades cleared.")
                st.rerun()


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Trade Logger", layout="wide")
    render_trade_logger_page()
