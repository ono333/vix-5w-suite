#!/usr/bin/env python3
"""
VIX 5% Weekly Suite – MAIN APP (Regime-Adaptive Version)

Features:
- Dashboard: Overview with regime timeline visualization
- Backtester: Standard backtest with grid scan
- Adaptive Backtester: Regime-aware dynamic parameter switching
- Per-Regime Optimizer: Optimize parameters separately for each regime
- Live Signals: Real-time regime detection and executable diagonal legs
- Trade Explorer: Detailed trade analysis

Regime System:
- ULTRA_LOW (0-10%): Extremely calm, best entry conditions
- LOW (10-25%): Primary entry zone
- MEDIUM (25-50%): Normal volatility, moderate positioning
- HIGH (50-75%): Elevated volatility, defensive
- EXTREME (75-100%): Crisis/spike, avoid new positions
"""

import io
import os
import datetime as dt
from datetime import date, timedelta
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

def format_html_report(data):
    """Compact HTML email with bid/ask data for trading."""
    today = dt.date.today().strftime("%b %d, %Y")
    pct = data['percentile']
    active = data['signal_active']
    emoji = "🟢" if active else "🔴"
    status = "ENTRY SIGNAL" if active else "HOLD"
    status_color = "#2e7d32" if active else "#c62828"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:12px;background:#fff;color:#333;padding:10px;max-width:700px;margin:auto;">
    
    <div style="text-align:center;border-bottom:2px solid #1f77b4;padding-bottom:8px;margin-bottom:12px;">
        <span style="font-size:18px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="color:#666;font-size:11px;">Thursday Signal • {today}</span>
    </div>

    <table style="width:100%;border-collapse:collapse;margin-bottom:10px;font-size:11px;">
        <tr>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>VIX</b><br>${data['vix_close']:.2f}</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>Percentile</b><br>{pct:.1f}%</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>Regime</b><br>{data['regime']}</td>
            <td style="padding:6px;background:#f5f5f5;border:1px solid #ddd;width:25%;"><b>UVXY</b><br>${data['uvxy_spot']:.2f}</td>
        </tr>
    </table>

    <div style="padding:8px;background:{'#e8f5e9' if active else '#ffebee'};border:1px solid {status_color};text-align:center;margin-bottom:12px;border-radius:4px;">
        <b style="color:{status_color};font-size:14px;">{emoji} {status}</b>
    </div>

    <div style="font-size:13px;font-weight:bold;color:#1f77b4;margin-bottom:8px;">5 DIAGONAL VARIANTS</div>
    """

    for v in data['variants']:
        # Handle both old and new data formats
        long_strike = v.get('long_strike', 0)
        long_exp = v.get('long_exp', v.get('long_expiry', '-'))
        long_dte = v.get('long_dte', '-')
        long_bid = v.get('long_bid', 0)
        long_ask = v.get('long_ask', 0)
        long_mid = v.get('long_mid', v.get('long_price', 0))
        
        short_strike = v.get('short_strike', 0)
        short_exp = v.get('short_exp', v.get('short_expiry', '-'))
        short_dte = v.get('short_dte', '-')
        short_bid = v.get('short_bid', 0)
        short_ask = v.get('short_ask', 0)
        short_mid = v.get('short_mid', v.get('short_price', 0))
        
        net_debit = v.get('net_debit', 0)
        risk = v.get('risk_per_contract', abs(net_debit) * 100)
        target_mult = v.get('target_mult', 1.2)
        target_price = v.get('target_price', long_mid * target_mult)
        stop_mult = v.get('stop_mult', 0.5)
        stop_price = v.get('stop_price', long_mid * stop_mult)
        suggested = v.get('suggested_contracts', v.get('suggested', 1))
        
        html += f"""
        <div style="border:1px solid #ddd;margin-bottom:8px;border-radius:4px;overflow:hidden;">
            <div style="background:#1f77b4;color:#fff;padding:5px 8px;font-size:11px;font-weight:bold;">
                {v['name']} <span style="font-weight:normal;opacity:0.8;">— {v.get('desc', '')[:30]}</span>
            </div>
            <div style="padding:6px;font-size:11px;">
                <table style="width:100%;border-collapse:collapse;margin-bottom:4px;">
                    <tr style="background:#f8f8f8;">
                        <th style="padding:4px;border:1px solid #eee;text-align:left;width:18%;">Leg</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:12%;">Strike</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:22%;">Expiry</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:10%;">DTE</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:12%;">Bid</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:12%;">Ask</th>
                        <th style="padding:4px;border:1px solid #eee;text-align:center;width:14%;">Mid</th>
                    </tr>
                    <tr>
                        <td style="padding:4px;border:1px solid #eee;color:#c62828;font-weight:bold;">SHORT</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${short_strike:.0f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">{short_exp}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">{short_dte}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${short_bid:.2f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${short_ask:.2f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;font-weight:bold;">${short_mid:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px;border:1px solid #eee;color:#2e7d32;font-weight:bold;">LONG</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${long_strike:.0f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">{long_exp}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">{long_dte}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${long_bid:.2f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;">${long_ask:.2f}</td>
                        <td style="padding:4px;border:1px solid #eee;text-align:center;font-weight:bold;">${long_mid:.2f}</td>
                    </tr>
                </table>
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#555;padding-top:4px;border-top:1px solid #eee;">
                    <span><b>Net:</b> ${net_debit:.2f} | <b>Risk:</b> ${risk:.0f}/ct</span>
                    <span><b>Target:</b> ${target_price:.2f} ({target_mult}x) | <b>Stop:</b> ${stop_price:.2f} ({stop_mult}x)</span>
                    <span><b>Suggested:</b> {suggested} ct</span>
                </div>
            </div>
        </div>
        """

    html += """
    <div style="font-size:10px;color:#888;text-align:center;margin-top:12px;padding-top:8px;border-top:1px solid #eee;">
        ⚠️ Research only — verify quotes with broker before trading.
    </div>
    </body>
    </html>
    """
    return html

# Core imports
from ui.sidebar import build_sidebar
from core.data_loader import load_vix_weekly, load_weekly

# Backtesters
from core.backtester import run_backtest
from core.adaptive_backtester import run_adaptive_backtest, compare_adaptive_vs_static

# Try to import Massive backtester (optional)
try:
    from core.backtester_massive import run_backtest_massive
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    run_backtest_massive = None

# Regime system
from core.regime_adapter import RegimeAdapter, REGIME_CONFIGS, create_regime_timeline_df

# Grid scans
from experiments.grid_scan import run_grid_scan
from experiments.per_regime_grid_scan import (
    run_per_regime_grid_scan,
    create_regime_comparison_df,
)

# Param history
from core.param_history import (
    apply_best_if_requested,
    get_best_for_strategy,
    get_all_regime_params,
    get_history_summary,
    get_best_for_regime,
    get_all_profiles,
    get_profile,
    save_profile,
    reset_profile_to_default,
    get_profile_summary_df,
    export_all_profiles,
    import_profiles,
    DEFAULT_PROFILES,
    REGIME_NAMES,
)

# =============================================================================
# 5 PAPER TRADING VARIANTS - Parallel Comparison
# =============================================================================
PAPER_VARIANTS = {
    "Existing Conservative": {
        "description": "Baseline — DTE 20-26 weeks, entry 0.1-0.2, OTM 8-12",
        "long_dte_weeks": 26,
        "entry_percentile": 0.15,
        "otm_pts": 10,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "use_regime": True,
    },
    "DTE 1 Week Aggressive": {
        "description": "Ultra-short theta burn — max decay in contango",
        "long_dte_weeks": 1,
        "entry_percentile": 0.15,
        "otm_pts": 6,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "use_regime": True,
    },
    "DTE 3 Weeks Aggressive": {
        "description": "Short-term sweet spot — high harvest with vega buffer",
        "long_dte_weeks": 3,
        "entry_percentile": 0.15,
        "otm_pts": 6,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "use_regime": True,
    },
    "Exit 1.5x Profit": {
        "description": "Tighter profit target — faster capital turnover",
        "long_dte_weeks": 26,
        "entry_percentile": 0.15,
        "otm_pts": 10,
        "target_mult": 1.5,
        "exit_mult": 0.3,
        "use_regime": True,
    },
    "Static No-Regime": {
        "description": "Benchmark — fixed conservative params, no adaptive switching",
        "long_dte_weeks": 26,
        "entry_percentile": 0.15,
        "otm_pts": 10,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "use_regime": False,  # No regime adaptation
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fmt_dollar(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def _fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
    except Exception:
        return "n/a"


def _compute_cagr(equity: np.ndarray, weeks_per_year: float = 52.0) -> float:
    if equity is None or len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / weeks_per_year
    if years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0


def _compute_max_dd(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    e = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / peak
    return float(dd.min())


def _parse_float_list(s: str) -> list[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals


def _parse_int_list(s: str) -> list[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            continue
    return vals


# ---------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------

def page_dashboard(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Dashboard page with regime timeline and overview."""
    st.title("📊 VIX 5% Weekly – Dashboard")
    
    underlying = params.get("underlying_symbol", "^VIX")
    
    # Initialize regime adapter
    lookback = int(params.get("entry_lookback_weeks", 52))
    adapter = RegimeAdapter(vix_weekly, lookback_weeks=lookback)
    adapter.compute_regime_history()
    
    # Current regime
    current_regime = adapter.get_regime_at_index(len(vix_weekly) - 1)
    current_pct = adapter.get_percentile_at_index(len(vix_weekly) - 1)
    current_config = REGIME_CONFIGS.get(current_regime)
    
    # Get current price as scalar
    current_price = vix_weekly.iloc[-1]
    if hasattr(current_price, 'item'):
        current_price = current_price.item()
    elif hasattr(current_price, 'values'):
        current_price = float(current_price.values[0])
    else:
        current_price = float(current_price)
    
    st.markdown("### Current Market Regime")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Current {underlying}", f"{current_price:.2f}")
    col2.metric("52-Week Percentile", f"{current_pct:.1%}")
    col3.metric("Current Regime", current_config.name if current_config else current_regime)
    col4.metric("Recommended Mode", current_config.mode if current_config else "diagonal")
    
    # Regime description
    if current_config:
        st.info(f"**{current_config.name}**: {current_config.description}")
    
    st.markdown("---")
    
    # Regime Timeline Chart
    st.markdown("### Regime Timeline")
    
    regime_df = create_regime_timeline_df(adapter)
    
    # Ensure price values are 1D
    price_values = np.asarray(vix_weekly.values).ravel()
    pct_values = regime_df["percentile"].values
    if hasattr(pct_values, 'ravel'):
        pct_values = pct_values.ravel()
    
    # Create chart with price and regime coloring
    chart_df = pd.DataFrame({
        "Price": price_values,
        "Percentile": pct_values * 100,
    }, index=vix_weekly.index)
    
    st.line_chart(chart_df)
    
    # Regime distribution
    st.markdown("### Regime Distribution")
    
    summary = adapter.get_regime_summary()
    st.dataframe(summary, use_container_width=True)
    
    # Recent transitions
    st.markdown("### Recent Regime Transitions")
    
    transitions = adapter.get_regime_transitions()
    if transitions:
        trans_df = pd.DataFrame(transitions[-10:])  # Last 10
        trans_df["date"] = pd.to_datetime(trans_df["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(trans_df, use_container_width=True)
    else:
        st.info("No regime transitions in this period.")
    
    # Optimized params summary
    st.markdown("### Optimized Parameters by Regime")
    
    mode = params.get("mode", "diagonal")
    regime_params = get_all_regime_params(mode)
    
    if regime_params:
        summary_rows = []
        for regime_name, rec in regime_params.items():
            row = rec.get("row", {})
            config = REGIME_CONFIGS.get(regime_name)
            summary_rows.append({
                "Regime": config.name if config else regime_name,
                "Entry %ile": f"{row.get('entry_percentile', 'N/A'):.2f}" if row.get('entry_percentile') else "N/A",
                "OTM pts": row.get("otm_pts", "N/A"),
                "Sigma ×": row.get("sigma_mult", "N/A"),
                "DTE weeks": row.get("long_dte_weeks", "N/A"),
                "Score": f"{row.get('Score', 0):.3f}" if row.get('Score') else "N/A",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    else:
        st.info("No optimized parameters found. Run Per-Regime Optimization first.")


# ---------------------------------------------------------------------
# Page: Standard Backtester
# ---------------------------------------------------------------------

def page_backtester(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Standard backtester with grid scan."""
    st.title("🔬 VIX 5% Weekly – Backtester")
    
    underlying = params.get("underlying_symbol", "^VIX")
    pricing_source = params.get("pricing_source", "Synthetic (BS)")
    effective_params = apply_best_if_requested(params)
    
    # Display engine info
    if pricing_source == "Massive historical":
        if MASSIVE_AVAILABLE:
            st.info(f"🔌 Using Massive historical data for {underlying}")
            
            # Debug toggle
            import os
            debug_on = st.checkbox("Enable Massive debug logging (see terminal)", value=False)
            if debug_on:
                os.environ["MASSIVE_DEBUG"] = "1"
            else:
                os.environ.pop("MASSIVE_DEBUG", None)
        else:
            st.warning("⚠️ Massive client not installed. Falling back to Synthetic (BS). Run: pip install massive-client")
            pricing_source = "Synthetic (BS)"
    
    # Run backtest with appropriate engine
    if pricing_source == "Massive historical" and MASSIVE_AVAILABLE:
        # Show progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        
        def _progress_cb(step: int, total: int):
            if total > 0:
                frac = min(max(step / float(total), 0.0), 1.0)
                progress_bar.progress(frac)
                progress_text.text(f"Massive backtest: {step}/{total} weeks")
        
        bt = run_backtest_massive(
            vix_weekly,
            effective_params,
            symbol=underlying.replace("^", ""),  # Remove ^ for Massive
            progress_cb=_progress_cb,
        )
        
        progress_bar.empty()
        progress_text.empty()
        
        # Show data source stats prominently
        massive_count = bt.get("massive_price_count", 0)
        bs_count = bt.get("bs_fallback_count", 0)
        total_prices = massive_count + bs_count
        
        if total_prices > 0:
            massive_pct = massive_count / total_prices * 100
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Massive Prices", massive_count, f"{massive_pct:.1f}%")
            col_m2.metric("BS Fallback", bs_count, f"{100-massive_pct:.1f}%")
            col_m3.metric("Total Lookups", total_prices)
            
            if massive_count == 0:
                st.warning(
                    "⚠️ **No Massive data retrieved!** All prices fell back to Black-Scholes.\n\n"
                    "Possible causes:\n"
                    "- API key doesn't have historical aggregates access (upgrade may be needed)\n"
                    "- Contract tickers not matching available data\n"
                    "- Date range has no UVXY options data\n\n"
                    "Enable debug logging above and check terminal for details."
                )
            elif massive_pct < 50:
                st.info(f"ℹ️ {massive_pct:.0f}% real data - early years/thin strikes fall back to BS pricing")
    else:
        bt = run_backtest(vix_weekly, effective_params)
    
    equity = np.asarray(bt["equity"], dtype=float).ravel()
    initial_cap = float(effective_params.get("initial_capital", 250000))
    final_eq = float(equity[-1]) if len(equity) > 0 else initial_cap
    
    cagr = _compute_cagr(equity)
    max_dd = _compute_max_dd(equity)
    total_ret = final_eq / initial_cap - 1.0 if initial_cap > 0 else 0.0
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Capital", _fmt_dollar(initial_cap))
    col2.metric("Final Equity", _fmt_dollar(final_eq))
    col3.metric("Total Return", _fmt_pct(total_ret))
    col4.metric("CAGR", _fmt_pct(cagr))
    col5.metric("Max Drawdown", _fmt_pct(max_dd))
    
    col6, col7, col8 = st.columns(3)
    col6.metric("Trades", bt.get("trades", 0))
    col7.metric("Win Rate", _fmt_pct(bt.get("win_rate", 0)))
    col8.metric("Avg Duration", f"{bt.get('avg_trade_dur', 0):.1f} weeks")
    
    # Charts
    st.markdown(f"### Equity & {underlying}")
    
    n_eq = len(equity)
    # Ensure underlying values are 1D
    underlying_values = np.asarray(vix_weekly.iloc[:n_eq].values).ravel()
    chart_df = pd.DataFrame({
        "Equity": equity[:n_eq],
        underlying: underlying_values,
    }, index=vix_weekly.index[:n_eq])
    st.line_chart(chart_df)
    
    # Weekly PnL
    st.markdown("### Weekly PnL")
    realized = np.asarray(bt.get("realized_weekly", []), dtype=float)
    unrealized = np.asarray(bt.get("unrealized_weekly", []), dtype=float)
    
    n_pnl = min(len(realized), len(unrealized), len(vix_weekly))
    pnl_df = pd.DataFrame({
        "Realized": realized[:n_pnl],
        "Unrealized": unrealized[:n_pnl],
    }, index=vix_weekly.index[:n_pnl])
    st.bar_chart(pnl_df)
    
    # Grid Scan section
    st.markdown("---")
    st.subheader("🔍 Grid Scan")
    
    with st.expander("Grid Scan Parameters", expanded=True):
        ep_str = st.text_input("Entry percentiles", "0.05,0.10,0.15,0.20,0.30")
        sigma_str = st.text_input("Sigma multipliers", "0.5,0.8,1.0,1.2")
        otm_str = st.text_input("OTM distances", "5,8,10,12,15")
        dte_str = st.text_input("Long DTE weeks", "13,20,26")
        
        criteria = st.selectbox(
            "Optimization criteria",
            ["balanced", "cagr", "maxdd"],
            index=0
        )
    
    if st.button("Run Grid Scan"):
        with st.spinner("Running grid scan..."):
            grid_df = run_grid_scan(
                vix_weekly,
                effective_params,
                criteria=criteria,
                entry_grid=_parse_float_list(ep_str),
                sigma_grid=_parse_float_list(sigma_str),
                otm_grid=_parse_float_list(otm_str),
                dte_grid=_parse_int_list(dte_str),
            )
            st.session_state["grid_df"] = grid_df
    
    grid_df = st.session_state.get("grid_df")
    if grid_df is not None and not grid_df.empty:
        st.dataframe(grid_df, use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            grid_df.to_excel(writer, index=False, sheet_name="grid_scan")
        buf.seek(0)
        st.download_button(
            "Download as XLSX",
            data=buf,
            file_name="grid_scan_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ---------------------------------------------------------------------
# Page: Adaptive Backtester
# ---------------------------------------------------------------------

def page_adaptive_backtester(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Adaptive backtester with regime-based parameter switching."""
    st.title("🔄 VIX 5% Weekly – Adaptive Backtester")
    
    st.markdown("""
    The adaptive backtester dynamically switches parameters based on the current
    market volatility regime. Each regime has optimized settings for entry thresholds,
    position sizing, OTM distance, and exit targets.
    """)
    
    underlying = params.get("underlying_symbol", "^VIX")
    
    # Options
    use_optimized = st.checkbox(
        "Use optimized regime parameters (from Per-Regime Optimization)",
        value=True,
        help="If unchecked, uses static default parameters for each regime"
    )
    
    compare_mode = st.checkbox(
        "Compare with static backtest",
        value=True,
        help="Run both adaptive and static backtests for comparison"
    )
    
    if st.button("Run Adaptive Backtest"):
        with st.spinner("Running adaptive backtest..."):
            if compare_mode:
                results = compare_adaptive_vs_static(vix_weekly, params)
                adaptive = results["adaptive_results"]
                static = results["static_results"]
                comparison = results["comparison"]
                
                st.session_state["adaptive_bt"] = adaptive
                st.session_state["static_bt"] = static
                st.session_state["comparison"] = comparison
            else:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                def progress_cb(current, total):
                    if total > 0:
                        progress_bar.progress(current / total)
                        progress_text.text(f"Processing week {current}/{total}")
                
                adaptive = run_adaptive_backtest(
                    vix_weekly, params,
                    use_optimized_params=use_optimized,
                    progress_cb=progress_cb
                )
                
                progress_bar.empty()
                progress_text.empty()
                
                st.session_state["adaptive_bt"] = adaptive
                st.session_state["static_bt"] = None
                st.session_state["comparison"] = None
    
    # Display results
    adaptive = st.session_state.get("adaptive_bt")
    static = st.session_state.get("static_bt")
    comparison = st.session_state.get("comparison")
    
    if adaptive:
        equity = np.asarray(adaptive["equity"], dtype=float).ravel()
        initial_cap = float(params.get("initial_capital", 250000))
        final_eq = float(equity[-1]) if len(equity) > 0 else initial_cap
        
        cagr = _compute_cagr(equity)
        max_dd = _compute_max_dd(equity)
        
        st.markdown("### Adaptive Backtest Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Final Equity", _fmt_dollar(final_eq))
        col2.metric("CAGR", _fmt_pct(cagr))
        col3.metric("Max Drawdown", _fmt_pct(max_dd))
        col4.metric("Trades", adaptive.get("trades", 0))
        col5.metric("Win Rate", _fmt_pct(adaptive.get("win_rate", 0)))
        
        # Comparison with static
        if comparison:
            st.markdown("### Adaptive vs Static Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            cagr_delta = comparison["cagr_delta"]
            dd_delta = comparison["maxdd_delta"]
            
            comp_col1.metric(
                "CAGR Advantage",
                _fmt_pct(cagr_delta),
                delta=f"{cagr_delta*100:+.2f}%",
                delta_color="normal"
            )
            comp_col2.metric(
                "Drawdown Advantage",
                _fmt_pct(-dd_delta),  # Less negative is better
                delta=f"{-dd_delta*100:+.2f}%",
                delta_color="normal"
            )
            comp_col3.metric(
                "Trade Count",
                f"{comparison['adaptive_trades']} vs {comparison['static_trades']}",
            )
        
        # Equity chart
        st.markdown(f"### Equity Curve")
        
        chart_df = pd.DataFrame({
            "Adaptive": equity,
        }, index=vix_weekly.index[:len(equity)])
        
        if static:
            static_eq = np.asarray(static["equity"], dtype=float).ravel()
            chart_df["Static"] = static_eq[:len(chart_df)]
        
        st.line_chart(chart_df)
        
        # Per-regime statistics
        st.markdown("### Per-Regime Performance")
        
        regime_stats = adaptive.get("per_regime_stats", {})
        if regime_stats:
            stats_rows = []
            for regime_name, stats in regime_stats.items():
                config = REGIME_CONFIGS.get(regime_name)
                stats_rows.append({
                    "Regime": config.name if config else regime_name,
                    "Trades": stats.get("trades", 0),
                    "Win Rate": f"{stats.get('win_rate', 0):.1%}",
                    "Total PnL": _fmt_dollar(stats.get("total_pnl", 0)),
                    "Avg PnL": _fmt_dollar(stats.get("avg_pnl", 0)),
                    "Avg Duration": f"{stats.get('avg_duration', 0):.1f}w",
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
        
        # Regime transitions
        transitions = adaptive.get("regime_transitions", [])
        if transitions:
            st.markdown("### Regime Transitions During Backtest")
            trans_df = pd.DataFrame(transitions)
            trans_df["date"] = pd.to_datetime(trans_df["date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(trans_df.tail(20), use_container_width=True)


# ---------------------------------------------------------------------
# Page: Profile Management (LBR-Grade)
# ---------------------------------------------------------------------

def page_profile_management(vix_weekly: pd.Series, params: Dict[str, Any]):
    """
    LBR-Grade Profile Management with 3 tabs:
    - Optimize: Run per-regime optimization
    - Visualize: Sample count chart + summary table
    - Edit: Manual parameter editing per profile
    """
    st.title("📊 Profile Management")
    st.markdown("**LBR-grade regime profile system** — Optimize, visualize, and edit parameters per market regime")
    
    # Get current mode
    mode = params.get("mode", "diagonal")
    
    # Create 3 tabs
    tab_optimize, tab_visualize, tab_edit = st.tabs(["🚀 Optimize", "📈 Visualize", "✏️ Edit"])
    
    # =========================================================================
    # TAB 1: OPTIMIZE
    # =========================================================================
    with tab_optimize:
        st.subheader("Per-Regime Optimization")
        st.markdown("Run separate grid scans for each market regime, optimizing parameters specifically for historical periods in each regime.")
        
        # Regime definitions table
        st.markdown("---")
        st.markdown("#### Current Regime Definitions")
        
        default_regimes = pd.DataFrame([
            {"Regime": "ULTRA_LOW", "Pct_Min": 0, "Pct_Max": 10, "Entry_Pct": 0.08, "OTM": 8, "Mode": "diagonal", "Description": "Aggressive entry, max premium harvest"},
            {"Regime": "LOW", "Pct_Min": 10, "Pct_Max": 25, "Entry_Pct": 0.15, "OTM": 10, "Mode": "diagonal", "Description": "Standard entry, balanced approach"},
            {"Regime": "MEDIUM", "Pct_Min": 25, "Pct_Max": 50, "Entry_Pct": 0.30, "OTM": 12, "Mode": "diagonal", "Description": "Cautious, selective entries"},
            {"Regime": "HIGH", "Pct_Min": 50, "Pct_Max": 75, "Entry_Pct": 0.60, "OTM": 15, "Mode": "long_only", "Description": "Defensive, reduced exposure"},
            {"Regime": "EXTREME", "Pct_Min": 75, "Pct_Max": 100, "Entry_Pct": 0.90, "OTM": 20, "Mode": "long_only", "Description": "No new positions, wait for calm"},
        ])
        
        if "regime_definitions" not in st.session_state:
            st.session_state.regime_definitions = default_regimes.copy()
        
        edited_regimes = st.data_editor(
            st.session_state.regime_definitions,
            column_config={
                "Regime": st.column_config.TextColumn("Regime", disabled=True, width="small"),
                "Pct_Min": st.column_config.NumberColumn("Min %ile", min_value=0, max_value=100, step=1, width="small"),
                "Pct_Max": st.column_config.NumberColumn("Max %ile", min_value=0, max_value=100, step=1, width="small"),
                "Entry_Pct": st.column_config.NumberColumn("Entry %ile", min_value=0.01, max_value=1.0, step=0.01, format="%.2f", width="small"),
                "OTM": st.column_config.NumberColumn("OTM pts", min_value=1, max_value=50, step=1, width="small"),
                "Mode": st.column_config.SelectboxColumn("Mode", options=["diagonal", "long_only"], width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
            },
            use_container_width=True,
            hide_index=True,
            key="regime_editor"
        )
        
        # Optimization parameters
        st.markdown("---")
        st.markdown("#### Optimization Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            entry_str = st.text_input(
                "Entry percentiles to test",
                value="0.05,0.10,0.15,0.20,0.25",
                key="opt_entry_percentiles",
            )
            entry_grid = _parse_float_list(entry_str) if entry_str else [0.05, 0.10, 0.15, 0.20, 0.25]
            
            sigma_str = st.text_input(
                "Sigma multipliers to test",
                value="0.5,0.8,1.0,1.2,1.5",
                key="opt_sigma_mult",
            )
            sigma_grid = _parse_float_list(sigma_str) if sigma_str else [0.5, 0.8, 1.0, 1.2, 1.5]
        
        with col2:
            otm_str = st.text_input(
                "OTM distances to test",
                value="5,8,10,12,15,20",
                key="opt_otm_distances",
            )
            otm_grid = [int(x) for x in _parse_float_list(otm_str)] if otm_str else [5, 8, 10, 12, 15, 20]
            
            dte_str = st.text_input(
                "DTE weeks to test",
                value="8,13,20,26",
                key="opt_dte_weeks",
            )
            dte_grid = [int(x) for x in _parse_float_list(dte_str)] if dte_str else [8, 13, 20, 26]
        
        criteria = st.selectbox(
            "Optimization criteria",
            options=["balanced", "cagr", "sharpe", "maxdd"],
            index=0,
            key="opt_criteria",
        )
        
        min_weeks = st.slider(
            "Minimum weeks per regime",
            min_value=26, max_value=156, value=52,
            key="opt_min_weeks",
        )
        
        st.markdown("---")
        
        if st.button("🚀 Run Per-Regime Optimization", type="primary", use_container_width=True):
            with st.spinner("Running per-regime grid scans..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                regime_count = len(edited_regimes)
                
                def progress_cb(regime_name, current, total):
                    regime_idx = list(edited_regimes['Regime']).index(regime_name) if regime_name in list(edited_regimes['Regime']) else 0
                    overall_progress = (regime_idx * total + current) / (regime_count * total)
                    progress_bar.progress(min(overall_progress, 1.0))
                    progress_text.text(f"Optimizing {regime_name}: {current}/{total} combinations")
                
                grid_df, best_by_regime = run_per_regime_grid_scan(
                    vix_weekly=vix_weekly,
                    base_params=params,
                    criteria=criteria,
                    entry_grid=entry_grid,
                    sigma_grid=sigma_grid,
                    otm_grid=otm_grid,
                    dte_grid=dte_grid,
                    min_weeks_per_regime=min_weeks,
                    progress_cb=progress_cb,
                )
                
                progress_bar.empty()
                progress_text.empty()
                
                st.session_state["regime_grid_df"] = grid_df
                st.session_state["best_by_regime"] = best_by_regime
                
                st.success("✅ Optimization complete!")
                st.rerun()
        
        # Display results if available
        grid_df = st.session_state.get("regime_grid_df")
        best_by_regime = st.session_state.get("best_by_regime")
        
        if best_by_regime:
            st.markdown("---")
            st.markdown("#### Optimization Results")
            
            comparison_df = create_regime_comparison_df(best_by_regime)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Per-regime expandable sections
            for regime_name, regime_result in best_by_regime.items():
                row = regime_result.get("row", {})
                weeks_tested = regime_result.get("weeks_tested", 0)
                cagr_val = row.get('CAGR', 0) or 0
                maxdd_val = row.get('MaxDD', 0) or 0
                
                with st.expander(f"🔹 {regime_name} — CAGR: {cagr_val*100:.1f}%, MaxDD: {maxdd_val*100:.1f}% ({weeks_tested} weeks)"):
                    st.success(f"""
                    **Best Parameters:**
                    - Entry: {row.get('entry_percentile', 'N/A')} | Sigma: {row.get('sigma_mult', 'N/A')} | OTM: {row.get('otm_pts', 'N/A')} | DTE: {row.get('long_dte_weeks', 'N/A')}
                    - Score: {row.get('Score', 0):.3f} | Trades: {row.get('Trades', 0)} | Win Rate: {(row.get('Win_rate', 0) or 0)*100:.1f}%
                    """)
                    
                    if st.button(f"💾 Save {regime_name} Profile", key=f"save_opt_{regime_name}"):
                        from core.param_history import record_best_from_grid
                        strategy_id = f"{mode}__{regime_name}"
                        best_df = pd.DataFrame([row])
                        record_best_from_grid(strategy_id, best_df, params, criteria)
                        st.success(f"✅ Saved {regime_name} profile!")
            
            # Save all button
            if st.button("💾 Save All Profiles", type="primary"):
                from core.param_history import record_best_from_grid
                for regime_name, regime_result in best_by_regime.items():
                    strategy_id = f"{mode}__{regime_name}"
                    row = regime_result.get("row", {})
                    best_df = pd.DataFrame([row])
                    record_best_from_grid(strategy_id, best_df, params, criteria)
                st.success("✅ All profiles saved!")
    
    # =========================================================================
    # TAB 2: VISUALIZE
    # =========================================================================
    with tab_visualize:
        st.subheader("Profile Visualization")
        st.markdown("View sample counts and profile status across all regimes")
        
        # Get all profiles
        profiles = get_all_profiles(mode)
        
        # Sample count bar chart
        st.markdown("---")
        st.markdown("#### Sample Count by Profile")
        
        sample_data = {regime: profiles[regime].get("sample_count", 0) for regime in REGIME_NAMES}
        sample_df = pd.DataFrame({
            "Profile": list(sample_data.keys()),
            "Sample Count (weeks)": list(sample_data.values())
        })
        
        st.bar_chart(sample_df.set_index("Profile"))
        
        # Profile summary table
        st.markdown("---")
        st.markdown("#### Profile Summary")
        
        summary_df = get_profile_summary_df(mode)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Quick stats
        st.markdown("---")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        total_samples = sum(p.get("sample_count", 0) for p in profiles.values())
        optimized_count = sum(1 for p in profiles.values() if p.get("last_optimized"))
        edited_count = sum(1 for p in profiles.values() if p.get("is_edited"))
        
        col_stat1.metric("Total Profiles", len(REGIME_NAMES))
        col_stat2.metric("Optimized", optimized_count)
        col_stat3.metric("Manually Edited", edited_count)
        col_stat4.metric("Total Samples", total_samples)
        
        # Export profiles
        st.markdown("---")
        st.markdown("#### Export / Import")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            profiles_json = export_all_profiles(mode)
            st.download_button(
                "📥 Export All Profiles (JSON)",
                data=profiles_json,
                file_name=f"VIX_Profiles_{mode}_{date.today()}.json",
                mime="application/json"
            )
        
        with col_exp2:
            uploaded = st.file_uploader("Import Profiles (JSON)", type=["json"], key="profile_import")
            if uploaded:
                json_str = uploaded.read().decode("utf-8")
                if import_profiles(json_str, mode):
                    st.success("✅ Profiles imported!")
                    st.rerun()
                else:
                    st.error("Import failed")
    
    # =========================================================================
    # TAB 3: EDIT
    # =========================================================================
    with tab_edit:
        st.subheader("Edit Profile Parameters")
        st.markdown("Manually adjust parameters for individual regime profiles")
        
        # Select profile to edit
        selected_regime = st.selectbox(
            "Select Profile to Edit",
            options=REGIME_NAMES,
            key="edit_regime_select"
        )
        
        # Get current profile
        profile = get_profile(selected_regime, mode)
        profile_params = profile.get("params", {})
        
        st.markdown("---")
        
        # Display current status
        col_status1, col_status2, col_status3 = st.columns(3)
        col_status1.metric("Last Optimized", profile.get("last_optimized", "Never")[:10] if profile.get("last_optimized") else "Never")
        col_status2.metric("Sample Count", profile.get("sample_count", 0))
        col_status3.metric("Is Edited", "Yes" if profile.get("is_edited") else "No")
        
        st.markdown("---")
        st.markdown("#### Parameters")
        
        # Parameter editing form
        with st.form(f"edit_profile_form_{selected_regime}"):
            col1, col2 = st.columns(2)
            
            with col1:
                edit_entry = st.number_input(
                    "Entry Percentile",
                    min_value=0.01, max_value=1.0,
                    value=float(profile_params.get("entry_percentile", 0.15)),
                    step=0.01, format="%.2f",
                    key=f"edit_entry_{selected_regime}"
                )
                
                edit_sigma = st.number_input(
                    "Sigma Multiplier",
                    min_value=0.1, max_value=3.0,
                    value=float(profile_params.get("sigma_mult", 1.0)),
                    step=0.1, format="%.1f",
                    key=f"edit_sigma_{selected_regime}"
                )
                
                edit_otm = st.number_input(
                    "OTM Points",
                    min_value=1, max_value=50,
                    value=int(profile_params.get("otm_pts", 10)),
                    step=1,
                    key=f"edit_otm_{selected_regime}"
                )
            
            with col2:
                edit_dte = st.number_input(
                    "DTE Weeks",
                    min_value=1, max_value=52,
                    value=int(profile_params.get("long_dte_weeks", 26)),
                    step=1,
                    key=f"edit_dte_{selected_regime}"
                )
                
                edit_target = st.number_input(
                    "Target Multiplier",
                    min_value=1.0, max_value=5.0,
                    value=float(profile_params.get("target_mult", 1.3)),
                    step=0.1, format="%.1f",
                    key=f"edit_target_{selected_regime}"
                )
                
                edit_exit = st.number_input(
                    "Exit Multiplier",
                    min_value=0.1, max_value=1.0,
                    value=float(profile_params.get("exit_mult", 0.4)),
                    step=0.1, format="%.1f",
                    key=f"edit_exit_{selected_regime}"
                )
            
            edit_mode = st.selectbox(
                "Mode",
                options=["diagonal", "long_only"],
                index=0 if profile_params.get("mode", "diagonal") == "diagonal" else 1,
                key=f"edit_mode_{selected_regime}"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                save_clicked = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)
            
            with col_btn2:
                reset_clicked = st.form_submit_button("🔄 Reset to Default", use_container_width=True)
        
        if save_clicked:
            new_params = {
                "entry_percentile": edit_entry,
                "sigma_mult": edit_sigma,
                "otm_pts": edit_otm,
                "long_dte_weeks": edit_dte,
                "target_mult": edit_target,
                "exit_mult": edit_exit,
                "mode": edit_mode,
            }
            save_profile(
                regime_name=selected_regime,
                params=new_params,
                mode=mode,
                is_edited=True,
                sample_count=profile.get("sample_count", 0),
                criteria="manual_edit",
            )
            st.success(f"✅ {selected_regime} profile saved!")
            st.rerun()
        
        if reset_clicked:
            reset_profile_to_default(selected_regime, mode)
            st.success(f"✅ {selected_regime} reset to default!")
            st.rerun()
        
        # Show default values for reference
        st.markdown("---")
        with st.expander("📋 Default Values Reference"):
            if selected_regime in DEFAULT_PROFILES:
                defaults = DEFAULT_PROFILES[selected_regime]
                st.json(defaults)


# Keep the old function name as an alias for backward compatibility
def page_per_regime_optimizer(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Alias for page_profile_management for backward compatibility."""
    page_profile_management(vix_weekly, params)

# ---------------------------------------------------------------------
# Page: Live Signals
# ---------------------------------------------------------------------

def get_current_vix_data() -> Dict[str, Any]:
    """
    Fetch current VIX data and compute percentile using yfinance.
    
    Returns dict with:
        - current_price: float
        - percentile: float (0-100)
        - prices_52w: list of last 52 weekly closes
        - last_updated: datetime
    """
    try:
        # Get 52 weeks of VIX data
        end_date = date.today()
        start_date = end_date - timedelta(weeks=60)  # Extra buffer
        
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        if vix.empty:
            return {"error": "No VIX data available"}
        
        # Handle multi-level columns from newer yfinance
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # Resample to weekly
        col = "Adj Close" if "Adj Close" in vix.columns else "Close"
        weekly = vix[col].resample("W-FRI").last().dropna()
        
        if len(weekly) < 10:
            return {"error": "Insufficient weekly data"}
        
        # Use last 52 weeks
        prices_52w = weekly.iloc[-52:].values.astype(float).ravel()
        current_price = float(prices_52w[-1])
        
        # Compute percentile
        below_count = np.sum(prices_52w[:-1] < current_price)
        percentile = (below_count / (len(prices_52w) - 1)) * 100
        
        return {
            "current_price": current_price,
            "percentile": percentile,
            "prices_52w": prices_52w.tolist(),
            "last_updated": dt.datetime.now(),
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_current_uvxy_data() -> Dict[str, Any]:
    """Fetch current UVXY spot price."""
    try:
        uvxy = yf.Ticker("UVXY")
        hist = uvxy.history(period="5d")
        
        if hist.empty:
            return {"error": "No UVXY data"}
        
        current_price = float(hist['Close'].iloc[-1])
        
        return {
            "current_price": current_price,
            "last_updated": dt.datetime.now(),
        }
    except Exception as e:
        return {"error": str(e)}


def get_uvxy_option_chain(exp_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch UVXY option chain for a specific expiration.
    
    Parameters
    ----------
    exp_date : str
        Expiration date as "YYYY-MM-DD"
        
    Returns
    -------
    DataFrame with calls, or None if unavailable
    """
    try:
        uvxy = yf.Ticker("UVXY")
        chain = uvxy.option_chain(exp_date)
        return chain.calls
    except Exception as e:
        return None


def find_best_expirations(weeks_out_short: int = 1, weeks_out_long: int = 26) -> Dict[str, Any]:
    """
    Find the best available expirations for diagonal spread.
    
    Returns dict with:
        - short_exp: nearest weekly expiration (~1 week out)
        - long_exp: LEAP expiration (~26 weeks out)
        - available_exps: all available expirations
    """
    try:
        uvxy = yf.Ticker("UVXY")
        exps = uvxy.options
        
        if not exps:
            return {"error": "No options available"}
        
        today = date.today()
        
        # Parse and filter future expirations
        future_exps = []
        for exp_str in exps:
            try:
                exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date > today + timedelta(days=2):
                    future_exps.append(exp_date)
            except:
                continue
        
        if not future_exps:
            return {"error": "No future expirations available"}
        
        future_exps.sort()
        
        # Find short-term expiration (nearest weekly, ~1 week)
        target_short = today + timedelta(weeks=weeks_out_short)
        short_exp = min(future_exps, key=lambda e: abs((e - target_short).days))
        
        # Find long-term expiration (~26 weeks for LEAP)
        target_long = today + timedelta(weeks=weeks_out_long)
        long_exp = min(future_exps, key=lambda e: abs((e - target_long).days))
        
        return {
            "short_exp": short_exp.strftime("%Y-%m-%d"),
            "long_exp": long_exp.strftime("%Y-%m-%d"),
            "short_dte": (short_exp - today).days,
            "long_dte": (long_exp - today).days,
            "available_exps": [e.strftime("%Y-%m-%d") for e in future_exps[:15]],
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_uvxy_expirations(target_long_weeks: int = 26, target_short_weeks: int = 1) -> Dict[str, Any]:
    """
    Alias for find_best_expirations with configurable long DTE.
    Used by variant signal generation.
    """
    return find_best_expirations(weeks_out_short=target_short_weeks, weeks_out_long=target_long_weeks)


def get_uvxy_diagonal_signal(
    spot: float,
    regime: str,
    expirations: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Alias for generate_diagonal_signal.
    Used by variant signal generation.
    """
    return generate_diagonal_signal(spot, regime, expirations, params)


def generate_diagonal_signal(
    spot: float,
    regime_name: str,
    expirations: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate executable diagonal spread signal using yfinance quotes.
    
    Parameters
    ----------
    spot : float
        Current UVXY spot price
    regime_name : str
        Current regime (ULTRA_LOW, LOW, etc.)
    expirations : dict
        Result from find_best_expirations()
    params : dict
        Strategy parameters (including otm_pts from regime config)
        
    Returns
    -------
    dict with trade details or error
    """
    try:
        # Get regime-specific parameters
        from core.param_history import apply_regime_params
        regime_params = apply_regime_params(params, regime_name)
        
        otm_pts = float(regime_params.get("otm_pts", 10.0))
        target_mult = float(regime_params.get("target_mult", 1.20))
        exit_mult = float(regime_params.get("exit_mult", 0.50))
        
        # Calculate target strikes
        # Long call: further OTM for more leverage
        long_strike_target = round(spot + otm_pts + 2, 0)
        # Short call: closer for higher premium collection
        short_strike_target = round(spot + otm_pts, 0)
        
        short_exp = expirations.get("short_exp")
        long_exp = expirations.get("long_exp")
        
        if not short_exp or not long_exp:
            return {"error": "No expirations available"}
        
        # Fetch option chains
        short_chain = get_uvxy_option_chain(short_exp)
        long_chain = get_uvxy_option_chain(long_exp)
        
        if short_chain is None or short_chain.empty:
            return {"error": f"No short chain for {short_exp}"}
        if long_chain is None or long_chain.empty:
            return {"error": f"No long chain for {long_exp}"}
        
        # Find best strikes
        # Long call - OTM
        long_otm = long_chain[long_chain['strike'] >= long_strike_target]
        if long_otm.empty:
            long_otm = long_chain[long_chain['strike'] >= spot]
        
        if long_otm.empty:
            return {"error": "No suitable long call strikes"}
        
        long_row = long_otm.iloc[0]
        long_strike = float(long_row['strike'])
        long_bid = float(long_row['bid']) if pd.notna(long_row['bid']) else 0.0
        long_ask = float(long_row['ask']) if pd.notna(long_row['ask']) else 0.0
        long_mid = (long_bid + long_ask) / 2 if long_ask > 0 else long_bid
        
        # Short call - slightly OTM
        short_otm = short_chain[
            (short_chain['strike'] >= short_strike_target) & 
            (short_chain['strike'] <= spot + otm_pts + 5)
        ]
        if short_otm.empty:
            short_otm = short_chain[short_chain['strike'] >= spot]
        
        if short_otm.empty:
            return {"error": "No suitable short call strikes"}
        
        short_row = short_otm.iloc[0]
        short_strike = float(short_row['strike'])
        short_bid = float(short_row['bid']) if pd.notna(short_row['bid']) else 0.0
        short_ask = float(short_row['ask']) if pd.notna(short_row['ask']) else 0.0
        short_mid = (short_bid + short_ask) / 2 if short_ask > 0 else short_bid
        
        # Calculate net debit (buy long at ask, sell short at bid for conservative)
        net_debit = long_ask - short_bid if (long_ask > 0 and short_bid > 0) else long_mid - short_mid
        net_debit_mid = long_mid - short_mid
        
        return {
            "spot": round(spot, 2),
            "regime": regime_name,
            
            "long_exp": long_exp,
            "long_dte": expirations.get("long_dte"),
            "long_strike": long_strike,
            "long_bid": round(long_bid, 2),
            "long_ask": round(long_ask, 2),
            "long_mid": round(long_mid, 2),
            "long_iv": float(long_row.get('impliedVolatility', 0)) * 100 if pd.notna(long_row.get('impliedVolatility')) else None,
            
            "short_exp": short_exp,
            "short_dte": expirations.get("short_dte"),
            "short_strike": short_strike,
            "short_bid": round(short_bid, 2),
            "short_ask": round(short_ask, 2),
            "short_mid": round(short_mid, 2),
            "short_iv": float(short_row.get('impliedVolatility', 0)) * 100 if pd.notna(short_row.get('impliedVolatility')) else None,
            
            "net_debit_conservative": round(net_debit, 2),
            "net_debit_mid": round(net_debit_mid, 2),
            
            "target_mult": target_mult,
            "exit_mult": exit_mult,
            "profit_target": round(net_debit_mid * target_mult, 2),
            "stop_loss": round(net_debit_mid * exit_mult, 2),
            
            "otm_pts_used": otm_pts,
            "timestamp": dt.datetime.now().isoformat(),
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def page_live_signals(vix_weekly: pd.Series, params: Dict[str, Any]):
    """
    Live Trading Signals page with real-time regime detection and executable diagonal legs.
    """
    st.title("🔴 Live Trading Signals")
    st.markdown("**Real-time regime detection and executable diagonal spreads (yfinance quotes)**")
    
    # Load trading log to check for open positions
    import json
    from pathlib import Path
    log_file = Path(__file__).parent / "core" / "trading_log.json"
    
    trade_log = []
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                trade_log = json.load(f)
        except:
            pass
    
    # Check for open long position
    open_long = get_open_long_position(trade_log)
    
    # Position Status Banner
    if open_long:
        days_to_exp = None
        if open_long.get('long_exp'):
            try:
                exp_date = dt.datetime.strptime(open_long['long_exp'], '%Y-%m-%d').date()
                days_to_exp = (exp_date - date.today()).days
            except:
                pass
        
        st.success(f"""
        ✅ **OPEN LONG POSITION HELD**
        - **Long Leg**: UVXY {open_long.get('long_exp')} ${open_long.get('long_strike')} Call × {open_long.get('contracts')} contracts
        - **Entry**: {open_long.get('entry_date')} @ ${open_long.get('entry_fill', 0):.2f}
        - **Days to Expiration**: {days_to_exp if days_to_exp else 'N/A'}
        - **Action**: Roll short leg only — do NOT open new long
        """)
    
    # Auto-refresh toggle
    col_refresh1, col_refresh2 = st.columns([1, 4])
    with col_refresh1:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    with col_refresh2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    if auto_refresh:
        st.markdown("*Auto-refreshing every 30 seconds...*")
        import time
        time.sleep(0.1)  # Small delay to prevent immediate rerun
        # Note: For true auto-refresh, use st.empty() with a loop or streamlit-autorefresh
    
    st.markdown("---")
    
    # Get current market data
    with st.spinner("Fetching market data..."):
        vix_data = get_current_vix_data()
        uvxy_data = get_current_uvxy_data()
    
    if "error" in vix_data:
        st.error(f"VIX Data Error: {vix_data['error']}")
        return
    
    # Calculate current regime
    current_vix = vix_data["current_price"]
    percentile = vix_data["percentile"]
    
    # Determine regime from percentile
    if percentile <= 10:
        regime_name = "ULTRA_LOW"
        regime_color = "🟢"
        regime_action = "AGGRESSIVE ENTRY"
    elif percentile <= 25:
        regime_name = "LOW"
        regime_color = "🟢"
        regime_action = "STANDARD ENTRY"
    elif percentile <= 50:
        regime_name = "MEDIUM"
        regime_color = "🟡"
        regime_action = "CAUTIOUS / HOLD"
    elif percentile <= 75:
        regime_name = "HIGH"
        regime_color = "🟠"
        regime_action = "DEFENSIVE"
    else:
        regime_name = "EXTREME"
        regime_color = "🔴"
        regime_action = "NO NEW POSITIONS"
    
    # Display current state
    st.subheader("📊 Current Market State")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VIX Close", f"{current_vix:.2f}")
    
    with col2:
        st.metric("52-Week Percentile", f"{percentile:.1f}%")
    
    with col3:
        st.metric("Current Regime", f"{regime_color} {regime_name}")
    
    with col4:
        st.metric("Action", regime_action)
    
    # UVXY spot
    if "error" not in uvxy_data:
        uvxy_spot = uvxy_data["current_price"]
        st.metric("UVXY Spot", f"${uvxy_spot:.2f}")
    else:
        uvxy_spot = None
        st.warning(f"UVXY data unavailable: {uvxy_data.get('error')}")
    
    st.markdown("---")
    
    # Regime-specific parameters
    st.subheader("⚙️ Regime-Optimized Parameters")
    
    # Get best params for current regime
    best_regime_params = get_best_for_regime(params.get("mode", "diagonal"), regime_name)
    
    if best_regime_params and "row" in best_regime_params:
        row = best_regime_params["row"]
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        with col_p1:
            st.metric("Entry Percentile", f"{row.get('entry_pct', row.get('entry_percentile', 'N/A'))}")
        with col_p2:
            st.metric("OTM Points", f"{row.get('otm_pts', 'N/A')}")
        with col_p3:
            st.metric("Long DTE (weeks)", f"{row.get('long_dte_weeks', 'N/A')}")
        with col_p4:
            st.metric("Sigma Mult", f"{row.get('sigma_mult', 'N/A')}")
        
        st.caption(f"*Optimized on {best_regime_params.get('timestamp', 'unknown')}*")
    else:
        st.info(f"No optimized parameters found for {regime_name} regime. Using defaults.")
        st.caption("Run Per-Regime Optimizer to generate optimized parameters.")
    
    st.markdown("---")
    
    # Entry Signal Logic
    st.subheader("📡 Entry Signal")
    
    # Entry threshold based on regime
    entry_threshold = 25.0  # Default: enter when percentile <= 25%
    
    if open_long:
        # Already have open position - show roll guidance
        st.info(f"""
        📋 **POSITION OPEN — ROLL SHORT LEG ONLY**
        - You hold: UVXY {open_long.get('long_exp')} ${open_long.get('long_strike')} Call
        - Percentile: {percentile:.1f}% ({regime_name} regime)
        - Action: Roll short leg weekly to harvest premium
        """)
        signal_active = False  # Don't show new entry
        show_full_diagonal = False
    elif percentile <= entry_threshold:
        st.success(f"🟢 **ENTRY SIGNAL ACTIVE** — Percentile ({percentile:.1f}%) ≤ threshold ({entry_threshold}%)")
        signal_active = True
        show_full_diagonal = True
    else:
        st.info(f"🔕 **HOLD** — Percentile ({percentile:.1f}%) > threshold ({entry_threshold}%)")
        signal_active = False
        show_full_diagonal = True  # Show reference diagonal even when not active
    
    st.markdown("---")
    
    # Diagonal Trade Details (if UVXY available and signal active)
    if open_long:
        st.subheader("📋 Short Leg Roll Details")
    else:
        st.subheader("📋 Diagonal Spread Details")
    
    if uvxy_spot is None:
        st.warning("Cannot generate trade details without UVXY data.")
        return
    
    with st.spinner("Fetching option chains..."):
        expirations = find_best_expirations(
            weeks_out_short=1,
            weeks_out_long=int(params.get("long_dte_weeks", 26))
        )
    
    if "error" in expirations:
        st.error(f"Expiration Error: {expirations['error']}")
        return
    
    # Show available expirations
    with st.expander("📅 Available Expirations"):
        st.write(f"**Short-term ({expirations['short_dte']} DTE):** {expirations['short_exp']}")
        st.write(f"**Long-term ({expirations['long_dte']} DTE):** {expirations['long_exp']}")
        st.write("All available:", expirations.get("available_exps", []))
    
    # Generate diagonal signal
    with st.spinner("Generating trade signal..."):
        signal = generate_diagonal_signal(uvxy_spot, regime_name, expirations, params)
    
    if "error" in signal:
        st.error(f"Signal Generation Error: {signal['error']}")
        if "traceback" in signal:
            with st.expander("Error Details"):
                st.code(signal["traceback"])
        return
    
    # Display trade signal based on position status
    if open_long:
        # Show short roll only
        st.info("### 🔄 SHORT LEG ROLL")
        
        col_held, col_roll = st.columns(2)
        
        with col_held:
            st.markdown("#### LONG LEG (HELD)")
            st.write(f"**UVXY {open_long.get('long_exp')} ${open_long.get('long_strike')} Call**")
            st.write(f"Entry Fill: ${open_long.get('entry_fill', 0):.2f}")
            st.write(f"Contracts: {open_long.get('contracts')}")
            st.write("*(Do NOT buy more)*")
        
        with col_roll:
            st.markdown("#### SHORT LEG (Sell to Open)")
            st.write(f"**UVXY {signal['short_exp']} ${signal['short_strike']} Call**")
            st.write(f"Bid: ${signal['short_bid']:.2f}")
            st.write(f"Ask: ${signal['short_ask']:.2f}")
            st.write(f"Mid: ${signal['short_mid']:.2f}")
            if signal.get('short_iv'):
                st.write(f"IV: {signal['short_iv']:.1f}%")
            st.write(f"DTE: {signal['short_dte']} days")
        
        # Credit from roll
        st.markdown("---")
        st.success(f"""
        **💰 Expected Credit from Roll: ${signal['short_mid']:.2f} × {open_long.get('contracts', 1)} contracts = ${signal['short_mid'] * open_long.get('contracts', 1) * 100:.0f}**
        
        Use bid price for conservative estimate: ${signal['short_bid'] * open_long.get('contracts', 1) * 100:.0f}
        """)
        
    else:
        # Show full diagonal for new entry
        if signal_active:
            st.success("### 🎯 EXECUTABLE DIAGONAL SPREAD")
        else:
            st.info("### 📋 Reference Diagonal Spread (signal not active)")
        
        # Trade details in columns
        col_long, col_short, col_net = st.columns(3)
        
        with col_long:
            st.markdown("#### LONG LEG (Buy)")
            st.write(f"**UVXY {signal['long_exp']} ${signal['long_strike']} Call**")
            st.write(f"Bid: ${signal['long_bid']:.2f}")
            st.write(f"Ask: ${signal['long_ask']:.2f}")
            st.write(f"Mid: ${signal['long_mid']:.2f}")
            if signal.get('long_iv'):
                st.write(f"IV: {signal['long_iv']:.1f}%")
            st.write(f"DTE: {signal['long_dte']} days")
        
        with col_short:
            st.markdown("#### SHORT LEG (Sell)")
            st.write(f"**UVXY {signal['short_exp']} ${signal['short_strike']} Call**")
            st.write(f"Bid: ${signal['short_bid']:.2f}")
            st.write(f"Ask: ${signal['short_ask']:.2f}")
            st.write(f"Mid: ${signal['short_mid']:.2f}")
            if signal.get('short_iv'):
                st.write(f"IV: {signal['short_iv']:.1f}%")
            st.write(f"DTE: {signal['short_dte']} days")
        
        with col_net:
            st.markdown("#### NET POSITION")
            st.write(f"**Net Debit (mid):** ${signal['net_debit_mid']:.2f}")
            st.write(f"**Net Debit (conservative):** ${signal['net_debit_conservative']:.2f}")
            st.write(f"---")
            st.write(f"**Profit Target ({signal['target_mult']}x):** ${signal['profit_target']:.2f}")
            st.write(f"**Stop Loss ({signal['exit_mult']}x):** ${signal['stop_loss']:.2f}")
            st.write(f"---")
            st.write(f"OTM pts used: {signal['otm_pts_used']}")
    
    # Position Sizing Section (only for new entries)
    if not open_long:
        st.markdown("---")
        st.subheader("📊 Position Sizing")
        
        initial_capital = float(params.get("initial_capital", 250000))
        alloc_pct = float(params.get("alloc_pct", 0.01))
        
        # Convert alloc_pct to percentage if needed
        if alloc_pct > 1.0:
            alloc_pct_display = alloc_pct
            alloc_pct = alloc_pct / 100.0
        else:
            alloc_pct_display = alloc_pct * 100
        
        risk_per_trade = initial_capital * alloc_pct
        net_debit = signal.get('net_debit_mid', 0)
        
        # Calculate suggested contracts
        if net_debit > 0:
            contracts = int(risk_per_trade // (net_debit * 100))
            contracts = max(1, min(contracts, 50))  # Min 1, max 50 contracts
        else:
            # For credit spreads or zero cost, use allocation / long leg cost
            long_mid = signal.get('long_mid', 1.0)
            if long_mid > 0:
                contracts = int(risk_per_trade // (long_mid * 100))
                contracts = max(1, min(contracts, 50))
            else:
                contracts = 1
        
        col_size1, col_size2, col_size3, col_size4 = st.columns(4)
        
        with col_size1:
            st.metric("Account Size", f"${initial_capital:,.0f}")
        
        with col_size2:
            st.metric("Allocation %", f"{alloc_pct_display:.1f}%")
        
        with col_size3:
            st.metric("Risk per Trade", f"${risk_per_trade:,.0f}")
        
        with col_size4:
            total_risk = contracts * net_debit * 100 if net_debit > 0 else contracts * signal.get('long_mid', 0) * 100
            st.metric("Suggested Contracts", f"{contracts}")
        
        # Risk summary
        if signal_active:
            st.success(f"""
            **📈 Suggested Position: {contracts} contract(s)**
            - Total capital at risk: ${total_risk:,.0f}
            - Max profit target: ${contracts * signal['profit_target'] * 100:,.0f}
            - Max loss (stop): ${contracts * signal['stop_loss'] * 100:,.0f}
            """)
        else:
            st.info(f"""
            **📋 Reference Position: {contracts} contract(s)** (signal not active)
            - Would risk: ${total_risk:,.0f} if entered
            """)
        
        # JSON export for new entry
        with st.expander("📤 Export Signal as JSON"):
            signal_with_sizing = dict(signal)
            signal_with_sizing["position_sizing"] = {
                "account_size": initial_capital,
                "alloc_pct": alloc_pct_display,
                "risk_per_trade": risk_per_trade,
                "suggested_contracts": contracts,
                "total_risk": total_risk,
            }
            st.json(signal_with_sizing)
    else:
        # JSON export for roll (open position case)
        with st.expander("📤 Export Roll Signal as JSON"):
            roll_signal = {
                "action": "ROLL_SHORT_ONLY",
                "open_long": open_long,
                "short_leg": {
                    "expiration": signal['short_exp'],
                    "strike": signal['short_strike'],
                    "bid": signal['short_bid'],
                    "ask": signal['short_ask'],
                    "mid": signal['short_mid'],
                    "dte": signal['short_dte'],
                },
                "expected_credit_per_contract": signal['short_mid'],
                "expected_total_credit": signal['short_mid'] * open_long.get('contracts', 1) * 100,
                "timestamp": signal.get('timestamp'),
            }
            st.json(roll_signal)
    
    # Risk warnings
    st.markdown("---")
    
    # =================================================================
    # 5 VARIANT SIGNALS COMPARISON
    # =================================================================
    st.subheader("📊 5 Variant Signal Comparison")
    st.markdown("**Compare signals across all paper trading variants**")
    
    # Get signals for all variants
    variant_signals = {}
    for variant_name, variant_config in PAPER_VARIANTS.items():
        # Build variant-specific params
        variant_params = dict(params)
        variant_params['long_dte_weeks'] = variant_config['long_dte_weeks']
        variant_params['entry_percentile'] = variant_config['entry_percentile']
        variant_params['otm_pts'] = variant_config['otm_pts']
        variant_params['target_mult'] = variant_config['target_mult']
        variant_params['exit_mult'] = variant_config['exit_mult']
        
        # Get expirations for this variant
        variant_exps = get_uvxy_expirations(target_long_weeks=variant_config['long_dte_weeks'])
        
        if variant_exps and "error" not in variant_exps:
            variant_signal = get_uvxy_diagonal_signal(
                spot=uvxy_spot if uvxy_spot else 0,
                regime=regime_name,
                expirations=variant_exps,
                params=variant_params,
            )
            variant_signals[variant_name] = variant_signal
        else:
            variant_signals[variant_name] = {"error": "No expirations available"}
    
    # Display variant signals in columns
    for variant_name, variant_config in PAPER_VARIANTS.items():
        v_signal = variant_signals.get(variant_name, {})
        
        with st.expander(f"**{variant_name}** — {variant_config['description'][:40]}...", expanded=False):
            if "error" in v_signal:
                st.error(f"Error: {v_signal.get('error')}")
            else:
                col_v1, col_v2, col_v3 = st.columns(3)
                
                with col_v1:
                    st.markdown("**Long Leg**")
                    st.write(f"UVXY {v_signal.get('long_exp', 'N/A')} ${v_signal.get('long_strike', 0):.0f}C")
                    st.write(f"Mid: ${v_signal.get('long_mid', 0):.2f}")
                    st.write(f"DTE: {v_signal.get('long_dte', 0)} days")
                
                with col_v2:
                    st.markdown("**Short Leg**")
                    st.write(f"UVXY {v_signal.get('short_exp', 'N/A')} ${v_signal.get('short_strike', 0):.0f}C")
                    st.write(f"Mid: ${v_signal.get('short_mid', 0):.2f}")
                    st.write(f"DTE: {v_signal.get('short_dte', 0)} days")
                
                with col_v3:
                    st.markdown("**Net Position**")
                    net_mid = v_signal.get('net_debit_mid', 0)
                    st.write(f"Net Debit: ${net_mid:.2f}")
                    st.write(f"Target ({variant_config['target_mult']}x): ${net_mid * variant_config['target_mult']:.2f}")
                    st.write(f"Stop ({variant_config['exit_mult']}x): ${net_mid * variant_config['exit_mult']:.2f}")
                
                # Quick sizing
                risk_budget = float(params.get("initial_capital", 250000)) * 0.01
                suggested = int(risk_budget // (net_mid * 100)) if net_mid > 0 else 1
                suggested = max(1, min(suggested, 50))
                st.info(f"**Suggested: {suggested} contracts** (1% risk = ${suggested * net_mid * 100:,.0f})")
    
    st.markdown("---")
    st.warning("""
    ⚠️ **Risk Warnings:**
    - This is a **paper trading / research tool** — not financial advice
    - Always verify quotes with your broker before trading
    - UVXY is a leveraged, decaying ETP — positions require active management
    - Bid-ask spreads can be wide; use limit orders
    """)
    
    # Last updated
    st.caption(f"*Last updated: {signal.get('timestamp', 'N/A')}*")
    
    # =================================================================
    # BUILD VARIANTS DATA FOR EXPORT (using variant_signals from above)
    # =================================================================
    export_variants = []
    for variant_name, variant_config in PAPER_VARIANTS.items():
        v_signal = variant_signals.get(variant_name, {})
        if "error" not in v_signal:
            net_mid = v_signal.get('net_debit_mid', 0)
            long_mid = v_signal.get('long_mid', 0)
            short_mid = v_signal.get('short_mid', 0)
            
            # Estimate bid/ask from mid (typical 5% spread for UVXY options)
            spread_pct = 0.05
            long_bid = long_mid * (1 - spread_pct) if long_mid > 0 else 0
            long_ask = long_mid * (1 + spread_pct) if long_mid > 0 else 0
            short_bid = short_mid * (1 - spread_pct) if short_mid > 0 else 0
            short_ask = short_mid * (1 + spread_pct) if short_mid > 0 else 0
            
            # Use actual bid/ask if available from signal
            long_bid = v_signal.get('long_bid', long_bid)
            long_ask = v_signal.get('long_ask', long_ask)
            short_bid = v_signal.get('short_bid', short_bid)
            short_ask = v_signal.get('short_ask', short_ask)
            
            export_variants.append({
                "name": variant_name,
                "desc": variant_config.get('description', ''),
                # Leg details
                "long_strike": v_signal.get('long_strike', 0),
                "long_exp": v_signal.get('long_exp', ''),
                "long_dte": v_signal.get('long_dte', 0),
                "long_bid": long_bid,
                "long_ask": long_ask,
                "long_mid": long_mid,
                "short_strike": v_signal.get('short_strike', 0),
                "short_exp": v_signal.get('short_exp', ''),
                "short_dte": v_signal.get('short_dte', 0),
                "short_bid": short_bid,
                "short_ask": short_ask,
                "short_mid": short_mid,
                # Position summary
                "net_debit": net_mid,
                "risk_per_contract": abs(net_mid) * 100,
                "target_mult": variant_config['target_mult'],
                "target_price": long_mid * variant_config['target_mult'],
                "stop_mult": variant_config['exit_mult'],
                "stop_price": long_mid * variant_config['exit_mult'],
                "suggested_contracts": max(1, min(int(2500 / (abs(net_mid) * 100)), 50)) if net_mid != 0 else 1,
            })
    
    # =================================================================
    # AUTO-EXPORT JSON FOR EMAILER (runs every page load)
    # =================================================================
    import json
    from pathlib import Path
    
    live_data = {
        "vix_close": float(current_vix),
        "percentile": float(percentile),  # Already 0-100 from vix_data
        "regime": regime_name,
        "signal_active": bool(signal_active),
        "uvxy_spot": float(uvxy_spot) if uvxy_spot else 0.0,
        "variants": export_variants,
        "generated_at": dt.datetime.now().isoformat(),
    }
    
    # Save to JSON for emailer
    data_path = Path(__file__).parent / "live_signal_data.json"
    try:
        with open(data_path, "w") as f:
            json.dump(live_data, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save data file: {e}")
    
    # =================================================================
    # SEND EMAIL BUTTON
    # =================================================================
    st.markdown("---")
    st.subheader("📧 Thursday Email")
    
    col_email1, col_email2 = st.columns([2, 1])
    
    with col_email1:
        recipient = st.text_input("Recipient Email", value="onoshin333@gmail.com", key="email_recipient")
    
    with col_email2:
        force_send = st.checkbox("Send even if no signal", key="force_email")
    
    if st.button("📤 Send Thursday Email Now", type="primary"):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_pass = os.environ.get("SMTP_PASS")
        
        if not smtp_user or not smtp_pass:
            st.error("❌ SMTP credentials missing. Set SMTP_USER and SMTP_PASS environment variables.")
        elif not signal_active and not force_send:
            st.warning("⚠️ No signal active. Check 'Send even if no signal' to send anyway.")
        else:
            # Build email
            emoji = "🟢" if signal_active else "🔴"
            subject = f"{emoji} VIX Signal: {regime_name} ({percentile:.1f}%)"
            
            # Format HTML
            html = format_html_report(live_data)
            
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = recipient
            msg.attach(MIMEText(html, 'html', 'utf-8'))
            
            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                st.success(f"✅ Email sent to {recipient}!")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")


# ---------------------------------------------------------------------
# Page: Trade Explorer
# ---------------------------------------------------------------------

def page_trade_explorer(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Trade explorer placeholder."""
    st.title("🔎 VIX 5% Weekly – Trade Explorer")
    
    st.info("""
    Trade Explorer is under development. It will provide:
    
    - Detailed trade-by-trade analysis
    - Entry/exit visualization on price chart
    - Regime-tagged trade performance
    - Individual trade drill-down
    
    For now, use the Backtester or Adaptive Backtester to view trade summaries.
    """)
    
    # Show stored param history
    st.markdown("### Parameter History")
    
    history_df = get_history_summary()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No parameter history yet. Run a grid scan to populate.")


# ---------------------------------------------------------------------
# Page: Trading Log (Multi-Variant Paper Trading)
# ---------------------------------------------------------------------

def get_open_long_position(trade_log: List[Dict[str, Any]], variant: str = None) -> Optional[Dict[str, Any]]:
    """
    Check if there's an open long position for a specific variant.
    
    Returns the open position dict if found, None otherwise.
    """
    if not trade_log:
        return None
    
    # Filter by variant if specified
    if variant:
        trade_log = [t for t in trade_log if t.get('variant') == variant]
    
    # Sort by date
    sorted_log = sorted(trade_log, key=lambda x: x.get('date', ''))
    
    open_position = None
    
    for trade in sorted_log:
        action = trade.get('action', '')
        
        if action == 'Entry':
            open_position = {
                'long_exp': trade.get('long_exp'),
                'long_strike': trade.get('long_strike'),
                'contracts': trade.get('contracts'),
                'entry_date': trade.get('date'),
                'entry_fill': trade.get('long_fill'),
                'regime': trade.get('regime'),
                'variant': trade.get('variant'),
            }
        elif 'Exit' in action and open_position:
            open_position = None
    
    # Check if position has expired
    if open_position and open_position.get('long_exp'):
        try:
            exp_date = dt.datetime.strptime(open_position['long_exp'], '%Y-%m-%d').date()
            if exp_date < date.today():
                return None  # Expired
        except:
            pass
    
    return open_position


def page_trading_log(vix_weekly: pd.Series, params: Dict[str, Any]):
    """
    Multi-Variant Trading Log for parallel paper trading comparison.
    Track 5 variants simultaneously with full logging.
    """
    st.title("📓 Multi-Variant Paper Trading Log")
    st.markdown("**Track 5 strategy variants in parallel — compare real fills, fees, slippage across variants**")
    
    import json
    from pathlib import Path
    
    # Trade log file path
    log_file = Path(__file__).parent / "core" / "trading_log.json"
    regime_log_file = Path(__file__).parent / "core" / "regime_transitions.json"
    
    # Load existing logs
    def load_trade_log() -> List[Dict[str, Any]]:
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_trade_log(trades: List[Dict[str, Any]]):
        with open(log_file, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    
    def load_regime_transitions() -> List[Dict[str, Any]]:
        if regime_log_file.exists():
            try:
                with open(regime_log_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_regime_transitions(transitions: List[Dict[str, Any]]):
        with open(regime_log_file, "w") as f:
            json.dump(transitions, f, indent=2, default=str)
    
    # Initialize session state
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = load_trade_log()
    if "regime_transitions" not in st.session_state:
        st.session_state.regime_transitions = load_regime_transitions()
    
    initial_capital = float(params.get("initial_capital", 250000))
    
    # =================================================================
    # VARIANT OVERVIEW
    # =================================================================
    st.markdown("---")
    st.subheader("📊 5 Variants Being Tracked")
    
    variant_cols = st.columns(5)
    for i, (variant_name, variant_config) in enumerate(PAPER_VARIANTS.items()):
        with variant_cols[i]:
            # Count trades for this variant
            variant_trades = [t for t in st.session_state.trade_log if t.get('variant') == variant_name]
            total_pnl = sum(t.get('realized_pnl', 0) for t in variant_trades)
            num_trades = len([t for t in variant_trades if t.get('action') == 'Entry'])
            
            # Check for open position
            open_pos = get_open_long_position(st.session_state.trade_log, variant_name)
            status = "🟢 OPEN" if open_pos else "⚪ FLAT"
            
            st.markdown(f"**{variant_name}**")
            st.caption(variant_config['description'][:50] + "...")
            st.metric("P&L", f"${total_pnl:,.0f}", delta=f"{num_trades} trades")
            st.write(status)
    
    st.markdown("---")
    
    # =================================================================
    # LOG NEW TRADE
    # =================================================================
    st.subheader("➕ Log New Trade")
    
    with st.form("log_trade_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            variant = st.selectbox("📋 Variant", list(PAPER_VARIANTS.keys()))
            trade_date = st.date_input("Trade Date", value=date.today())
            action = st.selectbox(
                "Action",
                ["Entry", "Roll Short Leg", "Exit - Target Hit", "Exit - Stop Hit", 
                 "Exit - Manual", "Adjustment", "Assignment"]
            )
        
        with col2:
            regime = st.selectbox("Regime at Entry", ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"])
            contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1)
            underlying_price = st.number_input("UVXY Spot Price", min_value=0.0, value=0.0, step=0.01)
        
        with col3:
            vix_level = st.number_input("VIX Level", min_value=0.0, value=0.0, step=0.01)
            fees = st.number_input("Fees ($)", min_value=0.0, value=1.30, step=0.65, help="Total fees for this action")
            slippage = st.number_input("Slippage ($ vs mid)", min_value=0.0, value=0.0, step=0.01, help="Actual fill vs mid price")
        
        st.markdown("**Long Leg Details**")
        col_long1, col_long2, col_long3 = st.columns(3)
        with col_long1:
            long_exp = st.text_input("Long Expiration (YYYY-MM-DD)", placeholder="2025-06-20")
        with col_long2:
            long_strike = st.number_input("Long Strike", min_value=0.0, value=0.0, step=0.5)
        with col_long3:
            long_fill = st.number_input("Long Fill Price", min_value=0.0, value=0.0, step=0.01)
        
        st.markdown("**Short Leg Details**")
        col_short1, col_short2, col_short3 = st.columns(3)
        with col_short1:
            short_exp = st.text_input("Short Expiration (YYYY-MM-DD)", placeholder="2025-01-10")
        with col_short2:
            short_strike = st.number_input("Short Strike", min_value=0.0, value=0.0, step=0.5)
        with col_short3:
            short_fill = st.number_input("Short Fill Price", min_value=0.0, value=0.0, step=0.01)
        
        st.markdown("**P&L**")
        col_pnl1, col_pnl2 = st.columns(2)
        with col_pnl1:
            net_debit_credit = st.number_input(
                "Net Debit(-) / Credit(+) per contract",
                value=0.0, step=0.01,
                help="Negative for debit (paid), positive for credit (received)"
            )
        with col_pnl2:
            realized_pnl = st.number_input(
                "Realized P&L (for exits only)",
                value=0.0, step=1.0,
                help="Total realized P&L for this trade action"
            )
        
        notes = st.text_area("Notes", placeholder="e.g., Filled at mid, quick execution, regime transition...")
        
        submitted = st.form_submit_button("📝 Log Trade", use_container_width=True)
    
    if submitted:
        open_pos = get_open_long_position(st.session_state.trade_log, variant)
        
        if action == "Entry" and open_pos:
            st.error(f"❌ **Cannot enter new position** — {variant} already has an open long leg. Exit first.")
        else:
            # For roll/adjustment, inherit the existing long leg info
            if action == "Roll Short Leg" and open_pos:
                effective_long_exp = open_pos.get('long_exp')
                effective_long_strike = open_pos.get('long_strike')
                effective_long_fill = open_pos.get('entry_fill', 0)
                effective_contracts = open_pos.get('contracts', contracts)
            else:
                effective_long_exp = long_exp
                effective_long_strike = long_strike
                effective_long_fill = long_fill
                effective_contracts = contracts
            
            new_trade = {
                "id": len(st.session_state.trade_log) + 1,
                "variant": variant,
                "date": str(trade_date),
                "action": action,
                "regime": regime,
                "contracts": effective_contracts,
                "uvxy_spot": underlying_price,
                "vix_level": vix_level,
                "long_exp": effective_long_exp,
                "long_strike": effective_long_strike,
                "long_fill": effective_long_fill,
                "short_exp": short_exp,
                "short_strike": short_strike,
                "short_fill": short_fill,
                "net_debit_credit": net_debit_credit,
                "realized_pnl": realized_pnl,
                "fees": fees,
                "slippage": slippage,
                "notes": notes,
                "logged_at": dt.datetime.now().isoformat(),
            }
            
            st.session_state.trade_log.append(new_trade)
            save_trade_log(st.session_state.trade_log)
            st.success(f"✅ Trade logged for **{variant}**: {action} on {trade_date}")
            st.rerun()
    
    # =================================================================
    # PER-VARIANT TRADE LOGS
    # =================================================================
    st.markdown("---")
    st.subheader("📋 Trade History by Variant")
    
    for variant_name in PAPER_VARIANTS.keys():
        variant_trades = [t for t in st.session_state.trade_log if t.get('variant') == variant_name]
        
        with st.expander(f"**{variant_name}** ({len(variant_trades)} actions)", expanded=len(variant_trades) > 0):
            if variant_trades:
                df = pd.DataFrame(variant_trades)
                
                # Summary metrics
                total_realized = df['realized_pnl'].sum()
                total_fees = df['fees'].sum()
                total_slippage = df['slippage'].sum()
                entries = len(df[df['action'] == 'Entry'])
                exits = len(df[df['action'].str.contains('Exit')])
                
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                col_m1.metric("Net P&L", f"${total_realized - total_fees:,.0f}")
                col_m2.metric("Gross P&L", f"${total_realized:,.0f}")
                col_m3.metric("Total Fees", f"${total_fees:,.2f}")
                col_m4.metric("Total Slippage", f"${total_slippage:,.2f}")
                col_m5.metric("Entries/Exits", f"{entries}/{exits}")
                
                # Cumulative P&L chart
                if total_realized != 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df_sorted = df.sort_values('date')
                    df_sorted['net_pnl'] = df_sorted['realized_pnl'] - df_sorted['fees']
                    df_sorted['cumulative_pnl'] = df_sorted['net_pnl'].cumsum()
                    
                    st.line_chart(df_sorted.set_index('date')['cumulative_pnl'])
                
                # Trade table
                display_cols = ['date', 'action', 'regime', 'contracts', 'uvxy_spot', 
                               'long_strike', 'short_strike', 'net_debit_credit', 
                               'realized_pnl', 'fees', 'slippage']
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols].sort_values('date', ascending=False), use_container_width=True)
                
                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    f"📥 Export {variant_name} to CSV",
                    data=csv,
                    file_name=f"Paper_{variant_name.replace(' ', '_')}_{date.today()}.csv",
                    mime="text/csv",
                    key=f"export_{variant_name}"
                )
            else:
                st.info(f"No trades logged for {variant_name} yet.")
    
    # =================================================================
    # REGIME TRANSITIONS LOG
    # =================================================================
    st.markdown("---")
    st.subheader("🔄 Regime Transitions During Paper Trade")
    
    with st.expander("Log Regime Transition", expanded=False):
        with st.form("regime_transition_form", clear_on_submit=True):
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                trans_date = st.date_input("Date", value=date.today(), key="trans_date")
            with col_r2:
                from_regime = st.selectbox("From Regime", ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"], key="from_regime")
            with col_r3:
                to_regime = st.selectbox("To Regime", ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"], key="to_regime")
            with col_r4:
                trans_vix = st.number_input("VIX at Transition", min_value=0.0, value=15.0, step=0.1, key="trans_vix")
            
            trans_notes = st.text_input("Notes", placeholder="e.g., VIX spike on earnings...", key="trans_notes")
            trans_submitted = st.form_submit_button("Log Transition")
        
        if trans_submitted:
            new_transition = {
                "date": str(trans_date),
                "from_regime": from_regime,
                "to_regime": to_regime,
                "vix_level": trans_vix,
                "notes": trans_notes,
                "logged_at": dt.datetime.now().isoformat(),
            }
            st.session_state.regime_transitions.append(new_transition)
            save_regime_transitions(st.session_state.regime_transitions)
            st.success(f"✅ Transition logged: {from_regime} → {to_regime}")
            st.rerun()
    
    if st.session_state.regime_transitions:
        trans_df = pd.DataFrame(st.session_state.regime_transitions)
        st.dataframe(trans_df.sort_values('date', ascending=False), use_container_width=True)
    else:
        st.info("No regime transitions logged yet.")
    
    # =================================================================
    # VARIANT COMPARISON SUMMARY
    # =================================================================
    st.markdown("---")
    st.subheader("📊 Variant Performance Comparison")
    
    if st.session_state.trade_log:
        comparison_data = []
        for variant_name in PAPER_VARIANTS.keys():
            variant_trades = [t for t in st.session_state.trade_log if t.get('variant') == variant_name]
            if variant_trades:
                df = pd.DataFrame(variant_trades)
                total_realized = df['realized_pnl'].sum()
                total_fees = df['fees'].sum()
                entries = len(df[df['action'] == 'Entry'])
                exits = len(df[df['action'].str.contains('Exit')])
                
                comparison_data.append({
                    "Variant": variant_name,
                    "Net P&L": total_realized - total_fees,
                    "Gross P&L": total_realized,
                    "Fees": total_fees,
                    "Entries": entries,
                    "Exits": exits,
                    "Open": "Yes" if get_open_long_position(st.session_state.trade_log, variant_name) else "No",
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # Bar chart comparison
            st.bar_chart(comp_df.set_index('Variant')['Net P&L'])
    else:
        st.info("Start logging trades to see comparison.")
    
    # =================================================================
    # EXPORT ALL & DANGER ZONE
    # =================================================================
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.subheader("📤 Export All Data")
        if st.session_state.trade_log:
            all_df = pd.DataFrame(st.session_state.trade_log)
            csv_all = all_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Trades (CSV)",
                data=csv_all,
                file_name=f"All_Paper_Trades_{date.today()}.csv",
                mime="text/csv"
            )
            
            json_all = json.dumps(st.session_state.trade_log, indent=2, default=str)
            st.download_button(
                "📥 Download All Trades (JSON)",
                data=json_all,
                file_name=f"All_Paper_Trades_{date.today()}.json",
                mime="application/json"
            )
    
    with col_exp2:
        st.subheader("⚠️ Danger Zone")
        with st.expander("Clear Data"):
            if st.button("🗑️ Clear All Trades", type="secondary"):
                st.session_state.trade_log = []
                save_trade_log([])
                st.warning("Trade log cleared!")
                st.rerun()
            
            if st.button("🗑️ Clear Regime Transitions", type="secondary"):
                st.session_state.regime_transitions = []
                save_regime_transitions([])
                st.warning("Regime transitions cleared!")
                st.rerun()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        page_icon="📈",
        layout="wide",
    )
    
    # Page selector FIRST (at top of sidebar) - Clean LBR-style
    st.sidebar.title("VIX 5% Weekly Suite")
    page = st.sidebar.radio(
        "Page",
        ["Dashboard", "Live Signals", "Trading Log", "Backtester", "Adaptive Backtester", "Trade Explorer", "Profile Management"],
        index=1,  # Default to Live Signals
        key="main_page_selector"
    )
    st.sidebar.markdown("---")
    
    # Build sidebar params (without duplicate page selector)
    params: Dict[str, Any] = build_sidebar()
    
    # Load data
    start_date = params.get("start_date", dt.date(2004, 1, 1))
    end_date = params.get("end_date", dt.date.today())
    
    vix_weekly = load_vix_weekly(start_date, end_date)
    
    if vix_weekly is None or vix_weekly.empty:
        st.error("No data available for the selected date range.")
        return
    
    # Route to page
    if page == "Dashboard":
        page_dashboard(vix_weekly, params)
    elif page == "Live Signals":
        page_live_signals(vix_weekly, params)
    elif page == "Trading Log":
        page_trading_log(vix_weekly, params)
    elif page == "Backtester":
        page_backtester(vix_weekly, params)
    elif page == "Adaptive Backtester":
        page_adaptive_backtester(vix_weekly, params)
    elif page == "Trade Explorer":
        page_trade_explorer(vix_weekly, params)
    elif page == "Profile Management":
        page_profile_management(vix_weekly, params)


if __name__ == "__main__":
    main()
