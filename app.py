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
import datetime as dt
from datetime import date, timedelta
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

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
)


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
# Page: Per-Regime Optimizer
# ---------------------------------------------------------------------

def page_per_regime_optimizer(vix_weekly: pd.Series, params: Dict[str, Any]):
    """Per-regime parameter optimization."""
    st.title("⚡ VIX 5% Weekly – Per-Regime Optimizer")
    
    st.markdown("""
    This tool runs separate grid scans for each market regime, optimizing parameters
    specifically for the historical periods that fell into each regime category.
    The optimized parameters are saved and can be used by the Adaptive Backtester.
    """)
    
    # Show current regime configs
    st.markdown("### Current Regime Definitions")
    
    config_rows = []
    for regime_name, config in REGIME_CONFIGS.items():
        config_rows.append({
            "Regime": config.name,
            "Percentile Range": f"{config.percentile_range[0]:.0%} – {config.percentile_range[1]:.0%}",
            "Default Entry %ile": config.entry_percentile,
            "Default OTM": config.otm_pts,
            "Default Mode": config.mode,
            "Description": config.description,
        })
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True)
    
    st.markdown("---")
    
    # Grid parameters
    st.markdown("### Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ep_str = st.text_input("Entry percentiles to test", "0.05,0.10,0.15,0.20,0.25")
        sigma_str = st.text_input("Sigma multipliers to test", "0.5,0.8,1.0,1.2,1.5")
    
    with col2:
        otm_str = st.text_input("OTM distances to test", "5,8,10,12,15,20")
        dte_str = st.text_input("DTE weeks to test", "8,13,20,26")
    
    criteria = st.selectbox(
        "Optimization criteria",
        ["balanced", "cagr", "maxdd"],
        index=0,
        help="balanced = CAGR + low drawdown, cagr = max returns, maxdd = min risk"
    )
    
    min_weeks = st.slider(
        "Minimum weeks per regime",
        min_value=26, max_value=104, value=52,
        help="Regimes with fewer weeks will be skipped"
    )
    
    if st.button("🚀 Run Per-Regime Optimization", type="primary"):
        with st.spinner("Running per-regime grid scans... This may take a few minutes."):
            progress_text = st.empty()
            
            def progress_cb(regime_name, current, total):
                progress_text.text(f"Optimizing {regime_name}: {current}/{total} combinations")
            
            grid_df, best_by_regime = run_per_regime_grid_scan(
                vix_weekly=vix_weekly,
                base_params=params,
                criteria=criteria,
                entry_grid=_parse_float_list(ep_str),
                sigma_grid=_parse_float_list(sigma_str),
                otm_grid=_parse_float_list(otm_str),
                dte_grid=_parse_int_list(dte_str),
                min_weeks_per_regime=min_weeks,
                progress_cb=progress_cb,
            )
            
            progress_text.empty()
            
            st.session_state["regime_grid_df"] = grid_df
            st.session_state["best_by_regime"] = best_by_regime
    
    # Display results
    grid_df = st.session_state.get("regime_grid_df")
    best_by_regime = st.session_state.get("best_by_regime")
    
    if best_by_regime:
        st.success("✅ Per-regime optimization complete! Parameters saved.")
        
        st.markdown("### Best Parameters by Regime")
        comparison_df = create_regime_comparison_df(best_by_regime)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("### Full Grid Scan Results")
        if grid_df is not None and not grid_df.empty:
            # Filter by regime
            regime_filter = st.selectbox(
                "Filter by regime",
                ["All"] + list(best_by_regime.keys())
            )
            
            if regime_filter != "All":
                config = REGIME_CONFIGS.get(regime_filter)
                display_df = grid_df[grid_df["Regime"] == (config.name if config else regime_filter)]
            else:
                display_df = grid_df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                grid_df.to_excel(writer, index=False, sheet_name="per_regime_scan")
            buf.seek(0)
            st.download_button(
                "Download Full Results as XLSX",
                data=buf,
                file_name="per_regime_optimization.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


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
    st.warning("""
    ⚠️ **Risk Warnings:**
    - This is a **paper trading / research tool** — not financial advice
    - Always verify quotes with your broker before trading
    - UVXY is a leveraged, decaying ETP — positions require active management
    - Bid-ask spreads can be wide; use limit orders
    """)
    
    # Last updated
    st.caption(f"*Last updated: {signal.get('timestamp', 'N/A')}*")


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
# Page: Trading Log
# ---------------------------------------------------------------------

def get_open_long_position(trade_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Check if there's an open long position.
    
    Returns the open position dict if found, None otherwise.
    An entry is considered "open" if:
    - Action was "Entry"
    - No subsequent "Exit" action for the same long leg
    - Long expiration hasn't passed
    """
    if not trade_log:
        return None
    
    # Sort by date
    sorted_log = sorted(trade_log, key=lambda x: x.get('date', ''))
    
    open_position = None
    
    for trade in sorted_log:
        action = trade.get('action', '')
        
        if action == 'Entry':
            # New entry - track it
            open_position = {
                'long_exp': trade.get('long_exp'),
                'long_strike': trade.get('long_strike'),
                'contracts': trade.get('contracts'),
                'entry_date': trade.get('date'),
                'entry_fill': trade.get('long_fill'),
                'regime': trade.get('regime'),
            }
        elif 'Exit' in action and open_position:
            # Exit closes the position
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
    Trading Log page for logging actual trades and comparing to backtest.
    Persists trades to a JSON file for tracking real performance.
    """
    st.title("📓 Trading Log & Real Performance")
    st.markdown("**Log actual fills vs synthetic backtest — track your real edge**")
    
    import json
    from pathlib import Path
    
    # Trade log file path
    log_file = Path(__file__).parent / "core" / "trading_log.json"
    
    # Load existing log
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
    
    # Initialize session state
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = load_trade_log()
    
    # Account settings
    initial_capital = float(params.get("initial_capital", 250000))
    
    # Check for open long position
    open_long = get_open_long_position(st.session_state.trade_log)
    
    st.markdown("---")
    
    # Position Status Section
    st.subheader("📍 Current Position Status")
    
    if open_long:
        days_to_exp = None
        if open_long.get('long_exp'):
            try:
                exp_date = dt.datetime.strptime(open_long['long_exp'], '%Y-%m-%d').date()
                days_to_exp = (exp_date - date.today()).days
            except:
                pass
        
        st.success(f"""
        ✅ **OPEN LONG POSITION**
        - **Long Leg**: UVXY {open_long.get('long_exp')} ${open_long.get('long_strike')} Call
        - **Contracts**: {open_long.get('contracts')}
        - **Entry Date**: {open_long.get('entry_date')}
        - **Entry Fill**: ${open_long.get('entry_fill', 0):.2f}
        - **Regime at Entry**: {open_long.get('regime')}
        - **Days to Expiration**: {days_to_exp if days_to_exp else 'N/A'}
        """)
        
        st.info("💡 **Action**: Roll short leg weekly, or exit when target/stop hit.")
    else:
        st.warning("⚠️ **NO OPEN POSITION** — Ready for new entry when signal fires.")
    
    st.markdown("---")
    
    # Trade entry form
    st.subheader("➕ Log New Trade")
    
    with st.form("log_trade_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            trade_date = st.date_input("Trade Date", value=date.today())
            action = st.selectbox(
                "Action",
                ["Entry", "Roll Short Leg", "Exit - Target Hit", "Exit - Stop Hit", 
                 "Exit - Manual", "Adjustment", "Assignment"]
            )
            regime = st.selectbox(
                "Regime at Entry",
                ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"]
            )
        
        with col2:
            contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1)
            underlying_price = st.number_input("UVXY Spot Price", min_value=0.0, value=0.0, step=0.01)
            vix_level = st.number_input("VIX Level", min_value=0.0, value=0.0, step=0.01)
        
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
        
        notes = st.text_area("Notes", placeholder="e.g., Filled at mid, quick execution...")
        
        submitted = st.form_submit_button("📝 Log Trade", use_container_width=True)
    
    # Validation and submission handling OUTSIDE the form
    if submitted:
        # Check if trying to enter when already have open position
        if action == "Entry" and open_long:
            st.error(f"""
            ❌ **Cannot enter new position** — You already have an open long leg:
            - UVXY {open_long.get('long_exp')} ${open_long.get('long_strike')} Call
            - Use "Roll Short Leg" to roll the short, or "Exit" to close first.
            """)
        else:
            # For roll/adjustment, inherit the existing long leg info
            if action == "Roll Short Leg" and open_long:
                effective_long_exp = open_long.get('long_exp')
                effective_long_strike = open_long.get('long_strike')
                effective_long_fill = open_long.get('entry_fill', 0)
                effective_contracts = open_long.get('contracts', contracts)
            else:
                effective_long_exp = long_exp
                effective_long_strike = long_strike
                effective_long_fill = long_fill
                effective_contracts = contracts
            
            new_trade = {
                "id": len(st.session_state.trade_log) + 1,
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
                "notes": notes,
                "logged_at": dt.datetime.now().isoformat(),
            }
            
            st.session_state.trade_log.append(new_trade)
            save_trade_log(st.session_state.trade_log)
            st.success(f"✅ Trade logged: {action} on {trade_date}")
            st.rerun()
    
    st.markdown("---")
    
    # Display trade log
    st.subheader("📋 Trade History")
    
    if st.session_state.trade_log:
        log_df = pd.DataFrame(st.session_state.trade_log)
        
        # Summary metrics
        total_trades = len(log_df)
        entries = len(log_df[log_df['action'] == 'Entry'])
        exits = len(log_df[log_df['action'].str.contains('Exit')])
        total_realized = log_df['realized_pnl'].sum()
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Actions", total_trades)
        col_m2.metric("Entries", entries)
        col_m3.metric("Exits", exits)
        col_m4.metric("Total Realized P&L", f"${total_realized:,.0f}")
        
        # Cumulative P&L chart
        if total_realized != 0:
            log_df['date'] = pd.to_datetime(log_df['date'])
            log_df_sorted = log_df.sort_values('date')
            log_df_sorted['cumulative_pnl'] = log_df_sorted['realized_pnl'].cumsum()
            
            st.markdown("### 📈 Cumulative Realized P&L")
            
            chart_df = log_df_sorted[['date', 'cumulative_pnl']].set_index('date')
            st.line_chart(chart_df)
            
            # Equity curve
            log_df_sorted['equity'] = initial_capital + log_df_sorted['cumulative_pnl']
            
            st.markdown("### 💰 Real Equity Curve")
            equity_df = log_df_sorted[['date', 'equity']].set_index('date')
            st.line_chart(equity_df)
        
        # Trade table
        st.markdown("### 📊 All Trades")
        
        # Display columns
        display_cols = ['date', 'action', 'regime', 'contracts', 'uvxy_spot', 
                       'long_strike', 'long_fill', 'short_strike', 'short_fill',
                       'net_debit_credit', 'realized_pnl']
        
        st.dataframe(
            log_df[display_cols].sort_values('date', ascending=False),
            use_container_width=True
        )
        
        # Export options
        st.markdown("### 📤 Export")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # CSV export
            csv = log_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"trading_log_{date.today()}.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # JSON export
            json_str = json.dumps(st.session_state.trade_log, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"trading_log_{date.today()}.json",
                mime="application/json"
            )
        
        # Clear log option
        st.markdown("---")
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Clear All Trades", type="secondary"):
                st.session_state.trade_log = []
                save_trade_log([])
                st.warning("Trade log cleared!")
                st.rerun()
    else:
        st.info("No trades logged yet. Use the form above to log your first trade.")
    
    # Backtest comparison section
    st.markdown("---")
    st.subheader("📊 Backtest vs Real Comparison")
    
    st.info("""
    **Coming Soon:** Side-by-side comparison of:
    - Synthetic backtest equity curve
    - Real trading log equity curve
    - Slippage analysis (fill vs mid)
    - Win rate comparison by regime
    """)
    
    # Quick stats if we have data
    if st.session_state.trade_log:
        entries_by_regime = log_df[log_df['action'] == 'Entry'].groupby('regime').size()
        if not entries_by_regime.empty:
            st.markdown("### 📈 Entries by Regime")
            st.bar_chart(entries_by_regime)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        page_icon="📈",
        layout="wide",
    )
    
    # Page selector FIRST (at top of sidebar)
    st.sidebar.title("VIX 5% Weekly Suite")
    page = st.sidebar.radio(
        "📄 Page",
        ["Dashboard", "Live Signals", "Trading Log", "Backtester", "Adaptive Backtester", "Per-Regime Optimizer", "Trade Explorer"],
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
    elif page == "Per-Regime Optimizer":
        page_per_regime_optimizer(vix_weekly, params)
    elif page == "Trade Explorer":
        page_trade_explorer(vix_weekly, params)


if __name__ == "__main__":
    main()
