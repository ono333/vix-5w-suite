#!/usr/bin/env python3
"""
VIX 5% Weekly Suite — MAIN APP (Improved Version)

Features:
- Dashboard with equity curves and regime visualization
- Backtester with trade log display and statistics
- Trade Explorer with detailed trade analysis
- Regime-Adaptive mode that adjusts parameters based on VIX percentile
- Support for Synthetic (BS) and Massive historical pricing

Pages:
- Dashboard
- Backtester (with grid scan)
- Trade Explorer
- Regime Analysis (new)
"""

import io
import datetime as dt
from typing import Dict, Any, Optional, List
from ui.trade_logger_page import render_trade_logger_page

import numpy as np
import pandas as pd
import streamlit as st

# Import sidebar builder
try:
    from ui.sidebar import build_sidebar
except ImportError:
    from sidebar import build_sidebar

# Import data loader
try:
    from core.data_loader import load_vix_weekly, load_weekly
except ImportError:
    from data_loader import load_vix_weekly

# Import backtesters
try:
    from core.backtester import run_backtest
    from core.backtester_massive import run_backtest_massive
except ImportError:
    from backtester import run_backtest
    from backtester_massive import run_backtest_massive

# Import grid scan
try:
    from experiments.grid_scan import run_grid_scan
except ImportError:
    from grid_scan import run_grid_scan

# Import param history
try:
    from core.param_history import apply_best_if_requested, get_best_for_strategy
except ImportError:
    from param_history import apply_best_if_requested, get_best_for_strategy

# Import regime adapter (new)
try:
    from regime_adapter import (
        RegimeAdapter, 
        run_regime_adaptive_backtest,
        get_regime_summary,
        get_regime_trade_stats,
    )
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False


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


def _compute_sharpe(weekly_returns: np.ndarray, annualize: bool = True) -> float:
    """Compute Sharpe ratio from weekly returns."""
    if weekly_returns is None or len(weekly_returns) < 2:
        return 0.0
    wr = np.asarray(weekly_returns, dtype=float)
    wr = wr[np.isfinite(wr)]
    if len(wr) < 2 or wr.std() == 0:
        return 0.0
    sharpe = wr.mean() / wr.std()
    if annualize:
        sharpe *= np.sqrt(52)
    return float(sharpe)


def _compute_vix_percentile_local(vix_weekly: pd.Series,
                                  lookback_weeks: int) -> pd.Series:
    """Rolling percentile of underlying level (VIX / UVXY / etc)."""
    prices = vix_weekly.values.astype(float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    lb = max(1, int(lookback_weeks))

    for i in range(lb, n):
        window = prices[i - lb: i]
        out[i] = (window < prices[i]).mean()

    return pd.Series(out, index=vix_weekly.index, name="vix_pct")


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


def _build_trade_log_df(trade_log: List[Dict], vix_weekly: pd.Series) -> pd.DataFrame:
    """Convert trade log to a nicely formatted DataFrame."""
    if not trade_log:
        return pd.DataFrame(columns=[
            "Entry Date", "Exit Date", "Duration (wks)", 
            "Entry Equity", "Exit Equity", "PnL ($)", "PnL (%)",
            "Strike", "Entry Regime", "Exit Regime"
        ])
    
    records = []
    for tr in trade_log:
        entry_idx = tr.get("entry_idx")
        exit_idx = tr.get("exit_idx")
        
        # Get dates
        if entry_idx is not None and entry_idx < len(vix_weekly):
            entry_date = vix_weekly.index[entry_idx]
        else:
            entry_date = tr.get("entry_date")
        
        if exit_idx is not None and exit_idx < len(vix_weekly):
            exit_date = vix_weekly.index[exit_idx]
        else:
            exit_date = tr.get("exit_date")
        
        entry_eq = tr.get("entry_equity", 0)
        exit_eq = tr.get("exit_equity", 0)
        pnl = tr.get("pnl", exit_eq - entry_eq if entry_eq else 0)
        pnl_pct = tr.get("pnl_pct", (pnl / entry_eq * 100) if entry_eq > 0 else 0)
        
        records.append({
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Duration (wks)": tr.get("duration_weeks", 0),
            "Entry Equity": entry_eq,
            "Exit Equity": exit_eq,
            "PnL ($)": pnl,
            "PnL (%)": pnl_pct,
            "Strike": tr.get("strike_long", 0),
            "Entry Regime": tr.get("entry_regime", "N/A"),
            "Exit Regime": tr.get("exit_regime", "N/A"),
        })
    
    df = pd.DataFrame(records)
    return df


def _display_trade_statistics(bt: Dict[str, Any], st_container=None):
    """Display trade statistics in a nice format."""
    container = st_container or st
    
    trades = bt.get("trades", 0)
    win_rate = bt.get("win_rate", 0)
    avg_dur = bt.get("avg_trade_dur", 0)
    
    col1, col2, col3 = container.columns(3)
    col1.metric("Total Trades", f"{trades}")
    col2.metric("Win Rate", f"{win_rate * 100:.1f}%")
    col3.metric("Avg Duration", f"{avg_dur:.1f} weeks")


def _display_trade_log_table(trade_log: List[Dict], vix_weekly: pd.Series, st_container=None):
    """Display trade log as a formatted table."""
    container = st_container or st
    
    df = _build_trade_log_df(trade_log, vix_weekly)
    
    if df.empty:
        container.info("No trades executed. Check your entry parameters or try Synthetic (BS) pricing.")
        return
    
    # Format for display
    display_df = df.copy()
    if "Entry Equity" in display_df.columns:
        display_df["Entry Equity"] = display_df["Entry Equity"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "Exit Equity" in display_df.columns:
        display_df["Exit Equity"] = display_df["Exit Equity"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "PnL ($)" in display_df.columns:
        display_df["PnL ($)"] = display_df["PnL ($)"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "N/A")
    if "PnL (%)" in display_df.columns:
        display_df["PnL (%)"] = display_df["PnL (%)"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    if "Strike" in display_df.columns:
        display_df["Strike"] = display_df["Strike"].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "N/A")
    
    container.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df.to_csv(index=False)
    container.download_button(
        "📥 Download Trade Log (CSV)",
        data=csv,
        file_name="trade_log.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        layout="wide",
    )

    # -------------------------------------------------
    # 1. Read sidebar params
    # -------------------------------------------------
    params: Dict[str, Any] = build_sidebar()
    page = params.get("page", "Dashboard")

    start_date: dt.date = params["start_date"]
    end_date: dt.date = params["end_date"]

    # Basic UI selections
    ui_pricing_source = params.get("pricing_source", "Synthetic (BS)")
    ui_underlying_symbol = params.get("underlying_symbol", "^VIX")
    
    # Check for regime-adaptive mode
    use_regime_adaptive = params.get("use_regime_adaptive", False)

    # -------------------------------------------------
    # 2. Load data (weekly underlying series)
    # -------------------------------------------------
    # Load based on underlying symbol
    if ui_underlying_symbol == "^VIX":
        vix_weekly = load_vix_weekly(start_date, end_date)
    else:
        try:
            vix_weekly = load_weekly(ui_underlying_symbol, start_date, end_date)
        except:
            vix_weekly = load_vix_weekly(start_date, end_date)
    
    if vix_weekly is None or vix_weekly.empty:
        st.error("No underlying data available for the selected date range.")
        return

    # -------------------------------------------------
    # 3. Best-param override & effective params
    # -------------------------------------------------
    effective_params = apply_best_if_requested(params)

    # Ensure pricing_source / underlying_symbol keys exist
    if "pricing_source" not in effective_params:
        effective_params["pricing_source"] = ui_pricing_source
    if "underlying_symbol" not in effective_params:
        effective_params["underlying_symbol"] = ui_underlying_symbol

    pricing_source = effective_params.get("pricing_source", "Synthetic (BS)")
    underlying_symbol = effective_params.get("underlying_symbol", "^VIX")

    # Engine label
    if use_regime_adaptive and REGIME_AVAILABLE:
        engine_label = f"Regime-Adaptive ({underlying_symbol})"
    elif pricing_source == "Massive historical":
        engine_label = f"Massive historical ({underlying_symbol})"
    else:
        engine_label = "Synthetic (Black–Scholes)"

    st.caption(f"🔧 Engine: {engine_label}")

    # -------------------------------------------------
    # 4. Run backtest
    # -------------------------------------------------
    bt = None

    if use_regime_adaptive and REGIME_AVAILABLE:
        # Regime-adaptive backtest
        adapter = RegimeAdapter()
        
        progress_text = st.empty()
        progress_bar = st.progress(0.0)

        def _progress_cb(step: int, total: int):
            if total <= 0:
                return
            frac = min(max(step / float(total), 0.0), 1.0)
            progress_bar.progress(frac)
            progress_text.text(f"Regime-adaptive backtest: {step}/{total} weeks")

        bt = run_regime_adaptive_backtest(
            vix_weekly,
            effective_params,
            adapter=adapter,
            progress_cb=_progress_cb,
        )

        progress_bar.empty()
        progress_text.empty()
        
    elif pricing_source == "Massive historical":
        # Massive historical backtest
        progress_text = st.empty()
        progress_bar = st.progress(0.0)

        def _progress_cb(step: int, total: int):
            if total <= 0:
                return
            frac = min(max(step / float(total), 0.0), 1.0)
            progress_bar.progress(frac)
            progress_text.text(f"Massive backtest: {step}/{total} weeks")

        bt = run_backtest_massive(
            vix_weekly,
            effective_params,
            symbol=underlying_symbol,
            progress_cb=_progress_cb,
        )

        progress_bar.empty()
        progress_text.empty()
    else:
        # Synthetic (Black–Scholes) engine
        bt = run_backtest(vix_weekly, effective_params)

    # -------------------------------------------------
    # 5. Normalize results & core metrics
    # -------------------------------------------------
    equity = np.asarray(bt["equity"], dtype=float).ravel()
    weekly_returns = np.asarray(bt.get("weekly_returns", []), dtype=float).ravel()
    realized_weekly = np.asarray(bt.get("realized_weekly", []), dtype=float).ravel()
    unrealized_weekly = np.asarray(bt.get("unrealized_weekly", []), dtype=float).ravel()
    trade_log = bt.get("trade_log", [])
    regime_log = bt.get("regime_log", [])

    if len(equity) > 0:
        final_eq = float(equity[-1])
    else:
        final_eq = float(effective_params.get("initial_capital", 0.0))

    cagr = _compute_cagr(equity)
    max_dd = _compute_max_dd(equity)
    sharpe = _compute_sharpe(weekly_returns)

    initial_cap = float(effective_params.get("initial_capital", 0.0))
    if initial_cap > 0:
        total_ret = final_eq / initial_cap - 1.0
    else:
        total_ret = 0.0

    # =================================================================
    # PAGE: Dashboard
    # =================================================================
    if page == "Dashboard":
        st.title(f"VIX 5% Weekly — Dashboard")

        # Key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Initial Capital", _fmt_dollar(initial_cap))
        col2.metric("Final Equity", _fmt_dollar(final_eq))
        col3.metric("Total Return", _fmt_pct(total_ret))
        col4.metric("CAGR", _fmt_pct(cagr))
        col5.metric("Max Drawdown", _fmt_pct(max_dd))
        col6.metric("Sharpe", f"{sharpe:.2f}")

        # Equity vs underlying chart
        st.markdown(f"### Equity Curve vs {underlying_symbol}")
        n_eq = len(equity)
        under_vals = np.asarray(vix_weekly.iloc[:n_eq]).astype(float).ravel()

        df_chart = pd.DataFrame({
            "Equity": np.asarray(equity[:n_eq], dtype=float).ravel(),
            underlying_symbol: under_vals,
        }, index=vix_weekly.index[:n_eq])
        st.line_chart(df_chart)

        # Percentile strip
        st.markdown(f"### 52-week {underlying_symbol} Percentile")
        pct_lb = int(effective_params.get("entry_lookback_weeks", 52))
        vix_pct = _compute_vix_percentile_local(vix_weekly, pct_lb)
        df_pct = pd.DataFrame({"Percentile": vix_pct})
        st.area_chart(df_pct)

        # Quick trade summary
        st.markdown("### Trade Summary")
        _display_trade_statistics(bt)
        
        if len(trade_log) > 0:
            with st.expander("📋 Recent Trades (last 10)", expanded=False):
                recent_trades = trade_log[-10:]
                _display_trade_log_table(recent_trades, vix_weekly)

        return

    # =================================================================
    # PAGE: Backtester (with Grid Scan)
    # =================================================================
    if page == "Backtester":
        st.title("VIX 5% Weekly — Backtester")

        # Key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Initial Capital", _fmt_dollar(initial_cap))
        col2.metric("Final Equity", _fmt_dollar(final_eq))
        col3.metric("Total Return", _fmt_pct(total_ret))
        col4.metric("CAGR", _fmt_pct(cagr))
        col5.metric("Max Drawdown", _fmt_pct(max_dd))
        col6.metric("Sharpe", f"{sharpe:.2f}")

        # Equity & underlying chart
        st.markdown(f"### Equity & {underlying_symbol}")
        n_eq = len(equity)
        vix_vals_eq = np.asarray(vix_weekly.iloc[:n_eq]).astype(float).ravel()

        df_eq = pd.DataFrame({
            "Equity": np.asarray(equity[:n_eq], dtype=float).ravel(),
            underlying_symbol: vix_vals_eq,
        }, index=vix_weekly.index[:n_eq])
        st.line_chart(df_eq)

        # Weekly PnL
        st.markdown("### Weekly PnL (realized + unrealized)")
        n_pnl = min(len(realized_weekly), len(unrealized_weekly), len(vix_weekly))
        df_pnl = pd.DataFrame({
            "realized": realized_weekly[:n_pnl],
            "unrealized": unrealized_weekly[:n_pnl],
        }, index=vix_weekly.index[:n_pnl])
        st.bar_chart(df_pnl)

        # ========== TRADE LOG SECTION ==========
        st.markdown("---")
        st.subheader("📊 Trade Log")
        
        trade_log = bt.get("trade_log", [])
        trades_count = bt.get("trades", 0)
        win_rate = bt.get("win_rate", 0)
        avg_dur = bt.get("avg_trade_dur", 0)
        
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Total Trades", f"{trades_count}")
        col_t2.metric("Win Rate", f"{win_rate * 100:.1f}%")
        col_t3.metric("Avg Duration", f"{avg_dur:.1f} wks")
        
        if trade_log:
            records = []
            for tr in trade_log:
                entry_idx = tr.get("entry_idx")
                exit_idx = tr.get("exit_idx")
                entry_date = vix_weekly.index[entry_idx] if entry_idx and entry_idx < len(vix_weekly) else None
                exit_date = vix_weekly.index[exit_idx] if exit_idx and exit_idx < len(vix_weekly) else None
                entry_eq = tr.get("entry_equity", 0)
                exit_eq = tr.get("exit_equity", 0)
                pnl = exit_eq - entry_eq if entry_eq else 0
                pnl_pct = (pnl / entry_eq * 100) if entry_eq and entry_eq > 0 else 0
                
                records.append({
                    "Entry": entry_date,
                    "Exit": exit_date,
                    "Weeks": tr.get("duration_weeks", 0),
                    "PnL $": f"${pnl:+,.0f}",
                    "PnL %": f"{pnl_pct:+.1f}%",
                    "Strike": f"{tr.get('strike_long', 0):.1f}",
                })
            
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
            
            # CSV download
            df_export = pd.DataFrame(records)
            st.download_button("📥 Download CSV", df_export.to_csv(index=False), "trade_log.csv", "text/csv")
        else:
            st.info("No trades. Try 'Synthetic (BS)' pricing or higher entry percentile.")
        # ========== END TRADE LOG ==========

        st.markdown("---")

        # ========== TRADE LOG SECTION (NEW) ==========
        st.subheader("📊 Trade Log & Statistics")
        
        _display_trade_statistics(bt)
        
        with st.expander("📋 Complete Trade Log", expanded=True):
            _display_trade_log_table(trade_log, vix_weekly)

        st.markdown("---")

        # Grid Scan section
        st.subheader("🔍 Grid Scan")

        with st.expander("Grid scan parameter ranges", expanded=False):
            ep_str = st.text_input(
                "Entry percentiles (0–1, comma-separated)",
                value="0.10,0.20,0.30,0.50,0.70,0.90",
                key="grid_entry_percentiles",
            )
            sigma_str = st.text_input(
                "Sigma multipliers for long option",
                value="0.5,0.8,1.0,1.2",
                key="grid_sigma_mults",
            )
            otm_str = st.text_input(
                "OTM distances (underlying points)",
                value="2,3,5,8,10,15",
                key="grid_otm_pts",
            )
            dte_str = st.text_input(
                "Long call DTE choices (weeks)",
                value="5,8,13,26",
                key="grid_long_dte_weeks",
            )

        entry_percentiles = _parse_float_list(ep_str)
        sigma_mults = _parse_float_list(sigma_str)
        otm_pts_list = _parse_float_list(otm_str)
        long_dte_weeks_list = _parse_int_list(dte_str)

        opt_mode = st.radio(
            "Optimization focus",
            ["Balanced: high CAGR & low Max DD", "Max CAGR only", "Min Max Drawdown only"],
            index=0,
            horizontal=True,
            key="grid_opt_mode",
        )

        if "Balanced" in opt_mode:
            criteria = "balanced"
        elif "Max CAGR" in opt_mode:
            criteria = "cagr"
        else:
            criteria = "maxdd"

        if st.button("🚀 Run Grid Scan"):
            with st.spinner("Running grid scan..."):
                grid_df = run_grid_scan(
                    vix_weekly,
                    effective_params,
                    criteria=criteria,
                    entry_grid=entry_percentiles,
                    sigma_grid=sigma_mults,
                    otm_grid=otm_pts_list,
                    dte_grid=long_dte_weeks_list,
                )
                st.session_state["grid_df"] = grid_df

        grid_df = st.session_state.get("grid_df")
        if grid_df is not None and not grid_df.empty:
            st.dataframe(grid_df, use_container_width=True)

            # Download XLSX
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                grid_df.to_excel(writer, index=False, sheet_name="grid_scan")
            buf.seek(0)
            st.download_button(
                "📥 Download Grid Scan (XLSX)",
                data=buf,
                file_name=f"{effective_params.get('mode', 'strategy')}_grid_scan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.markdown("#### Best Parameters")
            best = get_best_for_strategy(effective_params.get("mode", "diagonal"))
            if best:
                st.json(best["row"])
            else:
                st.info("No best-parameter history yet for this mode.")
        else:
            st.info("Run the grid scan to see ranked parameter combos.")

        return

    # =================================================================
    # PAGE: Trade Explorer
    # =================================================================
    if page == "Trade Explorer":
        st.title("VIX 5% Weekly — Trade Explorer")

        if not trade_log:
            st.warning("No trades to explore. Run a backtest with trades first.")
            st.info("💡 Tip: Try using 'Synthetic (BS)' pricing source with a higher entry percentile (e.g., 0.30)")
            return

        # Summary stats
        st.markdown("### Trade Summary")
        _display_trade_statistics(bt)

        # Trade log table
        st.markdown("### Complete Trade Log")
        df_trades = _build_trade_log_df(trade_log, vix_weekly)
        
        # Color code PnL
        def highlight_pnl(row):
            if "PnL ($)" in row:
                pnl = row["PnL ($)"]
                if isinstance(pnl, (int, float)) and pnl > 0:
                    return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                elif isinstance(pnl, (int, float)) and pnl < 0:
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_trades.style.apply(highlight_pnl, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Trade analysis
        st.markdown("### Trade Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### PnL Distribution")
            if "PnL ($)" in df_trades.columns:
                pnl_vals = df_trades["PnL ($)"].dropna()
                if len(pnl_vals) > 0:
                    st.bar_chart(pnl_vals.reset_index(drop=True))
        
        with col2:
            st.markdown("#### Duration Distribution")
            if "Duration (wks)" in df_trades.columns:
                dur_vals = df_trades["Duration (wks)"].dropna()
                if len(dur_vals) > 0:
                    st.bar_chart(dur_vals.value_counts().sort_index())

        # Trade on chart
        st.markdown(f"### Trades on {underlying_symbol} Chart")
        n_eq = len(equity)
        
        chart_df = pd.DataFrame({
            underlying_symbol: np.asarray(vix_weekly.iloc[:n_eq]).astype(float).ravel(),
        }, index=vix_weekly.index[:n_eq])
        
        # Add entry/exit markers as separate series
        entry_dates = []
        entry_prices = []
        exit_dates = []
        exit_prices = []
        
        for tr in trade_log:
            entry_idx = tr.get("entry_idx")
            exit_idx = tr.get("exit_idx")
            
            if entry_idx and entry_idx < len(vix_weekly):
                entry_dates.append(vix_weekly.index[entry_idx])
                entry_prices.append(float(vix_weekly.iloc[entry_idx]))
            
            if exit_idx and exit_idx < len(vix_weekly):
                exit_dates.append(vix_weekly.index[exit_idx])
                exit_prices.append(float(vix_weekly.iloc[exit_idx]))
        
        st.line_chart(chart_df)
        
        if entry_dates:
            entry_df = pd.DataFrame({"Entry": entry_prices}, index=entry_dates)
            st.caption(f"Entry points: {len(entry_dates)} trades")

        # Download
        csv = df_trades.to_csv(index=False)
        st.download_button(
            "📥 Download Trade Log (CSV)",
            data=csv,
            file_name="trade_explorer.csv",
            mime="text/csv",
        )

        return

    # =================================================================
    # PAGE: Trade Logger
    # =================================================================
    if page == "Trade Logger":
        render_trade_logger_page()
        return

    # =================================================================
    # PAGE: Regime Analysis (if regime adapter available)
    # =================================================================
    if page == "Regime Analysis" and REGIME_AVAILABLE:
        st.title("VIX 5% Weekly — Regime Analysis")
        
        if not regime_log:
            st.warning("No regime data available. Enable 'Regime-Adaptive Mode' in the sidebar and run a backtest.")
            return
        
        # Regime summary
        st.markdown("### Regime Distribution")
        regime_summary = get_regime_summary(regime_log)
        if not regime_summary.empty:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(regime_summary, use_container_width=True, hide_index=True)
            with col2:
                st.bar_chart(regime_summary.set_index("regime")["weeks"])
        
        # Trade stats by regime
        st.markdown("### Trade Performance by Entry Regime")
        regime_trade_stats = get_regime_trade_stats(trade_log)
        if not regime_trade_stats.empty:
            st.dataframe(regime_trade_stats.round(2), use_container_width=True, hide_index=True)
        
        # Regime timeline
        st.markdown("### Regime Timeline")
        if regime_log:
            regime_df = pd.DataFrame(regime_log)
            regime_df["date"] = pd.to_datetime(regime_df["date"])
            regime_df = regime_df.set_index("date")
            
            # Map regimes to numeric for visualization
            regime_map = {"Ultra Low": 1, "Low": 2, "Medium": 3, "High": 4, "Extreme": 5}
            regime_df["regime_num"] = regime_df["regime"].map(regime_map).fillna(3)
            
            st.area_chart(regime_df["regime_num"])
            st.caption("Regime levels: 1=Ultra Low, 2=Low, 3=Medium, 4=High, 5=Extreme")
        
        return

    # Fallback
    st.info("Select a page from the sidebar.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
