#!/usr/bin/env python3
"""
VIX 5% Weekly Suite — UNIFIED APPLICATION

Combines two operational modes in one Streamlit application:

RESEARCH MODE (Historical Analysis & Backtesting)
- Dashboard: Equity curves, VIX percentile visualization
- Backtester: Grid scan, parameter optimization, XLSX export
- Trade Explorer: Historical trade analysis

PAPER TRADING MODE (Live Signal Generation & Execution)
- Signal Dashboard: Thursday signal generation, V1-V5 variants
- Execution Window: Friday-Monday execution tracking
- Active Trades: Open position management
- Post-Mortem Review: Exit classification, lessons learned
- Variant Analytics: Promotion decisions, operational metrics
- System Health: Status checks

Relies on:
- core/* modules (backtester, data_loader, param_history, etc.)
- ui/* modules (sidebar, charts, tables)
- experiments/grid_scan (parameter optimization)
- Paper trading modules (regime_detector, variant_generator, trade_log, etc.)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import io
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from enums import VolatilityRegime
from variant_generator import generate_all_variants


# ============================================================
# Import Guards - Handle missing modules gracefully
# ============================================================

# Try to import backtesting modules (from original app)
try:
    from core.data_loader import load_vix_weekly, load_weekly
    from core.backtester import run_backtest
    from core.backtester_massive import run_backtest_massive
    from experiments.grid_scan import run_grid_scan
    from core.param_history import apply_best_if_requested, get_best_for_strategy
    BACKTEST_AVAILABLE = True
except ImportError as e:
    BACKTEST_AVAILABLE = False
    BACKTEST_IMPORT_ERROR = str(e)

# Try to import paper trading modules
try:
    from regime_detector import (
        classify_regime, RegimeState, VolatilityRegime,
        get_regime_color, get_regime_description
    )
    from variant_generator import (
        generate_all_variants, SignalBatch, VariantParams, VariantRole,
        get_variant_display_name, get_variant_color
    )
    from robustness_scorer import (
        calculate_robustness, batch_score_variants, RobustnessResult,
        get_robustness_color, get_robustness_label
    )
    from trade_log import (
        TradeLog, get_trade_log, Trade, TradeLeg, LegSide, LegStatus, TradeStatus
    )
    from exit_detector import (
        detect_all_exits, ExitEvent, ExitType, ExitUrgency, ExitStatus,
        get_exit_store, get_exit_urgency_color, get_exit_type_icon
    )
    from notification_engine import get_notifier
    PAPER_TRADING_AVAILABLE = True
except ImportError as e:
    PAPER_TRADING_AVAILABLE = False
    PAPER_TRADING_IMPORT_ERROR = str(e)


# ============================================================
# Configuration
# ============================================================

STORAGE_DIR = Path.home() / ".vix_suite"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_BATCH_FILE = STORAGE_DIR / "current_signal_batch.json"
REGIME_HISTORY_FILE = STORAGE_DIR / "regime_history.json"


# ============================================================
# Formatting Helpers
# ============================================================

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


def _compute_vix_percentile_local(vix_weekly: pd.Series, lookback_weeks: int) -> pd.Series:
    """Rolling percentile of underlying level (VIX / UVXY / etc)."""
    prices = vix_weekly.values.astype(float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    lb = max(1, int(lookback_weeks))

    for i in range(lb, n):
        window = prices[i - lb: i]
        out[i] = (window < prices[i]).mean()

    return pd.Series(out, index=vix_weekly.index, name="vix_pct")


def _parse_float_list(s: str) -> list:
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


def _parse_int_list(s: str) -> list:
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


# ============================================================
# Data Loading
# ============================================================

@st.cache_data(ttl=3600)
def load_underlying_data(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.Series:
    """Load weekly data for any underlying symbol."""
    try:
        import yfinance as yf
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date + timedelta(days=3),
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            return pd.Series(dtype=float)
        
        # Handle potential MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if col not in df.columns:
            return pd.Series(dtype=float)
        
        weekly = df[col].resample("W-FRI").last().dropna()
        weekly.name = symbol
        return weekly
    except Exception as e:
        st.error(f"Failed to load {symbol} data: {e}")
        return pd.Series(dtype=float)


# ============================================================
# Signal Batch Persistence (Paper Trading)
# ============================================================

def save_signal_batch(batch) -> None:
    """Save signal batch to disk."""
    with open(SIGNAL_BATCH_FILE, "w") as f:
        json.dump(batch.to_dict(), f, indent=2)


def load_signal_batch():
    """Load signal batch from disk."""
    if not PAPER_TRADING_AVAILABLE:
        return None
    
    if not SIGNAL_BATCH_FILE.exists():
        return None
    
    try:
        with open(SIGNAL_BATCH_FILE, "r") as f:
            data = json.load(f)
        
        # Reconstruct SignalBatch
        regime_data = data["regime_state"]
        regime = RegimeState(
            regime=VolatilityRegime(regime_data["regime"]),
            confidence=regime_data["confidence"],
            vix_level=regime_data["vix_level"],
            vix_percentile=regime_data["vix_percentile"],
            vix_slope=regime_data["vix_slope"],
            term_structure=regime_data["term_structure"],
            regime_age_days=regime_data["regime_age_days"],
        )
        
        variants = []
        for v_data in data["variants"]:
            v_data["role"] = VariantRole(v_data["role"])
            v_data["active_in_regimes"] = [VolatilityRegime(r) for r in v_data["active_in_regimes"]]
            v_data["suppressed_in_regimes"] = [VolatilityRegime(r) for r in v_data["suppressed_in_regimes"]]
            variants.append(VariantParams(**v_data))
        
        return SignalBatch(
            batch_id=data["batch_id"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            regime_state=regime,
            variants=variants,
            frozen=data["frozen"],
        )
    except Exception as e:
        return None


# ============================================================
# RESEARCH MODE PAGES
# ============================================================

def render_research_sidebar() -> Dict[str, Any]:
    """Build sidebar for research mode."""
    st.sidebar.markdown("## Research Settings")
    
    # Underlying selection
    underlying = st.sidebar.selectbox(
        "Underlying Symbol",
        options=["^VIX", "UVXY", "VXX"],
        index=0,
        key="research_underlying",
    )
    
    # Pricing source
    pricing_source = st.sidebar.selectbox(
        "Pricing Source",
        options=["Synthetic (BS)", "Massive historical"],
        index=0,
        key="research_pricing",
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=dt.date(2015, 1, 1),
            key="research_start",
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=dt.date.today(),
            key="research_end",
        )
    
    st.sidebar.markdown("### Capital & Risk")
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000.0,
        value=250000.0,
        step=10000.0,
        format="%.0f",
        key="research_capital",
    )
    
    alloc_pct = st.sidebar.slider(
        "Allocation (%)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        key="research_alloc",
    ) / 100.0
    
    st.sidebar.markdown("### Strategy Settings")
    
    mode = st.sidebar.selectbox(
        "Position Structure",
        options=["diagonal", "long_only"],
        format_func=lambda x: "Diagonal (LEAP + short)" if x == "diagonal" else "Long Only",
        key="research_mode",
    )
    
    entry_percentile = st.sidebar.slider(
        "Entry Percentile",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        key="research_entry_pct",
    )
    
    long_dte_weeks = st.sidebar.selectbox(
        "Long DTE (weeks)",
        options=[13, 26, 52],
        index=1,
        key="research_dte",
    )
    
    otm_pts = st.sidebar.number_input(
        "OTM Distance (pts)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        key="research_otm",
    )
    
    sigma_mult = st.sidebar.slider(
        "Sigma Multiplier",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="research_sigma",
    )
    
    # Advanced settings in expander
    with st.sidebar.expander("Advanced Settings"):
        target_mult = st.number_input(
            "Profit Target Multiple",
            min_value=1.05,
            max_value=3.0,
            value=1.20,
            step=0.05,
            key="research_target",
        )
        
        exit_mult = st.number_input(
            "Stop Multiple",
            min_value=0.1,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="research_exit",
        )
        
        risk_free = st.number_input(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=0.20,
            value=0.03,
            step=0.005,
            format="%.3f",
            key="research_rf",
        )
        
        fee_per_contract = st.number_input(
            "Fee per Contract ($)",
            min_value=0.0,
            max_value=5.0,
            value=0.65,
            step=0.05,
            key="research_fee",
        )
        
        realism = st.slider(
            "Realism Haircut",
            min_value=0.5,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="research_realism",
        )
        
        entry_lookback_weeks = st.number_input(
            "Percentile Lookback (weeks)",
            min_value=4,
            max_value=260,
            value=52,
            step=4,
            key="research_lookback",
        )
    
    return {
        "underlying_symbol": underlying,
        "pricing_source": pricing_source,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "alloc_pct": alloc_pct,
        "mode": mode,
        "entry_percentile": entry_percentile,
        "entry_lookback_weeks": entry_lookback_weeks,
        "long_dte_weeks": long_dte_weeks,
        "otm_pts": otm_pts,
        "sigma_mult": sigma_mult,
        "target_mult": target_mult,
        "exit_mult": exit_mult,
        "risk_free": risk_free,
        "fee_per_contract": fee_per_contract,
        "realism": realism,
    }


def render_dashboard(params: Dict[str, Any], data: pd.Series, bt: Dict[str, Any]):
    """Research Dashboard page."""
    st.title("📊 Research Dashboard")
    
    underlying = params.get("underlying_symbol", "^VIX")
    initial_cap = params.get("initial_capital", 250000)
    
    # Extract metrics
    equity = np.asarray(bt["equity"], dtype=float).ravel()
    final_eq = float(equity[-1]) if len(equity) > 0 else initial_cap
    cagr = _compute_cagr(equity)
    max_dd = _compute_max_dd(equity)
    total_ret = final_eq / initial_cap - 1.0 if initial_cap > 0 else 0.0
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Capital", _fmt_dollar(initial_cap))
    col2.metric("Final Equity", _fmt_dollar(final_eq))
    col3.metric("Total Return", _fmt_pct(total_ret))
    col4.metric("CAGR", _fmt_pct(cagr))
    col5.metric("Max Drawdown", _fmt_pct(max_dd))
    
    # Equity chart
    st.markdown(f"### Equity Curve vs {underlying}")
    n_eq = len(equity)
    under_vals = np.asarray(data.iloc[:n_eq]).astype(float).ravel()
    
    df_chart = pd.DataFrame({
        "Equity": equity[:n_eq],
        underlying: under_vals,
    }, index=data.index[:n_eq])
    st.line_chart(df_chart)
    
    # Percentile strip
    st.markdown(f"### 52-week {underlying} Percentile")
    pct_lb = int(params.get("entry_lookback_weeks", 52))
    vix_pct = _compute_vix_percentile_local(data, pct_lb)
    df_pct = pd.DataFrame({"Percentile": vix_pct})
    st.area_chart(df_pct)
    
    st.info("The percentile strip shows entry conditions. Low percentile = calm VIX = entry opportunity.")


def render_backtester(params: Dict[str, Any], data: pd.Series, bt: Dict[str, Any]):
    """Backtester page with grid scan."""
    st.title("🔬 Backtester")
    
    if not BACKTEST_AVAILABLE:
        st.error(f"Backtesting modules not available: {BACKTEST_IMPORT_ERROR}")
        return
    
    underlying = params.get("underlying_symbol", "^VIX")
    initial_cap = params.get("initial_capital", 250000)
    
    # Metrics
    equity = np.asarray(bt["equity"], dtype=float).ravel()
    final_eq = float(equity[-1]) if len(equity) > 0 else initial_cap
    cagr = _compute_cagr(equity)
    max_dd = _compute_max_dd(equity)
    total_ret = final_eq / initial_cap - 1.0 if initial_cap > 0 else 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Capital", _fmt_dollar(initial_cap))
    col2.metric("Final Equity", _fmt_dollar(final_eq))
    col3.metric("Total Return", _fmt_pct(total_ret))
    col4.metric("CAGR", _fmt_pct(cagr))
    col5.metric("Max Drawdown", _fmt_pct(max_dd))
    
    # Equity chart
    st.markdown(f"### Equity & {underlying}")
    n_eq = len(equity)
    df_eq = pd.DataFrame({
        "Equity": equity[:n_eq],
        underlying: np.asarray(data.iloc[:n_eq]).astype(float).ravel(),
    }, index=data.index[:n_eq])
    st.line_chart(df_eq)
    
    # Weekly PnL
    st.markdown("### Weekly PnL (realized + unrealized)")
    realized = np.asarray(bt.get("realized_weekly", []), dtype=float).ravel()
    unrealized = np.asarray(bt.get("unrealized_weekly", []), dtype=float).ravel()
    n_pnl = min(len(realized), len(unrealized), len(data))
    
    if n_pnl > 0:
        df_pnl = pd.DataFrame({
            "realized": realized[:n_pnl],
            "unrealized": unrealized[:n_pnl],
        }, index=data.index[:n_pnl])
        st.bar_chart(df_pnl)
    
    st.markdown("---")
    
    # Grid Scan
    st.subheader("🎯 Grid Scan")
    
    with st.expander("Grid Scan Parameter Ranges", expanded=True):
        ep_str = st.text_input(
            "Entry percentiles (0–1, comma-separated)",
            value="0.10,0.30,0.50,0.70,0.90",
            key="grid_entry_percentiles",
        )
        sigma_str = st.text_input(
            "Sigma multipliers",
            value="0.5,0.8,1.0",
            key="grid_sigma_mults",
        )
        otm_str = st.text_input(
            "OTM distances (points)",
            value="1,2,3,4,5,10,15",
            key="grid_otm_pts",
        )
        dte_str = st.text_input(
            "Long call DTE (weeks)",
            value="3,5,15,26",
            key="grid_long_dte_weeks",
        )
    
    entry_percentiles = _parse_float_list(ep_str)
    sigma_mults = _parse_float_list(sigma_str)
    otm_pts_list = _parse_float_list(otm_str)
    long_dte_weeks_list = _parse_int_list(dte_str)
    
    opt_mode = st.radio(
        "Optimization Focus",
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
    
    if st.button("🚀 Run Grid Scan", type="primary"):
        with st.spinner("Running grid scan..."):
            grid_df = run_grid_scan(
                data,
                params,
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
        
        # Download button
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            grid_df.to_excel(writer, index=False, sheet_name="grid_scan")
        buf.seek(0)
        
        st.download_button(
            "📥 Download Grid Scan (XLSX)",
            data=buf,
            file_name=f"{params['mode']}_grid_scan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        
        # Best params
        st.markdown("#### Best Parameters (from last scan)")
        best = get_best_for_strategy(params["mode"])
        if best:
            st.json(best["row"])
        else:
            st.info("No best-parameter history yet for this mode.")
    else:
        st.info("Click 'Run Grid Scan' to see ranked parameter combinations.")


def render_trade_explorer(params: Dict[str, Any], data: pd.Series, bt: Dict[str, Any]):
    """Trade Explorer page for historical trade analysis."""
    st.title("🔍 Trade Explorer")
    
    trade_log = bt.get("trade_log", [])
    
    if not trade_log:
        st.info("No trades in the current backtest. Adjust parameters or run a longer period.")
        return
    
    st.metric("Total Trades", len(trade_log))
    
    # Convert to DataFrame
    df_trades = pd.DataFrame(trade_log)
    
    if "entry_idx" in df_trades.columns and "exit_idx" in df_trades.columns:
        df_trades["entry_date"] = df_trades["entry_idx"].apply(
            lambda x: data.index[x] if x is not None and x < len(data) else None
        )
        df_trades["exit_date"] = df_trades["exit_idx"].apply(
            lambda x: data.index[x] if x is not None and x < len(data) else None
        )
    
    st.dataframe(df_trades, use_container_width=True)
    
    # Summary stats
    st.markdown("### Trade Statistics")
    
    win_rate = bt.get("win_rate", 0.0)
    avg_dur = bt.get("avg_trade_dur", 0.0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", _fmt_pct(win_rate))
    col2.metric("Avg Duration (weeks)", f"{avg_dur:.1f}")
    col3.metric("Total Trades", len(trade_log))


# ============================================================
# PAPER TRADING MODE PAGES
# ============================================================

def render_paper_sidebar() -> Dict[str, Any]:
    """Build sidebar for paper trading mode."""
    st.sidebar.markdown("## Paper Trading")
    
    underlying = st.sidebar.selectbox(
        "Underlying Symbol",
        options=["UVXY", "^VIX", "VXX"],
        index=0,
        key="paper_underlying",
    )
    
    st.sidebar.markdown("### Current State")
    
    batch = load_signal_batch()
    if batch:
        st.sidebar.write(f"**Batch:** {batch.batch_id[:15]}...")
        st.sidebar.write(f"**Status:** {'🔒 Frozen' if batch.frozen else '📝 Draft'}")
        st.sidebar.write(f"**Regime:** {batch.regime_state.regime.value.upper()}")
    else:
        st.sidebar.info("No active signal batch")
    
    return {
        "underlying_symbol": underlying,
    }


def render_signal_dashboard():
    """Signal Dashboard - Generate and freeze signals (Thursday 4:30 PM focus)."""
    st.title("📡 Signal Dashboard")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error(f"Paper trading modules not available: {PAPER_TRADING_IMPORT_ERROR}")
        return
    
    now = datetime.utcnow()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Time (UTC)", now.strftime("%Y-%m-%d %H:%M"))
    with col2:
        is_signal_time = now.weekday() == 3 and now.hour >= 16
        st.metric("Signal Window", "OPEN ✅" if is_signal_time else "CLOSED")
    with col3:
        days_to_thursday = (3 - now.weekday()) % 7
        st.metric("Days to Signal", days_to_thursday if days_to_thursday > 0 else "TODAY!")
    
    # Load data
    end_date = dt.date.today()
    start_date = end_date - timedelta(days=365*3)
    uvxy_data = load_underlying_data("UVXY", start_date, end_date)
    
    if uvxy_data.empty:
        st.error("No UVXY data available")
        return
    
    # Current regime
    st.markdown("---")
    st.subheader("Current Regime")
    
    regime = classify_regime(uvxy_data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div style='padding: 15px; background: {get_regime_color(regime.regime)}; "
            f"border-radius: 8px; text-align: center;'>"
            f"<h2 style='color: white; margin: 0;'>{regime.regime.value.upper()}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.metric("UVXY Level", f"${regime.vix_level:.2f}")
    with col3:
        st.metric("Percentile (52w)", f"{regime.vix_percentile:.0%}")
    with col4:
        st.metric("Confidence", f"{regime.confidence:.0%}")
    
    st.markdown(f"*{get_regime_description(regime.regime)}*")
    
    # Generate signals
    st.markdown("---")
    st.subheader("Generate Signals")
    
    if st.button("🔄 Generate New Signal Batch", type="primary"):
        with st.spinner("Generating variant signals..."):
            from utils.regime_utils import extract_current_regime

            current_regime = extract_current_regime(regime)

            # QUICK FIX: ensure scalar regime
            if hasattr(regime, "iloc"):
                regime = regime.iloc[-1]

            batch = generate_all_variants(uvxy_data, regime)

            save_signal_batch(batch)
            st.success(f"Generated batch: {batch.batch_id}")
            st.rerun()
    
    # Display current batch
    batch = load_signal_batch()
    if batch:
        st.markdown(f"### Current Batch: `{batch.batch_id}`")
        st.write(f"Generated: {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.write(f"Valid until: {batch.valid_until.strftime('%Y-%m-%d %H:%M UTC')}")
        
        # Variant cards
        for variant in batch.variants:
            # Check if variant is active in current regime
            is_active = regime.regime in variant.active_in_regimes
            
            with st.expander(
                f"{'✅' if is_active else '⛔'} {get_variant_display_name(variant.role)} "
                f"({variant.variant_id})",
                expanded=is_active
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Entry %ile:** {variant.entry_percentile:.0%}")
                    st.write(f"**Long DTE:** {variant.long_dte_weeks}w")
                with col2:
                    st.write(f"**Long Strike Offset:** {variant.long_strike_offset} pts")
                    st.write(f"**Sigma Mult:** {variant.sigma_mult}x")
                with col3:
                    st.write(f"**Target:** {variant.tp_pct:.0%}")
                    st.write(f"**Stop:** {variant.sl_pct:.0%}")
                
                # Robustness score
                robustness = calculate_robustness(variant, regime)
                st.progress(robustness.total_score / 100)
                st.caption(f"Robustness: {robustness.total_score:.0f}/100 - {get_robustness_label(robustness.total_score)}")
        
        # Freeze button
        if not batch.frozen:
            if st.button("🔒 Freeze Signal Batch"):
                batch.frozen = True
                save_signal_batch(batch)
                st.success("Batch frozen! Ready for execution window.")
                st.rerun()


def render_execution_window():
    """Execution Window - Decide whether/how to execute (Fri-Mon)."""
    st.title("⚡ Execution Window")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error(f"Paper trading modules not available")
        return
    
    batch = load_signal_batch()
    
    if not batch:
        st.warning("No signal batch available. Generate signals in Signal Dashboard first.")
        return
    
    if not batch.frozen:
        st.warning("Signal batch not frozen. Freeze it before execution.")
        return
    
    st.success(f"🔒 Frozen batch ready: {batch.batch_id}")
    
    # Show active variants for execution
    st.markdown("### Variants Ready for Execution")
    
    trade_log = get_trade_log()
    
    for variant in batch.variants:
        # Check if variant is active in current regime
        is_active = batch.regime_state.regime in variant.active_in_regimes
        
        if not is_active:
            continue
        
        robustness = calculate_robustness(variant, batch.regime_state)
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{get_variant_display_name(variant.role)}** ({variant.variant_id})")
                st.caption(f"Entry ≤{variant.entry_percentile:.0%} | OTM {variant.long_strike_offset}pts | DTE {variant.long_dte_weeks}w")
            
            with col2:
                color = get_robustness_color(robustness.total_score)
                st.markdown(
                    f"<span style='color:{color};font-weight:bold;'>"
                    f"{robustness.total_score:.0f}/100</span>",
                    unsafe_allow_html=True
                )
            
            with col3:
                if st.button(f"Execute {variant.variant_id}", key=f"exec_{variant.variant_id}"):
                    st.session_state[f"executing_{variant.variant_id}"] = True
            
            # Execution form
            if st.session_state.get(f"executing_{variant.variant_id}"):
                with st.form(key=f"exec_form_{variant.variant_id}"):
                    st.markdown("#### Log Paper Trade")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        long_strike = st.number_input("Long Strike", value=25.0, step=0.5)
                        long_price = st.number_input("Long Entry Price", value=2.50, step=0.05)
                    with col2:
                        short_strike = st.number_input("Short Strike", value=30.0, step=0.5)
                        short_price = st.number_input("Short Entry Price", value=0.80, step=0.05)
                    
                    contracts = st.number_input("Contracts", value=1, min_value=1, max_value=100)
                    
                    if st.form_submit_button("Confirm Entry"):
                        # Log the trade
                        trade = trade_log.create_trade(
                            variant_id=variant.variant_id,
                            signal_batch_id=batch.batch_id,
                            variant_role=variant.role.value,
                            structure=variant.structure,
                            underlying="UVXY",
                            suggested_tp_pct=variant.tp_pct,
                            suggested_sl_pct=variant.sl_pct,
                        )
                        
                        # Calculate expiration (approximate)
                        from datetime import timedelta
                        long_exp = (datetime.utcnow() + timedelta(weeks=variant.long_dte_weeks)).strftime("%Y-%m-%d")
                        short_exp = (datetime.utcnow() + timedelta(weeks=variant.short_dte_weeks)).strftime("%Y-%m-%d")
                        
                        # Add legs
                        trade_log.add_leg(
                            trade_id=trade.trade_id,
                            side=LegSide.LONG,
                            instrument=f"UVXY_{long_exp.replace('-','')}_C_{int(long_strike)}",
                            underlying="UVXY",
                            option_type="C",
                            strike=long_strike,
                            expiration=long_exp,
                            quantity=contracts,
                            entry_price=long_price,
                        )
                        
                        trade_log.add_leg(
                            trade_id=trade.trade_id,
                            side=LegSide.SHORT,
                            instrument=f"UVXY_{short_exp.replace('-','')}_C_{int(short_strike)}",
                            underlying="UVXY",
                            option_type="C",
                            strike=short_strike,
                            expiration=short_exp,
                            quantity=contracts,
                            entry_price=short_price,
                        )
                        
                        st.success(f"Trade logged: {trade.trade_id}")
                        st.session_state[f"executing_{variant.variant_id}"] = False
                        st.rerun()
            
            st.markdown("---")


def render_active_trades():
    """Active Trades - Monitor open positions."""
    st.title("📈 Active Trades")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = get_trade_log()
    open_trades = trade_log.get_open_trades()
    
    if not open_trades:
        st.info("No active trades. Execute signals from the Execution Window.")
        return
    
    st.metric("Open Positions", len(open_trades))
    
    # Load current data for pricing
    uvxy_data = load_underlying_data("UVXY", dt.date.today() - timedelta(days=30), dt.date.today())
    current_price = float(uvxy_data.iloc[-1]) if not uvxy_data.empty else 0.0
    
    for trade in open_trades:
        days_open = (datetime.utcnow() - trade.entry_datetime).days if trade.entry_datetime else 0
        
        # Count contracts from first leg
        leg_contracts = abs(trade.legs[0].quantity) if trade.legs else 0
        
        with st.expander(f"📊 {trade.variant_role} - {trade.trade_id[:12]}... ({days_open}d)", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Structure", trade.structure)
            with col2:
                st.metric("Days Open", days_open)
            with col3:
                st.metric("Contracts", leg_contracts)
            with col4:
                st.metric("Current UVXY", f"${current_price:.2f}")
            
            # Legs
            st.markdown("**Legs:**")
            for leg in trade.legs:
                status_icon = "🟢" if leg.status == LegStatus.OPEN else "⚫"
                st.write(
                    f"{status_icon} {leg.side.value}: Strike ${leg.strike:.2f} | "
                    f"Entry ${leg.entry_price:.2f}"
                )
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Update Prices", key=f"update_{trade.trade_id}"):
                    st.info("Update form would appear here")
            with col2:
                if st.button(f"Close Trade", key=f"close_{trade.trade_id}"):
                    st.warning("Close form would appear here")


def render_post_mortem():
    """Post-Mortem Review page."""
    st.title("📝 Post-Mortem Review")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = get_trade_log()
    closed_trades = [t for t in trade_log.trades.values() if t.status == TradeStatus.CLOSED]
    
    if not closed_trades:
        st.info("No closed trades yet. Complete some trades first.")
        return
    
    st.metric("Closed Trades for Review", len(closed_trades))
    
    for trade in closed_trades[-10:]:  # Last 10
        with st.expander(f"{trade.variant_role} - {trade.trade_id[:12]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Regime:** {trade.regime}")
                st.write(f"**P&L:** ${trade.realized_pnl:,.0f}")
            with col2:
                st.write(f"**Duration:** {(trade.exit_datetime - trade.entry_datetime).days if trade.exit_datetime else 0}d")
                st.write(f"**Interventions:** {trade.intervention_count}")
            
            # Post-mortem notes
            notes = st.text_area(
                "Lessons Learned",
                value=trade.notes or "",
                key=f"notes_{trade.trade_id}",
            )
            
            if st.button("Save Notes", key=f"save_{trade.trade_id}"):
                trade.notes = notes
                trade_log.save()
                st.success("Notes saved")


def render_variant_analytics():
    """Variant Analytics - Paper trading learning metrics."""
    st.title("📊 Variant Analytics")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = get_trade_log()
    
    st.markdown("""
    Track operational metrics to decide which variants survive the paper trading period.
    
    **Focus on:** Which variants create work? Which simplify decisions? Which break under stress?
    """)
    
    st.markdown("---")
    st.subheader("Metrics by Variant Role")
    
    for role in VariantRole:
        trades = trade_log.get_trades_by_variant(role.value)
        
        if not trades:
            continue
        
        with st.expander(f"{get_variant_display_name(role)} ({len(trades)} trades)"):
            open_count = sum(1 for t in trades if t.status == TradeStatus.OPEN)
            closed_count = sum(1 for t in trades if t.status == TradeStatus.CLOSED)
            
            total_pnl = sum(t.realized_pnl + t.unrealized_pnl for t in trades)
            total_interventions = sum(t.intervention_count for t in trades)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Open / Closed", f"{open_count} / {closed_count}")
            col2.metric("Total P&L", f"${total_pnl:,.0f}")
            col3.metric("Interventions", total_interventions)
            col4.metric("Avg Attention", f"{np.mean([t.attention_score for t in trades]):.1f}/5")
    
    # Summary
    st.markdown("---")
    summary = trade_log.get_summary()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", summary.get("total_trades", 0))
    col2.metric("Net P&L", f"${summary.get('net_pnl', 0):,.0f}")
    col3.metric("Open Positions", summary.get("open_trades", 0))


def render_system_health():
    """System Health page."""
    st.title("🏥 System Health")
    
    checks = []
    
    # Check backtesting modules
    checks.append(("Backtesting Modules", "✅ Available" if BACKTEST_AVAILABLE else f"❌ {BACKTEST_IMPORT_ERROR}"))
    
    # Check paper trading modules
    checks.append(("Paper Trading Modules", "✅ Available" if PAPER_TRADING_AVAILABLE else f"❌ {PAPER_TRADING_IMPORT_ERROR}"))
    
    # Check storage
    checks.append(("Storage Directory", f"✅ {STORAGE_DIR}" if STORAGE_DIR.exists() else "❌ Missing"))
    
    # Check data feed
    try:
        test_data = load_underlying_data("UVXY", dt.date.today() - timedelta(days=7), dt.date.today())
        checks.append(("UVXY Data Feed", f"✅ {len(test_data)} points"))
    except Exception as e:
        checks.append(("UVXY Data Feed", f"❌ {e}"))
    
    # Check signal batch
    batch = load_signal_batch()
    checks.append(("Signal Batch", f"✅ {batch.batch_id[:20]}..." if batch else "⚪ None"))
    
    # Display checks
    for name, status in checks:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.write(status)
    
    st.markdown("---")
    st.markdown("### Data Paths")
    st.code(f"""
Storage:     {STORAGE_DIR}
Trade Log:   {STORAGE_DIR / 'trade_log.json'}
Signals:     {SIGNAL_BATCH_FILE}
Regime:      {REGIME_HISTORY_FILE}
    """)


# ============================================================
# Main Application
# ============================================================

def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Mode selector
    st.sidebar.title("VIX 5% Weekly Suite")
    
    mode = st.sidebar.radio(
        "Mode",
        ["📊 Research", "📈 Paper Trading"],
        index=0,
        key="app_mode",
    )
    
    st.sidebar.markdown("---")
    
    if "Research" in mode:
        # Research mode navigation
        page = st.sidebar.radio(
            "Research Pages",
            ["Dashboard", "Backtester", "Trade Explorer"],
            index=0,
            key="research_page",
        )
        
        # Build sidebar params
        params = render_research_sidebar()
        
        # Load data
        data = load_underlying_data(
            params["underlying_symbol"],
            params["start_date"],
            params["end_date"]
        )
        
        if data.empty:
            st.error(f"No data available for {params['underlying_symbol']}")
            return
        
        # Run backtest
        bt = None
        if BACKTEST_AVAILABLE:
            pricing_source = params.get("pricing_source", "Synthetic (BS)")
            
            if pricing_source == "Massive historical":
                progress_text = st.empty()
                progress_bar = st.progress(0.0)
                
                def _progress_cb(step: int, total: int):
                    if total <= 0:
                        return
                    frac = min(max(step / float(total), 0.0), 1.0)
                    progress_bar.progress(frac)
                    progress_text.text(f"Massive backtest: {step}/{total} weeks")
                
                bt = run_backtest_massive(
                    data,
                    params,
                    symbol=params["underlying_symbol"],
                    progress_cb=_progress_cb,
                )
                progress_bar.empty()
                progress_text.empty()
            else:
                bt = run_backtest(data, params)
        else:
            # Fallback with empty results
            bt = {
                "equity": np.array([params["initial_capital"]]),
                "weekly_returns": np.array([0.0]),
                "realized_weekly": np.array([0.0]),
                "unrealized_weekly": np.array([0.0]),
                "trades": 0,
                "win_rate": 0.0,
                "avg_trade_dur": 0.0,
                "trade_log": [],
            }
        
        # Render page
        if page == "Dashboard":
            render_dashboard(params, data, bt)
        elif page == "Backtester":
            render_backtester(params, data, bt)
        elif page == "Trade Explorer":
            render_trade_explorer(params, data, bt)
    
    else:
        # Paper trading mode navigation
        page = st.sidebar.radio(
            "Paper Trading Pages",
            [
                "Signal Dashboard",
                "Execution Window",
                "Active Trades",
                "Post-Mortem Review",
                "Variant Analytics",
                "System Health",
            ],
            index=0,
            key="paper_page",
        )
        
        # Build sidebar
        render_paper_sidebar()
        
        # Render page
        if page == "Signal Dashboard":
            render_signal_dashboard()
        elif page == "Execution Window":
            render_execution_window()
        elif page == "Active Trades":
            render_active_trades()
        elif page == "Post-Mortem Review":
            render_post_mortem()
        elif page == "Variant Analytics":
            render_variant_analytics()
        elif page == "System Health":
            render_system_health()


if __name__ == "__main__":
    main()
