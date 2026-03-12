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
    from daily_signal import classify_variants, build_position_aware_email
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





# ============================================================
# Price Target Calculation Helpers
# ============================================================

def estimate_entry_credit(vix_level: float, strike_offset: float, dte_weeks: int) -> float:
    """Estimate entry credit for diagonal spread based on VIX level."""
    if vix_level < 15:
        vol_mult = 0.6
    elif vix_level < 25:
        vol_mult = 0.8
    elif vix_level < 40:
        vol_mult = 1.0
    else:
        vol_mult = 1.3
    
    otm_pct = strike_offset / max(vix_level, 10)
    short_premium = vix_level * 0.04 * vol_mult * max(0.2, 1 - otm_pct * 2)
    
    dte_factor = min(dte_weeks / 26, 1.5)
    long_cost = vix_level * 0.15 * vol_mult * dte_factor
    expected_rolls = max(1, dte_weeks // 4)
    amortized_long = (long_cost / expected_rolls) * 0.3
    
    return round(max(0.10, short_premium - amortized_long), 2)


def compute_price_targets(entry_credit: float, target_pct: float, stop_pct: float) -> dict:
    """Compute target/stop prices from entry credit."""
    target_price = round(entry_credit * (1 - target_pct), 2)
    stop_price = round(entry_credit * (1 + stop_pct), 2)
    return {
        "target": target_price,
        "stop": stop_price,
        "profit_per_contract": round(entry_credit * target_pct * 100, 0),
        "loss_per_contract": round(entry_credit * stop_pct * 100, 0),
    }


# ============================================================
# Market Data Fetcher - Real Option Prices
# ============================================================

import yfinance as yf
from functools import lru_cache
from datetime import datetime, timedelta
from real_trade_ui import render_real_trade_section

@lru_cache(maxsize=1)
def _get_option_chain_cached(symbol: str, cache_key: str):
    """Fetch option chain with caching (cache_key includes date for daily refresh)."""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None, []
        return ticker, expirations
    except Exception as e:
        print(f"Error fetching options for {symbol}: {e}")
        return None, []

def get_valid_strikes(symbol: str = "UVXY") -> list:
    """Get list of valid strikes from the market."""
    cache_key = datetime.now().strftime("%Y-%m-%d")
    ticker, expirations = _get_option_chain_cached(symbol, cache_key)
    if ticker is None or not expirations:
        return []
    try:
        chain = ticker.option_chain(expirations[0])
        return sorted(chain.calls['strike'].unique().tolist())
    except:
        return []

def round_to_valid_strike(price: float, symbol: str = "UVXY") -> float:
    """Round a price to the nearest valid option strike."""
    valid_strikes = get_valid_strikes(symbol)
    if valid_strikes:
        # Find nearest valid strike
        return min(valid_strikes, key=lambda x: abs(x - price))
    else:
        # Fallback: round to nearest 0.5 for UVXY, 1.0 for VIX
        if symbol.upper() == "UVXY":
            return round(price * 2) / 2  # Round to 0.5
        else:
            return round(price)

def get_option_price(symbol: str, strike: float, expiration_date: str, option_type: str = "call") -> dict:
    """
    Fetch real option price from Yahoo Finance.
    
    Returns dict with: bid, ask, mid, last, volume, open_interest, iv
    """
    cache_key = datetime.now().strftime("%Y-%m-%d")
    ticker, expirations = _get_option_chain_cached(symbol, cache_key)
    
    if ticker is None:
        return {"bid": 0, "ask": 0, "mid": 0, "last": 0, "error": "No data"}
    
    # Find closest expiration
    target_date = datetime.strptime(expiration_date, "%Y-%m-%d").date() if isinstance(expiration_date, str) else expiration_date
    
    best_exp = None
    min_diff = float('inf')
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        diff = abs((exp_date - target_date).days)
        if diff < min_diff:
            min_diff = diff
            best_exp = exp
    
    if not best_exp:
        return {"bid": 0, "ask": 0, "mid": 0, "last": 0, "error": "No expiration"}
    
    try:
        chain = ticker.option_chain(best_exp)
        options = chain.calls if option_type.lower() == "call" else chain.puts
        
        # Find the strike
        row = options[options['strike'] == strike]
        if row.empty:
            # Find nearest strike
            nearest = options.iloc[(options['strike'] - strike).abs().argsort()[:1]]
            if nearest.empty:
                return {"bid": 0, "ask": 0, "mid": 0, "last": 0, "error": "No strike"}
            row = nearest
        
        row = row.iloc[0]
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        mid = (bid + ask) / 2 if bid and ask else float(row.get('lastPrice', 0) or 0)
        
        return {
            "bid": bid,
            "ask": ask,
            "mid": round(mid, 2),
            "last": float(row.get('lastPrice', 0) or 0),
            "volume": int(row.get('volume', 0) or 0),
            "open_interest": int(row.get('openInterest', 0) or 0),
            "iv": float(row.get('impliedVolatility', 0) or 0),
            "expiration": best_exp,
            "strike": float(row.get('strike', strike)),
        }
    except Exception as e:
        return {"bid": 0, "ask": 0, "mid": 0, "last": 0, "error": str(e)}

def get_diagonal_prices(
    symbol: str,
    spot_price: float,
    long_offset: float,
    short_offset: float,
    long_dte_weeks: int,
    short_dte_weeks: int = 1,
) -> dict:
    """
    Get real market prices for a diagonal spread.
    
    Returns dict with long/short leg prices and net credit/debit.
    """
    from datetime import date, timedelta
    
    # Round to valid strikes
    long_strike = round_to_valid_strike(spot_price + long_offset, symbol)
    short_strike = round_to_valid_strike(spot_price + short_offset, symbol)
    
    # Calculate expiration dates
    today = date.today()
    long_exp = (today + timedelta(weeks=long_dte_weeks)).strftime("%Y-%m-%d")
    short_exp = (today + timedelta(weeks=short_dte_weeks)).strftime("%Y-%m-%d")
    
    # Fetch prices
    long_price = get_option_price(symbol, long_strike, long_exp, "call")
    short_price = get_option_price(symbol, short_strike, short_exp, "call")
    
    # Calculate net
    long_mid = long_price.get("mid", 0)
    short_mid = short_price.get("mid", 0)
    net_debit = long_mid - short_mid  # Positive = debit, Negative = credit
    
    return {
        "long_strike": long_strike,
        "long_expiration": long_price.get("expiration", long_exp),
        "long_bid": long_price.get("bid", 0),
        "long_ask": long_price.get("ask", 0),
        "long_mid": long_mid,
        "short_strike": short_strike,
        "short_expiration": short_price.get("expiration", short_exp),
        "short_bid": short_price.get("bid", 0),
        "short_ask": short_price.get("ask", 0),
        "short_mid": short_mid,
        "net_debit": round(net_debit, 2),
        "net_credit": round(-net_debit, 2) if net_debit < 0 else 0,
    }





def update_diagonal_live_prices(trade_log, position_id: str = None, symbol: str = "UVXY") -> dict:
    """
    Fetch live prices and update P&L for diagonal position(s).
    
    Args:
        trade_log: TradeLog instance
        position_id: Specific position to update (None = all open)
        symbol: Underlying symbol
    
    Returns:
        dict with update results
    """
    import yfinance as yf
    
    results = {"updated": 0, "errors": [], "positions": []}
    
    # Get positions to update
    if position_id:
        positions = [trade_log.get_diagonal(position_id)]
        positions = [p for p in positions if p]
    else:
        positions = trade_log.get_open_diagonals()
    
    if not positions:
        return results
    
    # Get current underlying price
    try:
        ticker = yf.Ticker(symbol)
        spot = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose', 0)
    except:
        spot = 0
    
    for pos in positions:
        try:
            # Fetch long leg price
            long_price_data = get_option_price(
                symbol=symbol,
                strike=pos.long_strike,
                expiration_date=pos.long_expiration,
                option_type="call"
            )
            long_mid = long_price_data.get("mid", 0)
            
            # Fetch short leg price (if active)
            short_mid = 0
            short = pos.current_short_leg
            if short and short.status == "open":
                short_price_data = get_option_price(
                    symbol=symbol,
                    strike=short.strike,
                    expiration_date=short.expiration_date,
                    option_type="call"
                )
                short_mid = short_price_data.get("mid", 0)
            
            # Update position
            trade_log.update_diagonal_prices(pos.position_id, long_mid, short_mid)
            
            # Calculate P&L
            long_pnl = (long_mid - pos.long_entry_price) * 100 * pos.contracts
            short_pnl = pos.short_pnl if short else 0
            total_pnl = long_pnl + short_pnl
            
            results["positions"].append({
                "position_id": pos.position_id,
                "variant": pos.variant_name,
                "spot": spot,
                "long_price": long_mid,
                "short_price": short_mid,
                "long_pnl": long_pnl,
                "short_pnl": short_pnl,
                "total_pnl": total_pnl,
            })
            results["updated"] += 1
            
        except Exception as e:
            results["errors"].append(f"{pos.position_id}: {str(e)}")
    
    return results


def get_position_live_summary(trade_log, symbol: str = "UVXY") -> list:
    # Auto-fetch live prices before building summary
    try:
        update_diagonal_live_prices(trade_log, symbol=symbol)
    except Exception:
        pass  # Use stored prices if fetch fails
    """
    Get live P&L summary for all open diagonal positions.
    Returns list of dicts suitable for display.
    """
    import yfinance as yf
    
    positions = trade_log.get_open_diagonals()
    if not positions:
        return []
    
    # Get spot price once
    try:
        ticker = yf.Ticker(symbol)
        spot = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose', 0)
    except:
        spot = 0
    
    summaries = []
    for pos in positions:
        short = pos.current_short_leg
        
        # Use stored current prices
        long_pnl = pos.long_pnl
        short_pnl = pos.short_pnl
        total_pnl = pos.total_pnl
        
        # Entry cost
        entry_cost = pos.long_entry_price * pos.contracts * 100
        if pos.short_legs:
            entry_cost -= pos.short_legs[0].entry_credit * pos.contracts * 100
        
        # Return %
        pnl_pct = (total_pnl / entry_cost * 100) if entry_cost > 0 else 0
        
        summaries.append({
            "Position": pos.position_id,
            "Variant": pos.variant_name,
            "Contracts": pos.contracts,
            "Long Strike": f"${pos.long_strike}",
            "Long DTE": pos.days_to_long_expiry(),
            "Long Price": f"${pos.long_current_price:.2f}",
            "Short Strike": f"${short.strike}" if short else "N/A",
            "Short DTE": short.days_to_expiry() if short else 0,
            "Short Price": f"${short.current_price:.2f}" if short else "N/A",
            "Long P&L": f"${long_pnl:+,.0f}",
            "Short P&L": f"${short_pnl:+,.0f}",
            "Total P&L": f"${total_pnl:+,.0f}",
            "Return %": f"{pnl_pct:+.1f}%",
            "Rolls": pos.total_rolls,
            "Roll Credits": f"${pos.total_roll_credits:.2f}",
            "Need Roll": "⚠️" if pos.should_roll() else "✓",
            "_total_pnl": total_pnl,  # For sorting
            "_spot": spot,
        })
    
    # Sort by variant name (V1, V2, V3, V4, V5)
    return sorted(summaries, key=lambda x: x["Variant"])



def send_signal_email_smtp(batch, regime, recipient: str = "onoshin333@gmail.com"):
    """Send position-aware email notification."""
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials missing. Set SMTP_USER and SMTP_PASS environment variables."
    
    # Get fresh trade log (clear singleton cache first)
    import trade_log as tl_module
    tl_module._trade_log_instance = None  # Clear cache
    trade_log = get_trade_log()
    
    # Classify variants with position awareness
    variant_states = classify_variants(batch, trade_log, regime.regime)
    
    # Count categories
    management = [s for s in variant_states if s.has_position]
    recommended = [s for s in variant_states if not s.has_position and s.is_recommended]
    paper_test = [s for s in variant_states if not s.has_position and not s.is_recommended]
    
    # Regime emoji
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    # Email subject
    subject = f"{regime_emoji} [PAPER TEST] VIX Signal: {regime.regime.value.upper()} ({regime.vix_percentile:.0%}) — {len(management)} Open, {len(recommended)} Entry, {len(paper_test)} Observe"
    
    # Build position-aware HTML
    html = build_position_aware_email(batch, variant_states)
    
    # Send email
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        return True, f"Email sent to {recipient}"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"


def send_roll_notification_email(positions_needing_roll, recipient: str = "onoshin333@gmail.com"):
    """Send email notification when positions need rolling."""
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from datetime import datetime, timedelta
    
    if not positions_needing_roll:
        return False, "No positions need rolling"
    
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials missing"
    
    # Get current UVXY price for suggestions
    try:
        import yfinance as yf
        ticker = yf.Ticker("UVXY")
        current_price = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice', 38.0)
    except:
        current_price = 38.0
    
    # Calculate suggested strikes
    suggested_strikes = {
        "conservative": round(current_price * 1.02, 0),
        "moderate": round(current_price * 1.05, 0),
        "aggressive": round(current_price * 1.10, 0),
    }
    
    # Next Friday
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    
    # Build position rows
    position_rows = ""
    total_contracts = 0
    for pos in positions_needing_roll:
        short = pos.current_short_leg
        dte = short.days_to_expiry() if short else 0
        dte_color = "#dc3545" if dte <= 0 else "#ffc107" if dte <= 3 else "#28a745"
        
        position_rows += f"""
        <tr style="border-bottom:1px solid #dee2e6;">
            <td style="padding:10px;font-weight:bold;">{pos.variant_name}</td>
            <td style="padding:10px;">{pos.contracts}</td>
            <td style="padding:10px;">${pos.long_strike:.0f}</td>
            <td style="padding:10px;">{f'${short.strike:.0f}' if short else 'N/A'}</td>
            <td style="padding:10px;color:{dte_color};font-weight:bold;">{dte} days</td>
            <td style="padding:10px;">{f'${short.entry_credit:.2f}' if short else '$0.00'}</td>
        </tr>
        """
        total_contracts += pos.contracts
    
    subject = f"🔄 ROLL ALERT: {len(positions_needing_roll)} position(s) expiring soon!"
    
    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:14px;background:#fff;color:#333;padding:20px;max-width:850px;margin:0 auto;">
    
    <div style="text-align:center;border-bottom:3px solid #ffc107;padding-bottom:15px;margin-bottom:20px;">
        <span style="font-size:24px;font-weight:bold;color:#ffc107;">🔄 ROLL ALERT</span><br>
        <span style="font-size:14px;color:#666;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    
    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:15px;margin-bottom:20px;">
        <strong>⚠️ {len(positions_needing_roll)} position(s) need rolling!</strong><br>
        Short legs are expiring within 3 days. Review and roll to maintain income.
    </div>
    
    <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;margin-bottom:10px;">📊 Current Market</div>
        <table style="width:100%;">
            <tr>
                <td><strong>UVXY:</strong> ${current_price:.2f}</td>
                <td><strong>Suggested Exp:</strong> {next_friday.strftime('%Y-%m-%d')} (Friday)</td>
            </tr>
        </table>
    </div>
    
    <div style="background:#e7f3ff;border:1px solid #b6d4fe;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;margin-bottom:10px;">💡 Suggested New Strikes</div>
        <table style="width:100%;">
            <tr>
                <td>🟢 Conservative (2% OTM): <strong>${suggested_strikes['conservative']:.0f}</strong></td>
                <td>🟡 Moderate (5% OTM): <strong>${suggested_strikes['moderate']:.0f}</strong></td>
                <td>🔴 Aggressive (10% OTM): <strong>${suggested_strikes['aggressive']:.0f}</strong></td>
            </tr>
        </table>
    </div>
    
    <div style="margin-bottom:20px;">
        <div style="font-weight:bold;margin-bottom:10px;">📋 Positions Needing Roll</div>
        <table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6;">
            <tr style="background:#f8f9fa;">
                <th style="padding:10px;text-align:left;">Variant</th>
                <th style="padding:10px;text-align:left;">Contracts</th>
                <th style="padding:10px;text-align:left;">Long Strike</th>
                <th style="padding:10px;text-align:left;">Short Strike</th>
                <th style="padding:10px;text-align:left;">DTE</th>
                <th style="padding:10px;text-align:left;">Credit Rcvd</th>
            </tr>
            {position_rows}
        </table>
    </div>
    
    <div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:15px;margin-bottom:20px;">
        <strong>📝 Action Required:</strong>
        <ol>
            <li>Review current short positions</li>
            <li>If short is near $0, use "🎉 Expire Profit" button</li>
            <li>Then roll into new short at suggested strike</li>
            <li>Or use "🔄 Roll" to buy back and sell new in one step</li>
        </ol>
    </div>
    
    <div style="text-align:center;color:#6c757d;font-size:12px;margin-top:20px;">
        VIX 5% Weekly Suite — Roll Notification
    </div>
    
    </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        return True, f"Roll notification sent to {recipient}"
    except Exception as e:
        return False, str(e)


def render_signal_dashboard(trade_log=None):
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
        try:
            with st.spinner("Generating variant signals..."):
                from utils.regime_utils import extract_current_regime

                current_regime = extract_current_regime(regime)

                # Ensure scalar regime
                if hasattr(regime, "iloc"):
                    regime = regime.iloc[-1]

                batch = generate_all_variants(uvxy_data, regime)
                
                if batch is None:
                    st.error("Signal generation returned None")
                elif not hasattr(batch, 'batch_id'):
                    st.error(f"Invalid batch: {type(batch)}")
                else:
                    save_signal_batch(batch)
                    st.success(f"✅ Generated batch: {batch.batch_id}")
                    st.balloons()
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    # Display current batch
    batch = load_signal_batch()
    if batch:
        st.markdown(f"### Current Batch: `{batch.batch_id}`")
        st.write(f"Generated: {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.write(f"Valid until: {batch.valid_until.strftime('%Y-%m-%d %H:%M UTC')}")
        
        # ═══════════════════════════════════════════════════════════════
        # POSITION-AWARE: Get existing positions from trade log
        # ═══════════════════════════════════════════════════════════════
        trade_log = trade_log or get_trade_log()

        open_diagonals = trade_log.get_open_diagonals()
        
        # Map variant prefixes (V1, V2, etc.) to existing positions
        existing_positions = {}
        for pos in open_diagonals:
            # Extract variant prefix from variant_id (e.g., "V1_INCOME_HARVESTER" → "V1")
            vid_upper = pos.variant_id.upper()
            for prefix in ["V1", "V2", "V3", "V4", "V5"]:
                if vid_upper.startswith(prefix) or prefix in vid_upper:
                    if prefix not in existing_positions:
                        existing_positions[prefix] = []
                    existing_positions[prefix].append(pos)
                    break
        
        # Show position summary at top if we have positions
        if existing_positions:
            total_positions = sum(len(v) for v in existing_positions.values())
            total_pnl = sum(p.total_pnl for positions in existing_positions.values() for p in positions)
            pnl_color = "green" if total_pnl >= 0 else "red"
            
            st.markdown("---")
            st.markdown("### 📊 Open Positions Summary")
            
            pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
            with pos_col1:
                st.metric("Open Positions", total_positions)
            with pos_col2:
                st.metric("Variants Held", len(existing_positions))
            with pos_col3:
                st.metric("Total P&L", f"${total_pnl:+,.0f}", 
                         delta_color="normal" if total_pnl >= 0 else "inverse")
            with pos_col4:
                # Count positions needing action
                needs_action = sum(1 for positions in existing_positions.values() 
                                   for p in positions 
                                   if p.get_health_status()["status"] in ["attention", "critical"])
                if needs_action > 0:
                    st.metric("⚠️ Need Action", needs_action)
                else:
                    st.metric("Status", "✅ All OK")
            
            st.markdown("---")
        
        # Separate variants into categories
        held_variants = []
        entry_candidates = []
        inactive_variants = []
        
        for variant in batch.variants:
            is_active = regime.regime in variant.active_in_regimes
            variant_prefix = variant.variant_id.split("-")[0].upper()  # Extract V1, V2, etc.
            has_position = variant_prefix in existing_positions
            
            if has_position:
                held_variants.append((variant, variant_prefix))
            elif is_active:
                entry_candidates.append((variant, variant_prefix))
            else:
                inactive_variants.append((variant, variant_prefix))
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 1: OPEN POSITIONS (Management Mode)
        # ═══════════════════════════════════════════════════════════════
        if held_variants:
            st.markdown("### 🔵 OPEN POSITIONS — Management Mode")
            st.caption("You already hold these variants. Showing management actions instead of entry signals.")
            
            for variant, prefix in held_variants:
                positions_for_variant = existing_positions.get(prefix, [])
                pos = positions_for_variant[0] if positions_for_variant else None
                
                if pos:
                    health = pos.get_health_status()
                    
                    # Status emoji based on health
                    if health["status"] == "critical":
                        status_emoji = "🔴"
                        status_label = "CRITICAL"
                    elif health["status"] == "attention":
                        status_emoji = "🟡"
                        status_label = "ATTENTION"
                    else:
                        status_emoji = "🟢"
                        status_label = "HEALTHY"
                    
                    pnl_str = f"${pos.total_pnl:+,.0f}"
                    short_dte = pos.days_to_short_expiry()
                    long_dte = pos.days_to_long_expiry()
                    
                    with st.expander(
                        f"{status_emoji} {get_variant_display_name(variant.role)} | "
                        f"P&L: {pnl_str} | Long: {long_dte}d | Short: {short_dte}d",
                        expanded=(health["status"] != "healthy")
                    ):
                        # Health banner
                        if health["status"] == "critical":
                            st.error(f"🚨 **{status_label}** — Immediate action required!")
                        elif health["status"] == "attention":
                            st.warning(f"⚠️ **{status_label}** — Action recommended")
                        else:
                            st.success(f"✅ **{status_label}** — No immediate action needed")
                        
                        # Position details
                        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                        with detail_col1:
                            st.markdown("**📈 Long Leg**")
                            st.write(f"Strike: ${pos.long_strike:.0f}")
                            st.write(f"DTE: {long_dte} days")
                            st.write(f"Entry: ${pos.long_entry_price:.2f}")
                        with detail_col2:
                            st.markdown("**📉 Short Leg**")
                            short_leg = pos.current_short_leg
                            if short_leg and short_leg.status == "open":
                                st.write(f"Strike: ${short_leg.strike:.0f}")
                                st.write(f"DTE: {short_leg.days_to_expiry()} days")
                                st.write(f"Credit: ${short_leg.entry_credit:.2f}")
                            else:
                                st.warning("📭 No active short")
                        with detail_col3:
                            st.markdown("**💰 P&L**")
                            pnl_color = "green" if pos.total_pnl >= 0 else "red"
                            st.markdown(f"**<span style='color:{pnl_color}'>{pnl_str}</span>**", unsafe_allow_html=True)
                            st.write(f"Credits: ${pos.total_credits_received:.2f}")
                            st.write(f"Rolls: {pos.total_rolls}")
                        with detail_col4:
                            st.markdown("**📋 Info**")
                            st.write(f"Contracts: {pos.contracts}")
                            st.write(f"Entry: {pos.entry_date}")
                            st.write(f"Regime: {pos.entry_regime}")
                        
                        # Recommendations
                        st.markdown("**🎯 Recommendations:**")
                        for alert in health.get("alerts", []):
                            st.write(f"  • {alert}")
                        
                        # Action buttons
                        st.markdown("---")
                        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                        
                        # Context-appropriate buttons
                        short_leg = pos.current_short_leg
                        
                        if health["short_status"] in ["expired", "none"]:
                            with btn_col1:
                                if st.button(f"🎉 Lock Profit", key=f"expire_{pos.position_id}", help="Short expired worthless - lock in the credit"):
                                    trade_log.expire_diagonal_short(pos.position_id)
                                    st.success("✅ Short expired - profit locked!")
                                    st.rerun()
                        
                        if health["short_status"] in ["expired", "roll_soon", "none"]:
                            with btn_col2:
                                if st.button(f"🔄 Roll Short", key=f"roll_{pos.position_id}"):
                                    st.session_state[f"rolling_{pos.position_id}"] = True
                        
                        with btn_col3:
                            if st.button(f"💲 Update Prices", key=f"prices_{pos.position_id}"):
                                st.session_state[f"updating_prices_{pos.position_id}"] = True
                        
                        with btn_col4:
                            if st.button(f"❌ Close Diagonal", key=f"close_{pos.position_id}"):
                                st.session_state[f"closing_{pos.position_id}"] = True
                        
                        # Additional buttons row for individual leg management
                        leg_col1, leg_col2, leg_col3 = st.columns(3)
                        with leg_col1:
                            if pos.current_short_leg and pos.current_short_leg.status == "open":
                                if st.button(f"📕 Close Short", key=f"close_short_{pos.position_id}"):
                                    st.session_state[f"closing_short_{pos.position_id}"] = True
                        with leg_col2:
                            if st.button(f"📘 Close Long", key=f"close_long_{pos.position_id}"):
                                st.session_state[f"closing_long_{pos.position_id}"] = True
                        
                        # Roll form
                        if st.session_state.get(f"rolling_{pos.position_id}"):
                            with st.form(key=f"roll_form_{pos.position_id}"):
                                st.markdown("#### 🔄 Roll to New Short")
                                
                                # Partial roll support
                                current_short = pos.current_short_leg
                                max_contracts = current_short.contracts if current_short else pos.contracts
                                
                                roll_col0, roll_col1, roll_col2, roll_col3 = st.columns(4)
                                with roll_col0:
                                    contracts_to_roll = st.number_input(
                                        "Contracts to Roll", 
                                        min_value=1, 
                                        max_value=max_contracts, 
                                        value=max_contracts,
                                        key=f"ctr_{pos.position_id}",
                                        help=f"Roll partial: 1-{max_contracts} contracts"
                                    )
                                with roll_col1:
                                    new_strike = st.number_input("New Strike", value=float(regime.vix_level + 2), step=1.0, key=f"rs_{pos.position_id}")
                                with roll_col2:
                                    new_exp = st.date_input("New Expiration", value=dt.date.today() + timedelta(days=7), key=f"re_{pos.position_id}")
                                with roll_col3:
                                    new_credit = st.number_input("New Credit ($)", value=0.50, step=0.05, key=f"rc_{pos.position_id}")
                                
                                exit_price = st.number_input("Buyback Price (old short)", value=0.05, step=0.01, key=f"ep_{pos.position_id}")
                                
                                if contracts_to_roll < max_contracts:
                                    st.info(f"⚠️ Partial roll: {contracts_to_roll} of {max_contracts} contracts. Remaining {max_contracts - contracts_to_roll} will stay in current short.")
                                
                                sub_col1, sub_col2 = st.columns(2)
                                with sub_col1:
                                    if st.form_submit_button("✅ Execute Roll"):
                                        trade_log.roll_diagonal_short(
                                            position_id=pos.position_id,
                                            exit_price=exit_price,
                                            new_strike=new_strike,
                                            new_expiration=new_exp.isoformat(),
                                            new_credit=new_credit,
                                            underlying_price=regime.vix_level,
                                            regime=regime.regime.value,
                                            contracts=contracts_to_roll,
                                        )
                                        if contracts_to_roll < max_contracts:
                                            st.success(f"✅ Partial roll: {contracts_to_roll} contracts rolled!")
                                        else:
                                            st.success("✅ Short rolled successfully!")
                                        st.session_state[f"rolling_{pos.position_id}"] = False
                                        st.rerun()
                                with sub_col2:
                                    if st.form_submit_button("Cancel"):
                                        st.session_state[f"rolling_{pos.position_id}"] = False
                                        st.rerun()
                        
                        # Close Short form
                        if st.session_state.get(f"closing_short_{pos.position_id}"):
                            with st.form(key=f"close_short_form_{pos.position_id}"):
                                st.markdown("#### 📕 Close Short Leg")
                                current_short = pos.current_short_leg
                                if current_short:
                                    st.write(f"Current short: ${current_short.strike} exp {current_short.expiration_date}")
                                    st.write(f"Entry credit: ${current_short.entry_credit:.2f}")
                                    
                                    cs_col1, cs_col2 = st.columns(2)
                                    with cs_col1:
                                        buyback_price = st.number_input(
                                            "Buyback Price ($)", 
                                            value=0.05, 
                                            step=0.01, 
                                            key=f"cs_bp_{pos.position_id}"
                                        )
                                    with cs_col2:
                                        close_reason = st.selectbox(
                                            "Reason",
                                            ["closed_manual", "expired_worthless", "expired_itm", "stop_loss", "take_profit"],
                                            key=f"cs_reason_{pos.position_id}"
                                        )
                                    
                                    cs_sub1, cs_sub2 = st.columns(2)
                                    with cs_sub1:
                                        if st.form_submit_button("✅ Close Short"):
                                            trade_log.close_short_leg(
                                                pos.position_id,
                                                exit_price=buyback_price,
                                                exit_reason=close_reason
                                            )
                                            st.success("✅ Short leg closed!")
                                            st.session_state[f"closing_short_{pos.position_id}"] = False
                                            st.rerun()
                                    with cs_sub2:
                                        if st.form_submit_button("Cancel"):
                                            st.session_state[f"closing_short_{pos.position_id}"] = False
                                            st.rerun()
                        
                        # Close Long form
                        if st.session_state.get(f"closing_long_{pos.position_id}"):
                            with st.form(key=f"close_long_form_{pos.position_id}"):
                                st.markdown("#### 📘 Close Long Leg")
                                st.write(f"Long: ${pos.long_strike} exp {pos.long_expiration}")
                                st.write(f"Entry price: ${pos.long_entry_price:.2f}")
                                st.write(f"Current price: ${pos.long_current_price:.2f}")
                                
                                cl_col1, cl_col2 = st.columns(2)
                                with cl_col1:
                                    sell_price = st.number_input(
                                        "Sell Price ($)", 
                                        value=float(pos.long_current_price) if pos.long_current_price else 1.0, 
                                        step=0.05, 
                                        key=f"cl_sp_{pos.position_id}"
                                    )
                                with cl_col2:
                                    close_long_reason = st.selectbox(
                                        "Reason",
                                        ["closed_manual", "expired_worthless", "expired_itm", "stop_loss", "take_profit", "roll_to_new"],
                                        key=f"cl_reason_{pos.position_id}"
                                    )
                                
                                st.warning("⚠️ Closing the long leg will close the entire diagonal position!")
                                
                                cl_sub1, cl_sub2 = st.columns(2)
                                with cl_sub1:
                                    if st.form_submit_button("✅ Close Long (& Position)"):
                                        # Close any open short first
                                        if pos.current_short_leg and pos.current_short_leg.status == "open":
                                            trade_log.close_short_leg(pos.position_id, exit_price=0.0, exit_reason="closed_with_long")
                                        # Close the position
                                        trade_log.close_diagonal(
                                            pos.position_id,
                                            long_exit_price=sell_price,
                                            exit_reason=close_long_reason
                                        )
                                        st.success("✅ Long leg and position closed!")
                                        st.session_state[f"closing_long_{pos.position_id}"] = False
                                        st.rerun()
                                with cl_sub2:
                                    if st.form_submit_button("Cancel"):
                                        st.session_state[f"closing_long_{pos.position_id}"] = False
                                        st.rerun()
                        
                        # Update prices form
                        if st.session_state.get(f"updating_prices_{pos.position_id}"):
                            with st.form(key=f"prices_form_{pos.position_id}"):
                                st.markdown("#### 💲 Update Current Prices")
                                price_col1, price_col2 = st.columns(2)
                                with price_col1:
                                    new_long_price = st.number_input("Long Price", value=pos.long_current_price or pos.long_entry_price, step=0.10, key=f"lp_{pos.position_id}")
                                with price_col2:
                                    current_short = pos.current_short_leg
                                    short_val = current_short.current_price if current_short else 0.0
                                    new_short_price = st.number_input("Short Price", value=short_val, step=0.05, key=f"sp_{pos.position_id}")
                                
                                if st.form_submit_button("Update"):
                                    trade_log.update_diagonal_prices(pos.position_id, new_long_price, new_short_price)
                                    st.success("Prices updated!")
                                    st.session_state[f"updating_prices_{pos.position_id}"] = False
                                    st.rerun()
                        
                        # Close position form
                        if st.session_state.get(f"closing_{pos.position_id}"):
                            with st.form(key=f"close_form_{pos.position_id}"):
                                st.markdown("#### ❌ Close Entire Position")
                                st.warning("This will close both the long and short legs.")
                                close_col1, close_col2 = st.columns(2)
                                with close_col1:
                                    close_long_price = st.number_input("Long Exit Price", value=pos.long_current_price or pos.long_entry_price, step=0.10, key=f"clp_{pos.position_id}")
                                with close_col2:
                                    current_short = pos.current_short_leg
                                    close_short_price = st.number_input("Short Exit Price", value=current_short.current_price if current_short else 0.0, step=0.05, key=f"csp_{pos.position_id}")
                                
                                close_reason = st.selectbox("Close Reason", ["target_hit", "stop_hit", "manual", "regime_change", "expired"], key=f"cr_{pos.position_id}")
                                
                                if st.form_submit_button("🔴 Close Position"):
                                    trade_log.close_diagonal(pos.position_id, close_long_price, close_short_price, close_reason)
                                    st.success("Position closed!")
                                    st.session_state[f"closing_{pos.position_id}"] = False
                                    st.rerun()
                        
                        st.caption(f"*Position ID: {pos.position_id}*")
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 2: ENTRY CANDIDATES (No position held, regime active)
        # ═══════════════════════════════════════════════════════════════
        if entry_candidates:
            st.markdown("### ✅ ENTRY CANDIDATES — Active in Current Regime")
            st.caption("These variants are recommended for the current regime and you have no open position.")
            
            for variant, prefix in entry_candidates:
                with st.expander(
                    f"✅ {get_variant_display_name(variant.role)} ({variant.variant_id})",
                    expanded=True
                ):
                    # Fetch real market prices
                    try:
                        short_offset = getattr(variant, 'short_strike_offset', 2)
                        market = get_diagonal_prices(
                            symbol="UVXY",
                            spot_price=regime.vix_level,
                            long_offset=variant.long_strike_offset,
                            short_offset=short_offset,
                            long_dte_weeks=variant.long_dte_weeks,
                        )
                        long_strike = market["long_strike"]
                        short_strike = market["short_strike"]
                        
                        if market["short_mid"] > 0:
                            est_credit = market["short_mid"]
                        else:
                            est_credit = estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
                        
                        long_cost = market["long_mid"] if market["long_mid"] > 0 else 0
                    except Exception:
                        long_strike = round(regime.vix_level + variant.long_strike_offset)
                        short_strike = round(regime.vix_level + getattr(variant, 'short_strike_offset', 2))
                        est_credit = estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
                        long_cost = 0
                    
                    targets = compute_price_targets(est_credit, variant.tp_pct, variant.sl_pct)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Long Strike:** ${long_strike:.0f}")
                        st.write(f"**Short Strike:** ${short_strike:.0f}")
                    with col2:
                        if long_cost > 0:
                            st.write(f"**Long Cost:** ${long_cost:.2f}")
                        st.write(f"**Short Credit:** ${est_credit:.2f}")
                        st.write(f"**Long DTE:** {variant.long_dte_weeks}w")
                    with col3:
                        st.write(f"**Target:** ${targets['target']:.2f} (+${targets['profit_per_contract']:.0f})")
                        st.write(f"**Stop:** ${targets['stop']:.2f} (-${targets['loss_per_contract']:.0f})")
                    
                    # Robustness score
                    robustness = calculate_robustness(variant, regime)
                    st.progress(robustness.total_score / 100)
                    st.caption(f"Robustness: {robustness.total_score:.0f}/100 - {get_robustness_label(robustness.total_score)}")
                    
                    # Entry condition check
                    st.markdown("---")
                    entry_check = regime.vix_percentile <= variant.entry_percentile
                    if entry_check:
                        st.success(f"✅ **ENTRY CONDITION MET** — Percentile {regime.vix_percentile:.0%} ≤ {variant.entry_percentile:.0%}")
                    else:
                        st.warning(f"⚠️ **Wait for better entry** — Percentile {regime.vix_percentile:.0%} > {variant.entry_percentile:.0%}")
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 3: INACTIVE VARIANTS (Not active in current regime)
        # ═══════════════════════════════════════════════════════════════
        if inactive_variants:
            st.markdown("### ⛔ INACTIVE VARIANTS — Not Active in Current Regime")
            st.caption("These variants are not recommended for the current regime.")
            
            for variant, prefix in inactive_variants:
                with st.expander(
                    f"⛔ {get_variant_display_name(variant.role)} ({variant.variant_id})",
                    expanded=False
                ):
                    st.info(f"This variant is active in: {', '.join([r.value.upper() for r in variant.active_in_regimes])}")
                    st.write(f"Current regime: **{regime.regime.value.upper()}**")
                    
                    # Still show parameters for reference
                    try:
                        short_offset = getattr(variant, 'short_strike_offset', 2)
                        long_strike = round(regime.vix_level + variant.long_strike_offset)
                        short_strike = round(regime.vix_level + short_offset)
                        est_credit = estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
                    except:
                        long_strike = 0
                        short_strike = 0
                        est_credit = 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Long Strike: ${long_strike:.0f}")
                        st.write(f"Long DTE: {variant.long_dte_weeks}w")
                    with col2:
                        st.write(f"Short Strike: ${short_strike:.0f}")
                        st.write(f"Est Credit: ${est_credit:.2f}")
        
        # Freeze button
        if not batch.frozen:
            if st.button("🔒 Freeze Signal Batch"):
                batch.frozen = True
                save_signal_batch(batch)
                st.success("Batch frozen! Ready for execution window.")
                st.rerun()
        
        # ===== EMAIL SECTION =====
        st.markdown("---")
        st.subheader("📧 Thursday Email")
        
        col_email1, col_email2 = st.columns([2, 1])
        with col_email1:
            recipient = st.text_input("Recipient Email", value="onoshin333@gmail.com", key="email_recipient")
        with col_email2:
            force_send = st.checkbox("Send even if no active signal", key="force_email")
        
        # Check if any variant is active
        has_active = any(regime.regime in v.active_in_regimes for v in batch.variants)
        
        if st.button("📤 Send Thursday Email Now", type="primary"):
            if not has_active and not force_send:
                st.warning("⚠️ No active signal. Check 'Send even if no active signal' to send anyway.")
            else:
                success, msg = send_signal_email_smtp(batch, regime, recipient)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        # Quick copy button
        if st.button("📋 Copy Signal Summary"):
            summary_text = f"Batch: {batch.batch_id}\nRegime: {regime.regime.value}\nUVXY: ${regime.vix_level:.2f} ({regime.vix_percentile:.0%})\n\nActive Variants:\n"
            for v in batch.variants:
                if regime.regime in v.active_in_regimes:
                    summary_text += f"- {get_variant_display_name(v.role)}: entry ≤{v.entry_percentile:.0%}, +{v.long_strike_offset}pts, {v.long_dte_weeks}w\n"
            st.code(summary_text)


def render_execution_window(trade_log=None):
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
    
    trade_log = trade_log or get_trade_log()

    
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
                            trade_id=trade.position_id,
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
                            trade_id=trade.position_id,
                            side=LegSide.SHORT,
                            instrument=f"UVXY_{short_exp.replace('-','')}_C_{int(short_strike)}",
                            underlying="UVXY",
                            option_type="C",
                            strike=short_strike,
                            expiration=short_exp,
                            quantity=contracts,
                            entry_price=short_price,
                        )
                        
                        st.success(f"Trade logged: {trade.position_id}")
                        st.session_state[f"executing_{variant.variant_id}"] = False
                        st.rerun()
            
            st.markdown("---")


def render_active_trades(trade_log=None):
    """Active Trades - Monitor open positions."""
    st.title("📈 Active Trades")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = trade_log or get_trade_log()

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
        
        with st.expander(f"📊 {trade.variant_role} - {trade.position_id[:12]}... ({days_open}d)", expanded=True):
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
                status_icon = "🟢" if str(leg.status).upper() in ("OPEN", "LEGSTATUS.OPEN") else "⚫"
                st.write(
                    f"{status_icon} {leg.side.value}: Strike ${leg.strike:.2f} | "
                    f"Entry ${leg.entry_price:.2f}"
                )
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Update Prices", key=f"update_{trade.position_id}"):
                    st.info("Update form would appear here")
            with col2:
                if st.button(f"Close Trade", key=f"close_{trade.position_id}"):
                    st.warning("Close form would appear here")


def render_post_mortem(trade_log=None):
    """Post-Mortem Review page."""
    st.title("📝 Post-Mortem Review")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = trade_log or get_trade_log()

    closed_trades = [t for t in trade_log.diagonal_positions.values() if t.status == "closed"]
    
    if not closed_trades:
        st.info("No closed trades yet. Complete some trades first.")
        return
    
    st.metric("Closed Trades for Review", len(closed_trades))
    
    for trade in closed_trades[-10:]:  # Last 10
        with st.expander(f"{trade.variant_name} - {trade.position_id[:12]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Regime:** {trade.entry_regime}")
                st.write(f"**P&L:** ${trade.total_pnl:,.0f}")
            with col2:
                duration = 0
                if trade.exit_date and trade.entry_date:
                    from datetime import datetime
                    try:
                        entry = datetime.strptime(trade.entry_date, "%Y-%m-%d")
                        exit = datetime.strptime(trade.exit_date, "%Y-%m-%d")
                        duration = (exit - entry).days
                    except:
                        pass
                st.write(f"**Duration:** {duration}d")
                st.write(f"**Rolls:** {trade.total_rolls}")
            
            # Post-mortem notes
            notes = st.text_area(
                "Lessons Learned",
                value=trade.notes or "",
                key=f"notes_{trade.position_id}",
            )
            
            if st.button("Save Notes", key=f"save_{trade.position_id}"):
                trade.notes = notes
                trade_log.save()
                st.success("Notes saved")


def render_variant_analytics(trade_log=None):
    """Variant Analytics — full spreadsheet with long duration & P&L analysis."""
    st.title("📊 Variant Analytics")
    trade_log = trade_log or get_trade_log()

    import pandas as pd
    from datetime import date, datetime

    all_pos   = list(trade_log.diagonal_positions.values())
    open_pos  = [p for p in all_pos if p.status == "open"]
    closed_pos = [p for p in all_pos if p.status == "closed"]

    if not all_pos:
        st.info("No positions yet.")
        return

    today = date.today()

    def _net_credits(p):
        """Net short credits — paper uses short_pnl, real uses net_short_credits."""
        for attr in ("net_short_credits", "short_pnl"):
            if hasattr(p, attr):
                v = getattr(p, attr)
                return v() if callable(v) else float(v)
        # fallback: total_short_credits + total_roll_credits
        return (getattr(p, "total_short_credits", 0.0)
                + getattr(p, "total_roll_credits", 0.0))

    def _long_pnl(p):
        if hasattr(p, "long_pnl"):
            v = p.long_pnl
            return v() if callable(v) else float(v)
        price = getattr(p, "long_current_price", 0.0) or 0.0
        entry = getattr(p, "long_entry_price", 0.0) or 0.0
        c     = getattr(p, "contracts", 1)
        return (price - entry) * c * 100

    def _total_pnl(p):
        if hasattr(p, "total_pnl"):
            v = p.total_pnl
            return v() if callable(v) else float(v)
        return _long_pnl(p) + _net_credits(p) - _commissions(p)

    def _commissions(p):
        return float(getattr(p, "total_commissions", 0.0) or 0.0)

    def _total_rolls(p):
        v = getattr(p, "total_rolls", None)
        if v is not None:
            return v() if callable(v) else int(v)
        return len(p.roll_history) if hasattr(p, "roll_history") else 0

    def _long_cost(p):
        if hasattr(p, "long_cost"):
            v = p.long_cost
            return v() if callable(v) else float(v)
        # paper: long_entry_price * contracts * 100 + first short credit removed
        fill = getattr(p, "long_fill_price",
               getattr(p, "long_entry_price", 0.0)) or 0.0
        c    = getattr(p, "contracts", 1)
        comm = getattr(p, "long_commission",
               getattr(p, "fee_per_contract", 0.65))
        return fill * c * 100 + comm * c

    def _gross_credits(p):
        if hasattr(p, "gross_short_credits"):
            v = p.gross_short_credits
            return v() if callable(v) else float(v)
        # paper: total_short_credits (raw, before buybacks)
        return float(getattr(p, "total_short_credits",
                     getattr(p, "total_credits_received",
                     _net_credits(p))))

    def _buybacks(p):
        if hasattr(p, "total_buybacks"):
            v = p.total_buybacks
            return v() if callable(v) else float(v)
        return max(0.0, _gross_credits(p) - _net_credits(p))

    def _coverage(p):
        if hasattr(p, "short_coverage_pct"):
            v = p.short_coverage_pct
            return v() if callable(v) else float(v)
        lc = _long_cost(p)
        return min(100.0, _net_credits(p) / lc * 100) if lc > 0 else 0.0

    # ═══════════════════════════════════════════════════════
    # SUMMARY BAR
    # ═══════════════════════════════════════════════════════
    total_pnl   = sum(_total_pnl(p)   for p in all_pos)
    total_long  = sum(_long_pnl(p)    for p in all_pos)
    total_short = sum(_net_credits(p) for p in all_pos)
    total_comm  = sum(_commissions(p) for p in all_pos)
    total_rolls = sum(_total_rolls(p) for p in all_pos)
    win_rate    = (sum(1 for p in closed_pos if _total_pnl(p) > 0)
                   / len(closed_pos) * 100) if closed_pos else 0.0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Positions", len(all_pos))
    c2.metric("Open / Closed",   f"{len(open_pos)} / {len(closed_pos)}")
    c3.metric("Total P&L",       f"${total_pnl:+,.0f}")
    c4.metric("Short Credits",   f"${total_short:+,.0f}")
    c5.metric("Total Rolls",     total_rolls)
    c6.metric("Win Rate",        f"{win_rate:.0f}%")

    st.divider()

    # ═══════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Position Scorecard",
        "📈 Long Duration Analysis",
        "💰 P&L Breakdown",
        "🔄 Roll Efficiency",
    ])

    # ── Build core analytics rows ────────────────────────
    rows = []
    for pos in sorted(all_pos, key=lambda p: p.variant_id):
        try:
            from datetime import date as _date
            long_entry   = _date.fromisoformat(str(pos.entry_date)[:10])
            long_expiry  = _date.fromisoformat(str(pos.long_expiration)[:10])
            _today       = _date.today()
            long_total   = max(1, (long_expiry - long_entry).days)
            long_elapsed = max(1, (_today - long_entry).days)
            long_remaining = max(0, (long_expiry - _today).days)
            long_pct_used  = long_elapsed / long_total * 100
        except Exception as _e:
            long_total = long_elapsed = long_remaining = 1
            long_pct_used = 0.0

        nc  = _net_credits(pos)
        lc  = _long_cost(pos)
        gc  = _gross_credits(pos)
        bb  = _buybacks(pos)
        lpnl = _long_pnl(pos)
        tpnl = _total_pnl(pos)
        cov  = _coverage(pos)
        cpd  = nc / long_elapsed if long_elapsed > 0 else 0
        proj_total = cpd * long_total
        be_days = int(lc / cpd) if cpd > 0 else 9999
        be_date = (long_entry + __import__("datetime").timedelta(days=be_days)
                   ).isoformat() if be_days < 9999 else "N/A"
        bb_drag = bb / gc * 100 if gc > 0 else 0
        tr = _total_rolls(pos)
        tc = _commissions(pos)

        short = pos.current_short_leg
        rows.append({
            "_pos":          pos,
            "Variant":       pos.variant_name,
            "Status":        pos.status.upper(),
            "Entry":         pos.entry_date,
            "Regime":        getattr(pos, "entry_regime", ""),
            "Long Strike":   f"${pos.long_strike:.0f}",
            "Long Exp":      pos.long_expiration,
            "Long DTE":      long_remaining,
            "Long % Used":   f"{long_pct_used:.0f}%",
            "Long Cost":     lc,
            "Long P&L":      lpnl if pos.long_current_price > 0 else 0.0,
            "Long Cost $":   lc,
            "Long Fill":     f"${getattr(pos,'long_fill_price', getattr(pos,'long_entry_price',0)):.2f}",
            "Short Strike":  f"${short.strike:.0f}" if short else "—",
            "Short Exp":     short.expiration_date if short else "—",
            "Short DTE":     pos.days_to_expiry() if hasattr(pos,"days_to_expiry") else 0,
            "Short Fill":    f"${getattr(short, 'fill_price', getattr(short, 'entry_credit', 0.0)):.2f}" if short else "—",
            "Gross Credits": gc,
            "Buy-backs":     bb,
            "Net Credits":   nc,
            "BB Drag%":      f"{bb_drag:.0f}%",
            "Coverage%":     cov,
            "Long P&L $":    lpnl,
            "Short P&L $":   nc,
            "Total P&L $":   tpnl,
            "Return%":       tpnl / lc * 100 if lc > 0 else 0,
            "$/day":         cpd,
            "Proj Total":    proj_total,
            "BE Date":       be_date,
            "Days Open":     long_elapsed,
            "Contracts":     pos.contracts,
            "Rolls":         tr,
            "Commission":    tc,
        })

    # ═══════════════════════════════════════════════════════
    # TAB 1 — POSITION SCORECARD
    # ═══════════════════════════════════════════════════════
    with tab1:
        st.markdown("##### All Positions — Scorecard")
        display_cols = [
            "Variant","Status","Entry","Regime","Contracts",
            "Long Strike","Long Exp","Long DTE","Long % Used",
            "Long Cost $",
            "Short Strike","Short Exp","Short DTE",
            "Coverage%","Total P&L $","Return%","Rolls",
        ]
        df = pd.DataFrame(rows)[display_cols].copy()
        df["Total P&L $"] = df["Total P&L $"].apply(lambda v: f"${v:+,.0f}")
        df["Return%"]     = df["Return%"].apply(lambda v: f"{v:+.1f}%")
        df["Coverage%"]   = df["Coverage%"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-variant summary
        st.markdown("##### By Variant")
        var_rows = []
        by_var = {}
        for r in rows:
            v = r["Variant"]
            by_var.setdefault(v, []).append(r)
        for v, vrows in sorted(by_var.items()):
            var_rows.append({
                "Variant":       v,
                "Positions":     len(vrows),
                "Open":          sum(1 for r in vrows if r["Status"]=="OPEN"),
                "Total P&L":     f"${sum(r['Total P&L $'] for r in vrows):+,.0f}",
                "Long P&L":      f"${sum(r['Long P&L $'] for r in vrows):+,.0f}",
                "Short Credits": f"${sum(r['Net Credits'] for r in vrows):+,.0f}",
                "Avg $/day":     f"${sum(r['$/day'] for r in vrows)/len(vrows):.2f}",
                "Avg Coverage%": f"{sum(r['Coverage%'] for r in vrows)/len(vrows):.0f}%",
                "Total Rolls":   sum(r["Rolls"] for r in vrows),
                "Commission":    f"${sum(r['Commission'] for r in vrows):.2f}",
            })
        st.dataframe(pd.DataFrame(var_rows),
                     use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════
    # TAB 2 — LONG DURATION ANALYSIS
    # ═══════════════════════════════════════════════════════
    with tab2:
        st.markdown("##### Long Leg Duration vs Credit Recovery")
        dur_cols = [
            "Variant","Status","Long Fill","Long Exp",
            "Long DTE","Long % Used","Days Open",
            "Long Cost","Long P&L $","Net Credits",
            "Coverage%","$/day","Proj Total","BE Date",
        ]
        df2 = pd.DataFrame(rows)[dur_cols].copy()
        df2["Long Cost"]    = df2["Long Cost"].apply(lambda v: f"${v:,.0f}")
        df2["Long P&L $"]   = df2["Long P&L $"].apply(lambda v: f"${v:+,.0f}")
        df2["Net Credits"]  = df2["Net Credits"].apply(lambda v: f"${v:,.0f}")
        df2["Coverage%"]    = df2["Coverage%"].apply(lambda v: f"{v:.0f}%")
        df2["$/day"]        = df2["$/day"].apply(lambda v: f"${v:.2f}")
        df2["Proj Total"]   = df2["Proj Total"].apply(lambda v: f"${v:,.0f}")
        st.dataframe(df2, use_container_width=True, hide_index=True)

        # Duration buckets
        st.markdown("##### P&L by Long Expiry Duration (weeks remaining)")
        buckets = {"0-4w": [], "4-8w": [], "8-13w": [], "13-26w": [], "26w+": []}
        for r in rows:
            d = r["Long DTE"]
            if   d <= 28:  buckets["0-4w"].append(r)
            elif d <= 56:  buckets["4-8w"].append(r)
            elif d <= 91:  buckets["8-13w"].append(r)
            elif d <= 182: buckets["13-26w"].append(r)
            else:          buckets["26w+"].append(r)
        brows = []
        for bucket, blist in buckets.items():
            if not blist: continue
            brows.append({
                "Long DTE Bucket": bucket,
                "Positions":       len(blist),
                "Avg Long P&L":    f"${sum(r['Long P&L $'] for r in blist)/len(blist):+,.0f}",
                "Avg Net Credits": f"${sum(r['Net Credits'] for r in blist)/len(blist):,.0f}",
                "Avg Total P&L":   f"${sum(r['Total P&L $'] for r in blist)/len(blist):+,.0f}",
                "Avg Coverage%":   f"{sum(r['Coverage%'] for r in blist)/len(blist):.0f}%",
                "Avg $/day":       f"${sum(r['$/day'] for r in blist)/len(blist):.2f}",
            })
        if brows:
            st.dataframe(pd.DataFrame(brows),
                         use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════
    # TAB 3 — P&L BREAKDOWN
    # ═══════════════════════════════════════════════════════
    with tab3:
        st.markdown("##### Full P&L Breakdown per Position")
        pnl_cols = [
            "Variant","Status","Contracts",
            "Long Cost","Long P&L $",
            "Gross Credits","Buy-backs","BB Drag%","Net Credits",
            "Commission","Total P&L $","Return%","Coverage%",
        ]
        df3 = pd.DataFrame(rows)[pnl_cols].copy()
        for col in ["Long Cost","Long P&L $","Gross Credits",
                    "Buy-backs","Net Credits","Commission","Total P&L $"]:
            df3[col] = df3[col].apply(
                lambda v: f"${v:+,.0f}" if isinstance(v, (int,float)) else v)
        df3["Return%"]  = df3["Return%"].apply(lambda v: f"{v:+.1f}%")
        df3["Coverage%"]= df3["Coverage%"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(df3, use_container_width=True, hide_index=True)

        # Totals row
        st.markdown("##### Totals")
        t1,t2,t3,t4,t5,t6 = st.columns(6)
        t1.metric("Long Cost",     f"${sum(r['Long Cost']     for r in rows):,.0f}")
        t2.metric("Long P&L",      f"${sum(r['Long P&L $']   for r in rows):+,.0f}")
        t3.metric("Gross Credits", f"${sum(r['Gross Credits'] for r in rows):,.0f}")
        t4.metric("Net Credits",   f"${sum(r['Net Credits']   for r in rows):,.0f}")
        t5.metric("Commission",    f"${sum(r['Commission']    for r in rows):.2f}")
        t6.metric("Total P&L",     f"${sum(r['Total P&L $']  for r in rows):+,.0f}")

    # ═══════════════════════════════════════════════════════
    # TAB 4 — ROLL EFFICIENCY
    # ═══════════════════════════════════════════════════════
    with tab4:
        st.markdown("##### Roll Efficiency by Position")
        roll_cols = [
            "Variant","Status","Days Open","Rolls",
            "Gross Credits","Buy-backs","BB Drag%","Net Credits",
            "$/day","Proj Total","BE Date","Commission",
        ]
        df4 = pd.DataFrame(rows)[roll_cols].copy()
        for col in ["Gross Credits","Buy-backs","Net Credits","Proj Total","Commission"]:
            df4[col] = df4[col].apply(
                lambda v: f"${v:,.0f}" if isinstance(v,(int,float)) else v)
        df4["$/day"] = df4["$/day"].apply(lambda v: f"${v:.2f}")
        st.dataframe(df4, use_container_width=True, hide_index=True)

        # Roll detail — all rolls across all positions
        all_rolls = []
        for pos in all_pos:
            for r in pos.roll_history:
                all_rolls.append({
                    "Date":       getattr(r, "roll_date", getattr(r, "exit_date", "")),
                    "Variant":    pos.variant_name,
                    "Old Strike": f"${getattr(r, 'old_strike', 0):.0f}",
                    "Old Exit":   f"${getattr(r, 'old_exit_price', getattr(r, 'exit_price', 0.0)):.2f}",
                    "New Strike": f"${getattr(r, 'new_strike', 0):.0f}",
                    "New Credit": f"${getattr(r, 'new_credit', 0.0):.2f}",
                    "Net Credit": f"${getattr(r, 'roll_credit', 0.0):.2f}",
                    "Underlying": f"${getattr(r, 'underlying_price', 0.0):.2f}",
                    "Regime":     getattr(r, "regime", ""),
                })
        if all_rolls:
            st.markdown("##### All Roll History")
            st.dataframe(pd.DataFrame(all_rolls),
                         use_container_width=True, hide_index=True)


def render_system_health(trade_log=None):
    """System Health page with Backup Management."""
    st.title("🏥 System Health")
    
    # ═══════════════════════════════════════════════════════════════
    # SYSTEM CHECKS
    # ═══════════════════════════════════════════════════════════════
    st.subheader("System Status")
    
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
    
    # Check backup system
    try:
        from backup_manager import get_backup_manager
        backup_mgr = get_backup_manager()
        backup_status = backup_mgr.get_status()
        checks.append(("Backup System", f"✅ {backup_status['total_backups']} backups"))
        BACKUP_AVAILABLE = True
    except ImportError:
        checks.append(("Backup System", "❌ backup_manager.py not found"))
        BACKUP_AVAILABLE = False
    except Exception as e:
        checks.append(("Backup System", f"❌ {e}"))
        BACKUP_AVAILABLE = False
    
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
Backups:     {STORAGE_DIR / 'backups'}
    """)
    
    # ═══════════════════════════════════════════════════════════════
    # BACKUP MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("💾 Backup Management")
    
    if not BACKUP_AVAILABLE:
        st.warning("Backup system not available. Add backup_manager.py to enable.")
        return
    
    # Backup status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Backups", backup_status['total_backups'])
    with col2:
        latest = backup_status.get('latest_backup')
        if latest:
            latest_time = latest.get('timestamp', 'Unknown')
            st.metric("Latest Backup", latest_time)
        else:
            st.metric("Latest Backup", "None")
    with col3:
        # Check both local folder sync AND Google Drive API
        gdrive_api_enabled = False
        gdrive_backup_count = 0
        try:
            from gdrive_backup import GDriveBackupManager
            gdrive_mgr = GDriveBackupManager()
            gdrive_status = gdrive_mgr.get_backup_status()
            gdrive_api_enabled = gdrive_status.get('folder_id') is not None
            gdrive_backup_count = gdrive_status.get('backup_count', 0)
        except:
            pass
        
        cloud_enabled = "✅ Yes" if (backup_status['cloud_sync_enabled'] or gdrive_api_enabled) else "❌ No"
        st.metric("Cloud Sync", cloud_enabled)
    with col4:
        st.metric("Max Kept", backup_status['max_local_backups'])
    
    # Cloud sync paths
    if backup_status['cloud_sync_paths']:
        st.success(f"☁️ Cloud sync enabled: {', '.join(backup_status['cloud_sync_paths'])}")
    elif gdrive_api_enabled:
        st.success(f"☁️ Google Drive API enabled ({gdrive_backup_count} backups in cloud)")
    else:
        st.info("💡 To enable cloud sync, create a folder named 'VIX_Suite_Backup' in Dropbox, Google Drive, or OneDrive")
    
    # Backup actions
    st.markdown("#### Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📸 Create Backup Now", type="primary"):
            try:
                result = backup_mgr.backup_now(reason="manual_ui")
                if result.get('files'):
                    st.success(f"✅ Backup created: {len(result['files'])} files saved")
                    if result.get('cloud_synced'):
                        st.info(f"☁️ Synced to: {', '.join(result['cloud_synced'])}")
                else:
                    st.warning("No files to backup")
            except Exception as e:
                st.error(f"Backup failed: {e}")
    
    with action_col2:
        if st.button("📤 Export to CSV"):
            try:
                csv_path = backup_mgr.export_trades_csv()
                st.success(f"✅ Exported to: {csv_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with action_col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════
    # DOWNLOAD SECTION - Download trade_log.json to sync across machines
    # ═══════════════════════════════════════════════════════════════
    st.markdown("#### 📥 Download Data")
    st.caption("Download files to sync between Ubuntu server and Mac development machine")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        # Download trade_log.json
        trade_log_path = STORAGE_DIR / "trade_log.json"
        if trade_log_path.exists():
            with open(trade_log_path, "r") as f:
                trade_log_content = f.read()
            st.download_button(
                label="📒 Download trade_log.json",
                data=trade_log_content,
                file_name="trade_log.json",
                mime="application/json",
                help="Download to sync positions with Mac",
            )
        else:
            st.warning("trade_log.json not found")
    
    with download_col2:
        # Download current_signal_batch.json
        signal_batch_path = STORAGE_DIR / "current_signal_batch.json"
        if signal_batch_path.exists():
            with open(signal_batch_path, "r") as f:
                signal_content = f.read()
            st.download_button(
                label="📊 Download signal_batch.json",
                data=signal_content,
                file_name="current_signal_batch.json",
                mime="application/json",
                help="Download current signals",
            )
        else:
            st.info("No signal batch yet")
    
    with download_col3:
        # Download all as ZIP
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add trade_log.json
            if trade_log_path.exists():
                zf.write(trade_log_path, "trade_log.json")
            # Add signal batch
            if signal_batch_path.exists():
                zf.write(signal_batch_path, "current_signal_batch.json")
            # Add regime history if exists
            regime_path = STORAGE_DIR / "regime_history.json"
            if regime_path.exists():
                zf.write(regime_path, "regime_history.json")
        
        zip_buffer.seek(0)
        st.download_button(
            label="📦 Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"vix_suite_backup_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            help="Download all data files as ZIP",
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════
    # UPLOAD SECTION - Upload from Mac to sync
    # ═══════════════════════════════════════════════════════════════
    st.markdown("#### 📤 Upload Data")
    st.caption("Upload trade_log.json from Mac to sync positions to Ubuntu")
    
    uploaded_file = st.file_uploader(
        "Upload trade_log.json",
        type=["json"],
        help="Upload a trade_log.json file to replace current data",
        key="upload_trade_log",
    )
    
    if uploaded_file is not None:
        try:
            import json
            uploaded_content = uploaded_file.read().decode("utf-8")
            uploaded_data = json.loads(uploaded_content)
            
            # Validate structure
            if "diagonal_positions" not in uploaded_data:
                st.error("❌ Invalid file: missing 'diagonal_positions' key")
            else:
                positions_count = len(uploaded_data.get("diagonal_positions", {}))
                st.info(f"📋 File contains {positions_count} diagonal position(s)")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    for pos_id, pos in uploaded_data.get("diagonal_positions", {}).items():
                        st.write(f"• **{pos_id}**: {pos.get('variant_name', 'Unknown')} - {pos.get('status', 'unknown')}")
                
                # Confirm upload
                upload_confirm = st.checkbox("I confirm I want to replace current data", key="upload_confirm")
                if st.button("⬆️ Upload & Replace", disabled=not upload_confirm, type="primary"):
                    # Backup current first
                    try:
                        backup_mgr.backup_now(reason="before_upload")
                    except:
                        pass
                    
                    # Save uploaded file
                    with open(trade_log_path, "w") as f:
                        f.write(uploaded_content)
                    st.success("✅ trade_log.json updated! Refresh the page to see changes.")
                    st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # List recent backups
    st.markdown("#### Recent Backups")
    backups = backup_mgr.list_backups()[:10]  # Show last 10
    
    if not backups:
        st.info("No backups found. Click 'Create Backup Now' to create one.")
    else:
        for i, backup in enumerate(backups):
            timestamp = backup.get('timestamp', 'Unknown')
            reason = backup.get('reason', 'unknown')
            files = backup.get('files', [])
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"📁 `{timestamp}`")
            with col2:
                st.write(f"_{reason}_")
            with col3:
                st.write(f"{len(files)} files")
            with col4:
                if st.button("Restore", key=f"restore_{i}"):
                    st.session_state[f"confirm_restore_{i}"] = True
            
            # Confirmation dialog
            if st.session_state.get(f"confirm_restore_{i}"):
                st.warning(f"⚠️ This will replace current data with backup from {timestamp}")
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ Yes, Restore", key=f"confirm_yes_{i}"):
                        try:
                            result = backup_mgr.restore_backup(backup['path'], confirm=True)
                            if result['success']:
                                st.success(f"✅ Restored: {', '.join(result['restored'])}")
                                st.session_state[f"confirm_restore_{i}"] = False
                                st.rerun()
                            else:
                                st.error(f"Restore failed: {result.get('errors')}")
                        except Exception as e:
                            st.error(f"Restore failed: {e}")
                with confirm_col2:
                    if st.button("❌ Cancel", key=f"confirm_no_{i}"):
                        st.session_state[f"confirm_restore_{i}"] = False
                        st.rerun()


# ============================================================
# Main Application
# ============================================================





def _render_diagonal_positions(trade_log):
    """Render diagonal positions with roll tracking."""
    st.subheader("🔄 Diagonal Positions with Roll Tracking")
    
    # Get diagonal positions and health summary
    diagonals = trade_log.get_all_diagonals()
    open_diagonals = trade_log.get_open_diagonals()
    health_summary = trade_log.get_position_health_summary()
    
    # Health status metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Open", health_summary["total"])
    with col2:
        st.metric("🟢 Healthy", health_summary["healthy"])
    with col3:
        st.metric("🟡 Attention", health_summary["attention"], 
                  delta="⚠️" if health_summary["attention"] > 0 else None)
    with col4:
        st.metric("🔴 Critical", health_summary["critical"],
                  delta="🚨" if health_summary["critical"] > 0 else None)
    with col5:
        roll_stats = trade_log.get_roll_summary()
        st.metric("Roll Credits", f"${roll_stats['total_roll_credits']:,.2f}")
    with col6:
        total_pnl = sum(p.total_pnl for p in open_diagonals)
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${total_pnl:,.0f}", delta_color=pnl_color)
    
    # Action needed summary
    st.markdown("### 🎯 Actions Needed")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        need_short = health_summary["need_short_roll"]
        if need_short > 0:
            st.error(f"🔄 **{need_short}** position(s) need SHORT roll/expire")
        else:
            st.success("✅ All shorts OK")
    
    with action_col2:
        need_long = health_summary["need_long_roll"]
        if need_long > 0:
            st.warning(f"📈 **{need_long}** position(s) need LONG roll (DTE < 60)")
        else:
            st.success("✅ All longs OK")
    
    with action_col3:
        need_new = health_summary["need_new_short"]
        if need_new > 0:
            st.info(f"📭 **{need_new}** position(s) have no short leg")
        else:
            st.success("✅ All have shorts")
    
    # Filter selector
    st.markdown("---")
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        health_filter = st.selectbox(
            "Filter by Health",
            ["All", "🔴 Critical", "🟡 Attention", "🟢 Healthy", "📭 Need New Short"],
            key="health_filter"
        )
    
    # Get filtered positions
    if health_filter == "🔴 Critical":
        filtered_positions = trade_log.get_diagonals_by_health("critical")
    elif health_filter == "🟡 Attention":
        filtered_positions = trade_log.get_diagonals_by_health("attention")
    elif health_filter == "🟢 Healthy":
        filtered_positions = trade_log.get_diagonals_by_health("healthy")
    elif health_filter == "📭 Need New Short":
        filtered_positions = trade_log.get_diagonals_without_short()
    else:
        filtered_positions = open_diagonals
    
    # Critical alerts with email button
    needing_roll = trade_log.get_diagonals_needing_roll(dte_threshold=3)
    needing_long_roll = trade_log.get_diagonals_needing_long_roll(dte_threshold=60)
    
    if needing_roll or needing_long_roll:
        st.markdown("### ⚠️ Alerts")
        
        # Short roll alerts
        if needing_roll:
            for pos in needing_roll:
                short = pos.current_short_leg
                if short:
                    st.error(f"🔴 {pos.variant_name}: Short ${short.strike} expires in {short.days_to_expiry()} days!")
        
        # Long roll alerts
        if needing_long_roll:
            for pos in needing_long_roll:
                st.warning(f"🟡 {pos.variant_name}: LONG expires in {pos.long_dte} days - plan to roll!")
        
        # Send roll alert email button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📧 Send Roll Alert Email", key="send_roll_email"):
                all_needing_attention = list(set(needing_roll + needing_long_roll))
                success, msg = send_roll_notification_email(all_needing_attention)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
    
    # Live P&L Summary Table
    if open_diagonals:
        st.markdown("### 📊 Live P&L Summary")
        if st.button("🔄 Refresh Prices", key="refresh_live_pnl",
                     help="Fetch live option prices from yfinance"):
            update_diagonal_live_prices(get_trade_log(), symbol="UVXY")
            st.rerun()
        summaries = get_position_live_summary(trade_log, symbol="UVXY")
        if summaries:
            import pandas as pd
            display_data = [{k: v for k, v in s.items()
                             if not k.startswith('_')} for s in summaries]
            df = pd.DataFrame(display_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Total P&L": st.column_config.TextColumn(
                        "Total P&L", help="Combined long + short P&L"),
                    "Return %": st.column_config.TextColumn(
                        "Return %", help="Return on entry cost"),
                    "Need Roll": st.column_config.TextColumn(
                        "Roll?", help="⚠️ = needs roll soon"),
                }
            )
    
    st.markdown("---")
    
    # Add new diagonal position
    with st.expander("➕ Open New Diagonal Position", expanded=False):
        _render_diagonal_entry_form(trade_log)
    
    # Display existing positions
    if not diagonals:
        st.info("No diagonal positions. Use the form above to create one.")
        return
    
    # Use filtered positions if filter is set, otherwise show all
    display_positions = filtered_positions if 'filtered_positions' in dir() else diagonals
    
    # Sort by variant name (V1, V2, V3, V4, V5) then by entry date
    for pos in sorted(display_positions, key=lambda p: (p.variant_id.upper(), p.entry_date)):
        # Get health status
        health = pos.get_health_status() if pos.status == "open" else None
        
        # Status icon based on health
        if pos.status != "open":
            status_icon = "⬛"  # Closed
        elif health and health["status"] == "critical":
            status_icon = "🔴"
        elif health and health["status"] == "attention":
            status_icon = "🟡"
        else:
            status_icon = "🟢"
        
        short = pos.current_short_leg
        pnl_color = "green" if pos.total_pnl >= 0 else "red"
        
        # Build header with health info
        header = f"{status_icon} {pos.variant_name} | {pos.entry_date} | "
        header += f"L${pos.long_strike}"
        if pos.status == "open":
            header += f" ({pos.long_dte}d)"
        header += f" / S${short.strike if short else 'N/A'}"
        if short and pos.status == "open":
            header += f" ({pos.short_dte}d)"
        header += f" | Rolls: {pos.total_rolls}"
        
        with st.expander(header, expanded=pos.status == "open"):
            # Health alerts for open positions
            if health and health["alerts"]:
                for alert in health["alerts"]:
                    if "⚠️" in alert or "ROLL IMMEDIATELY" in alert:
                        st.error(alert)
                    elif "🔴" in alert:
                        st.error(alert)
                    elif "🟡" in alert:
                        st.warning(alert)
                    elif "💰" in alert:
                        st.success(alert)
                    elif "📭" in alert:
                        st.info(alert)
                    elif "🎉" in alert:
                        st.success(alert)
                    else:
                        st.info(alert)
                st.markdown("---")
            
            # Position details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Long Leg (LEAP)**")
                st.write(f"Strike: ${pos.long_strike}")
                st.write(f"Expiration: {pos.long_expiration}")
                st.write(f"DTE: {pos.days_to_long_expiry()} days")
                st.write(f"Entry: ${pos.long_entry_price:.2f}")
                st.write(f"Current: ${pos.long_current_price:.2f}")
                long_pnl = pos.long_pnl
                st.markdown(f"P&L: <span style='color:{'green' if long_pnl >= 0 else 'red'}'>${long_pnl:+,.0f}</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📉 Short Leg (Weekly)**")
                if short:
                    st.write(f"Strike: ${short.strike}")
                    st.write(f"Expiration: {short.expiration_date}")
                    st.write(f"DTE: {short.days_to_expiry()} days")
                    st.write(f"Credit: ${short.entry_credit:.2f}")
                    st.write(f"Current: ${short.current_price:.2f}")
                    st.markdown(f"P&L: <span style='color:{'green' if short.pnl >= 0 else 'red'}'>${short.pnl:+,.0f}</span>", unsafe_allow_html=True)
                else:
                    st.write("No active short leg")
            
            with col3:
                st.markdown("**📊 Position Summary**")
                st.write(f"Contracts: {pos.contracts}")
                st.write(f"Total Rolls: {pos.total_rolls}")
                st.write(f"Total Credits: ${pos.total_credits_received:.2f}")
                st.write(f"Commissions: ${pos.total_commissions:.2f}")
                st.markdown(f"**Total P&L:** <span style='color:{pnl_color}'>${pos.total_pnl:+,.0f}</span> (net of fees)", unsafe_allow_html=True)
            
            # Roll history
            if pos.roll_history:
                st.markdown("---")
                rh_col1, rh_col2 = st.columns([4, 1])
                with rh_col1:
                    st.markdown("**🔄 Roll History**")
                with rh_col2:
                    edit_rolls_clicked = st.button("✏️ Edit Rolls", key=f"edit_rolls_{pos.position_id}")
                    if edit_rolls_clicked:
                        st.session_state[f"editing_rolls_{pos.position_id}"] = True
                roll_data = []
                for roll in pos.roll_history:
                    roll_data.append({
                        "Date": roll.roll_date,
                        "Old Strike": f"${roll.old_strike}",
                        "New Strike": f"${roll.new_strike}",
                        "Buy Back": f"${roll.old_exit_price:.2f}",
                        "New Credit": f"${roll.new_credit:.2f}",
                        "Net Credit": f"${roll.roll_credit:.2f}",
                        "Underlying": f"${roll.underlying_price:.2f}",
                    })
                st.dataframe(roll_data, use_container_width=True, hide_index=True)
                
                if st.session_state.get(f"editing_rolls_{pos.position_id}"):
                    _render_roll_history_edit_form(trade_log, pos)
            
            # Action buttons
            st.markdown("---")
            
            if pos.status == "open":
                # Check if position has active short
                has_active_short = short and short.status == "open"
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    if has_active_short:
                        roll_clicked = st.button("🔄 Roll Short", key=f"roll_{pos.position_id}")
                    else:
                        sell_short_clicked = st.button("📈 Sell Short", key=f"sell_short_{pos.position_id}",
                                                       help="Sell a new short leg")
                        roll_clicked = False
                
                with col2:
                    if has_active_short:
                        expire_clicked = st.button("🎉 Expire Profit", key=f"expire_{pos.position_id}", 
                                                   help="Mark short as expired worthless (keep LEAP)")
                    else:
                        expire_clicked = False
                        st.write("")  # Empty space
                
                with col3:
                    update_clicked = st.button("💰 Prices", key=f"update_{pos.position_id}")
                
                with col4:
                    close_clicked = st.button("🚪 Close All", key=f"close_{pos.position_id}",
                                              help="Close entire diagonal position")
                
                with col5:
                    edit_clicked = st.button("✏️ Edit", key=f"edit_{pos.position_id}")
                
                with col6:
                    delete_clicked = st.button("🗑️ Del", key=f"delete_{pos.position_id}")
                
                # Second row of buttons for leg management
                col7, col8, col9, col10 = st.columns([1, 1, 1, 1])
                with col7:
                    roll_long_clicked = st.button("🔄 Roll Long", key=f"roll_long_{pos.position_id}",
                                                   help="Roll the LEAP to new strike/expiration")
                with col8:
                    if has_active_short:
                        close_short_clicked = st.button("📕 Close Short", key=f"close_short_{pos.position_id}",
                                                        help="Close just the short leg")
                    else:
                        close_short_clicked = False
                with col9:
                    close_long_clicked = st.button("📘 Close Long", key=f"close_long_{pos.position_id}",
                                                   help="Close long leg (closes entire position)")
                with col10:
                    long_dte = pos.days_to_long_expiry()
                    if long_dte <= 60:
                        st.warning(f"⚠️ Long DTE: {long_dte}d")
                    else:
                        st.caption(f"Long DTE: {long_dte}d")
                
                # Handle button clicks
                if roll_clicked:
                    st.session_state[f"rolling_{pos.position_id}"] = True
                if not has_active_short and 'sell_short_clicked' in dir() and sell_short_clicked:
                    st.session_state[f"selling_short_{pos.position_id}"] = True
                if expire_clicked:
                    st.session_state[f"expiring_{pos.position_id}"] = True
                if update_clicked:
                    st.session_state[f"updating_{pos.position_id}"] = True
                if close_clicked:
                    st.session_state[f"closing_{pos.position_id}"] = True
                if edit_clicked:
                    st.session_state[f"editing_{pos.position_id}"] = True
                    st.rerun()
                    st.rerun()
                if False:  # placeholder
                    st.session_state[f"editing_{pos.position_id}"] = True
                if delete_clicked:
                    st.session_state[f"deleting_{pos.position_id}"] = True
                if roll_long_clicked:
                    st.session_state[f"rolling_long_{pos.position_id}"] = True
                if close_short_clicked:
                    st.session_state[f"closing_short_{pos.position_id}"] = True
                    st.rerun()
                if close_long_clicked:
                    st.session_state[f"closing_long_{pos.position_id}"] = True
                    st.rerun()
                
                # Render forms based on state
                if st.session_state.get(f"rolling_{pos.position_id}"):
                    _render_roll_form(trade_log, pos)
                if st.session_state.get(f"selling_short_{pos.position_id}"):
                    _render_sell_short_form(trade_log, pos)
                if st.session_state.get(f"expiring_{pos.position_id}"):
                    _render_expire_confirm(trade_log, pos)
                if st.session_state.get(f"updating_{pos.position_id}"):
                    _render_price_update_form(trade_log, pos)
                if st.session_state.get(f"closing_{pos.position_id}"):
                    _render_close_form(trade_log, pos)
                if st.session_state.get(f"editing_{pos.position_id}"):
                    _render_edit_form(trade_log, pos)
                if st.session_state.get(f"deleting_{pos.position_id}"):
                    _render_delete_confirm(trade_log, pos)
                if st.session_state.get(f"rolling_long_{pos.position_id}"):
                    _render_roll_long_form(trade_log, pos)
                if st.session_state.get(f"closing_short_{pos.position_id}"):
                    _render_close_short_form(trade_log, pos)
                if st.session_state.get(f"closing_long_{pos.position_id}"):
                    _render_close_long_form(trade_log, pos)
            
            else:
                # Closed positions can still be edited or deleted
                col1, col2 = st.columns(2)
                with col1:
                    edit_clicked = st.button("✏️ Edit", key=f"edit_closed_{pos.position_id}")
                with col2:
                    delete_clicked = st.button("🗑️ Delete", key=f"delete_closed_{pos.position_id}")
                
                if edit_clicked:
                    st.session_state[f"editing_{pos.position_id}"] = True
                    st.rerun()
                    st.rerun()
                if False:  # placeholder
                    st.session_state[f"editing_{pos.position_id}"] = True
                if delete_clicked:
                    st.session_state[f"deleting_{pos.position_id}"] = True
                
                if st.session_state.get(f"editing_{pos.position_id}"):
                    _render_edit_form(trade_log, pos)
                if st.session_state.get(f"deleting_{pos.position_id}"):
                    _render_delete_confirm(trade_log, pos)
                if st.session_state.get(f"rolling_long_{pos.position_id}"):
                    _render_roll_long_form(trade_log, pos)
                



def _render_diagonal_entry_form(trade_log):
    """Form to create a new diagonal position."""
    from trade_log import DiagonalPosition
    
    col1, col2 = st.columns(2)
    with col1:
        variant = st.selectbox(
            "Variant",
            options=[role.value for role in VariantRole],
            key="diag_entry_variant"
        )
        contracts = st.number_input("Contracts", min_value=1, max_value=100, value=5, key="diag_entry_contracts")
    
    with col2:
        entry_regime = st.selectbox("Entry Regime", ["CALM", "ELEVATED", "HIGH", "EXTREME"], key="diag_entry_regime")
        entry_vix = st.number_input("VIX Level", min_value=10.0, max_value=80.0, value=20.0, key="diag_entry_vix")
    
    st.markdown("##### Long Leg")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        long_strike = st.number_input("Long Strike", min_value=1.0, value=40.0, step=0.5, key="diag_long_strike")
    with lcol2:
        long_exp = st.date_input("Long Expiration", key="diag_long_exp")
    with lcol3:
        long_price = st.number_input("Long Debit ($)", min_value=0.01, value=4.00, step=0.05, key="diag_long_price")
    
    st.markdown("##### Short Leg")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        short_strike = st.number_input("Short Strike", min_value=1.0, value=38.0, step=0.5, key="diag_short_strike")
    with scol2:
        short_exp = st.date_input("Short Expiration", key="diag_short_exp")
    with scol3:
        short_credit = st.number_input("Short Credit ($)", min_value=0.01, value=0.80, step=0.05, key="diag_short_credit")
    
    # Commission settings
    fee_per_contract = st.number_input(
        "Commission per contract ($)",
        min_value=0.0, max_value=5.0, value=0.65, step=0.05,
        key="diag_fee_per_contract",
        help="Broker fee per contract (e.g., $0.65 for most brokers)"
    )
    
    net = short_credit - long_price
    total_commission = fee_per_contract * contracts * 2  # Long buy + Short sell
    st.info(f"Net {'Credit' if net > 0 else 'Debit'}: ${abs(net):.2f} per spread | Total: ${abs(net) * contracts * 100:.2f} | Est. Commission: ${total_commission:.2f}")
    
    if st.button("✅ Open Diagonal Position", key="diag_entry_submit"):
        try:
            variant_names = {r.value: r.value.replace("_", " ").title() for r in VariantRole}
            pos = trade_log.open_diagonal(
                variant_id=variant.upper(),
                variant_name=variant_names.get(variant, variant),
                contracts=contracts,
                long_strike=long_strike,
                long_expiration=long_exp.isoformat(),
                long_price=long_price,
                short_strike=short_strike,
                short_expiration=short_exp.isoformat(),
                short_credit=short_credit,
                entry_regime=entry_regime,
                entry_vix_level=entry_vix,
                fee_per_contract=fee_per_contract,
            )
            st.success(f"✅ Opened diagonal position: {pos.position_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def _render_roll_form(trade_log, pos):
    """Form to roll a short leg with smart suggestions."""
    st.markdown("##### 🔄 Roll Short Leg")
    
    short = pos.current_short_leg
    
    # Get current underlying price for suggestions
    try:
        import yfinance as yf
        ticker = yf.Ticker("UVXY")
        current_price = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice', 38.0)
    except:
        current_price = 38.0
    
    # Roll suggestions based on position type
    st.markdown("##### 💡 Roll Suggestions")
    suggested_strikes = [
        round(current_price * 1.02, 0),  # 2% OTM
        round(current_price * 1.05, 0),  # 5% OTM  
        round(current_price * 1.10, 0),  # 10% OTM
    ]
    
    # Suggest next Friday expiration
    from datetime import datetime, timedelta
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7  # Next week if today is Friday
    suggested_exp = today + timedelta(days=days_until_friday)
    
    st.info(f"""
    **Current UVXY:** ${current_price:.2f}
    
    **Suggested Strikes (OTM):**
    - Conservative (2% OTM): ${suggested_strikes[0]:.0f}
    - Moderate (5% OTM): ${suggested_strikes[1]:.0f}  
    - Aggressive (10% OTM): ${suggested_strikes[2]:.0f}
    
    **Suggested Expiration:** {suggested_exp.strftime('%Y-%m-%d')} (next Friday)
    """)
    
    st.markdown("---")
    
    # Partial roll support
    max_contracts = short.contracts if short else pos.contracts
    
    col0, col1, col2 = st.columns([1, 1, 1])
    with col0:
        st.write(f"Current Short: ${short.strike} exp {short.expiration_date}")
        st.write(f"Contracts: {max_contracts}")
        contracts_to_roll = st.number_input(
            "Contracts to Roll",
            min_value=1, max_value=max_contracts, value=max_contracts,
            key=f"roll_contracts_{pos.position_id}",
            help=f"Partial roll: 1-{max_contracts}"
        )
        exit_price = st.number_input(
            "Buy Back Price ($)",
            min_value=0.0, max_value=20.0, value=0.05, step=0.01,
            key=f"roll_exit_{pos.position_id}",
            help="If expired worthless, enter 0"
        )
    
    with col1:
        new_strike = st.number_input(
            "New Strike",
            min_value=1.0, value=float(suggested_strikes[1]), step=0.5,
            key=f"roll_new_strike_{pos.position_id}"
        )
        new_exp = st.date_input("New Expiration", value=suggested_exp.date(), key=f"roll_new_exp_{pos.position_id}")
        new_credit = st.number_input(
            "New Credit ($)",
            min_value=0.01, value=0.20, step=0.05,
            key=f"roll_new_credit_{pos.position_id}"
        )
    
    with col2:
        underlying = st.number_input("Current Underlying Price", min_value=1.0, value=current_price, key=f"roll_underlying_{pos.position_id}")
        
        if contracts_to_roll < max_contracts:
            st.info(f"⚠️ Partial roll: {contracts_to_roll} of {max_contracts} contracts")
    
    net_roll = new_credit - exit_price
    st.info(f"Net Roll {'Credit' if net_roll > 0 else 'Debit'}: ${abs(net_roll):.2f} per contract")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Execute Roll", key=f"roll_submit_{pos.position_id}"):
            try:
                new_leg, roll = trade_log.roll_diagonal_short(
                    position_id=pos.position_id,
                    exit_price=exit_price,
                    new_strike=new_strike,
                    new_expiration=new_exp.isoformat(),
                    new_credit=new_credit,
                    underlying_price=underlying,
                    regime="CALM",
                    contracts=contracts_to_roll,
                )
                if contracts_to_roll < max_contracts:
                    st.success(f"✅ Partial roll ({contracts_to_roll} contracts): Net credit ${roll.roll_credit:.2f}")
                else:
                    st.success(f"✅ Rolled: Net credit ${roll.roll_credit:.2f}")
                st.session_state[f"rolling_{pos.position_id}"] = False
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"Roll failed: {e}")
                st.code(traceback.format_exc())
    
    with col2:
        if st.button("❌ Cancel", key=f"roll_cancel_{pos.position_id}"):
            st.session_state[f"rolling_{pos.position_id}"] = False
            st.rerun()


def _render_price_update_form(trade_log, pos):
    """Form to update current prices."""
    st.markdown("##### 💰 Update Current Prices")
    
    col1, col2 = st.columns(2)
    with col1:
        long_price = st.number_input(
            "Long Current Price",
            min_value=0.0, value=pos.long_current_price or pos.long_entry_price, step=0.05,
            key=f"upd_long_{pos.position_id}"
        )
    with col2:
        short = pos.current_short_leg
        short_price = st.number_input(
            "Short Current Price",
            min_value=0.0, value=short.current_price if short else 0.0, step=0.05,
            key=f"upd_short_{pos.position_id}"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Update", key=f"upd_submit_{pos.position_id}"):
            trade_log.update_diagonal_prices(pos.position_id, long_price, short_price)
            st.success("✅ Prices updated")
            st.session_state[f"updating_{pos.position_id}"] = False
            st.rerun()
    with col2:
        if st.button("❌ Cancel", key=f"upd_cancel_{pos.position_id}"):
            st.session_state[f"updating_{pos.position_id}"] = False
            st.rerun()


def _render_close_form(trade_log, pos):
    """Form to close a diagonal position."""
    st.markdown("##### 🚪 Close Position")
    
    col1, col2 = st.columns(2)
    with col1:
        long_exit = st.number_input(
            "Long Exit Price",
            min_value=0.0, value=pos.long_current_price or 0.0, step=0.05,
            key=f"close_long_{pos.position_id}"
        )
    with col2:
        short = pos.current_short_leg
        short_exit = st.number_input(
            "Short Exit Price",
            min_value=0.0, value=short.current_price if short else 0.0, step=0.05,
            key=f"close_short_{pos.position_id}"
        )
    
    reason = st.selectbox(
        "Exit Reason",
        ["target_hit", "stop_hit", "manual", "expired"],
        key=f"close_reason_{pos.position_id}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Close Position", key=f"close_submit_{pos.position_id}"):
            trade_log.close_diagonal(pos.position_id, long_exit, short_exit, reason)
            st.success("✅ Position closed")
            st.session_state[f"closing_{pos.position_id}"] = False
            st.rerun()
    with col2:
        if st.button("❌ Cancel", key=f"close_cancel_{pos.position_id}"):
            st.session_state[f"closing_{pos.position_id}"] = False
            st.rerun()




def _render_edit_form(trade_log, pos):
    """Form to edit a diagonal position."""
    st.markdown("##### ✏️ Edit Position")
    
    col1, col2 = st.columns(2)
    with col1:
        new_variant = st.text_input(
            "Variant Name",
            value=pos.variant_name,
            key=f"edit_variant_{pos.position_id}"
        )
        new_contracts = st.number_input(
            "Contracts",
            min_value=1, max_value=1000,
            value=pos.contracts,
            key=f"edit_contracts_{pos.position_id}"
        )
        new_regime = st.selectbox(
            "Entry Regime",
            ["CALM", "ELEVATED", "HIGH", "EXTREME", "DECLINING", "RISING", "STRESSED"],
            index=0,
            key=f"edit_regime_{pos.position_id}"
        )
    
    with col2:
        new_notes = st.text_area(
            "Notes",
            value=pos.notes or "",
            key=f"edit_notes_{pos.position_id}"
        )
        new_commissions = st.number_input(
            "Total Commissions ($)",
            min_value=0.0, 
            value=float(pos.total_commissions) if pos.total_commissions else 0.0,
            step=0.65,
            key=f"edit_commissions_{pos.position_id}",
            help="Cumulative commissions paid (entry + rolls)"
        )
        new_fee = st.number_input(
            "Fee per Contract ($)",
            min_value=0.0, max_value=5.0,
            value=float(pos.fee_per_contract) if pos.fee_per_contract else 0.65,
            step=0.05,
            key=f"edit_fee_{pos.position_id}",
            help="Broker fee per contract for future rolls"
        )
    
    st.markdown("**Long Leg**")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        new_long_strike = st.number_input(
            "Long Strike",
            min_value=1.0, value=float(pos.long_strike), step=0.5,
            key=f"edit_long_strike_{pos.position_id}"
        )
    with lcol2:
        new_long_exp = st.text_input(
            "Long Expiration (YYYY-MM-DD)",
            value=pos.long_expiration,
            key=f"edit_long_exp_{pos.position_id}"
        )
    with lcol3:
        new_long_price = st.number_input(
            "Long Entry Price",
            min_value=0.01, value=float(pos.long_entry_price), step=0.05,
            key=f"edit_long_price_{pos.position_id}"
        )
    
    st.markdown("**Short Leg**")
    short = pos.current_short_leg
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        new_short_strike = st.number_input(
            "Short Strike",
            min_value=1.0, value=float(short.strike) if short else 38.0, step=0.5,
            key=f"edit_short_strike_{pos.position_id}"
        )
    with scol2:
        new_short_exp = st.text_input(
            "Short Expiration (YYYY-MM-DD)",
            value=short.expiration_date if short else "",
            key=f"edit_short_exp_{pos.position_id}"
        )
    with scol3:
        new_short_credit = st.number_input(
            "Short Credit",
            min_value=0.01, value=float(short.entry_credit) if short else 0.50, step=0.05,
            key=f"edit_short_credit_{pos.position_id}"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Changes", key=f"edit_save_{pos.position_id}"):
            try:
                # Update main position
                trade_log.update_diagonal(
                    pos.position_id,
                    variant_name=new_variant,
                    contracts=new_contracts,
                    long_strike=new_long_strike,
                    long_expiration=new_long_exp,
                    long_entry_price=new_long_price,
                    entry_regime=new_regime,
                    notes=new_notes,
                    total_commissions=new_commissions,
                    fee_per_contract=new_fee,
                )
                
                # Update short leg if exists
                if short:
                    trade_log.update_diagonal_short_leg(
                        pos.position_id,
                        strike=new_short_strike,
                        expiration_date=new_short_exp,
                        entry_credit=new_short_credit,
                    )
                
                st.success("✅ Position updated!")
                st.session_state[f"editing_{pos.position_id}"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("❌ Cancel", key=f"edit_cancel_{pos.position_id}"):
            st.session_state[f"editing_{pos.position_id}"] = False
            st.rerun()




def _render_roll_history_edit_form(trade_log, pos):
    """Spreadsheet-style editor for ALL roll history records at once."""
    st.markdown("##### ✏️ Edit Roll History (Spreadsheet Mode)")
    
    import pandas as pd
    
    # Show existing roll history in spreadsheet if any
    if pos.roll_history:
        # Convert roll history to DataFrame
        roll_data = []
        for roll in pos.roll_history:
            roll_data.append({
                "roll_id": roll.roll_id,
                "roll_date": roll.roll_date,
                "roll_type": roll.roll_type,
                "contracts": getattr(roll, "contracts", pos.contracts),
                "old_strike": roll.old_strike,
                "old_expiration": roll.old_expiration,
                "old_exit_price": roll.old_exit_price,
                "new_strike": roll.new_strike,
                "new_expiration": roll.new_expiration,
                "new_credit": roll.new_credit,
                "roll_credit": roll.roll_credit,
                "underlying_price": roll.underlying_price,
                "regime": roll.regime or "",
                "notes": roll.notes or "",
            })
        
        df = pd.DataFrame(roll_data)
        
        # Show legend for column meanings
        with st.expander("📖 Column Guide (Short vs Long Rolls)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Short Roll** (weekly call):
                - Old Exit = Buyback price (0 if expired)
                - New Credit = Premium received
                - Net = Credit received
                """)
            with col2:
                st.markdown("""
                **Long Roll** (LEAP):
                - Old Exit = Sale price of old LEAP
                - New Credit = Negative of new LEAP cost
                - Net = Usually negative (debit)
                """)
        
        # Configure column display
        column_config = {
            "roll_id": st.column_config.TextColumn(
                "Roll ID",
                disabled=True,  # Read-only
                width="small",
            ),
            "roll_date": st.column_config.TextColumn(
                "Date",
                width="small",
            ),
            "roll_type": st.column_config.SelectboxColumn(
                "Type",
                options=["short", "long"],
                width="small",
            ),
            "contracts": st.column_config.NumberColumn(
                "Qty",
                min_value=1,
                max_value=100,
                step=1,
                width="small",
            ),
            "old_strike": st.column_config.NumberColumn(
                "Old K",
                min_value=1.0,
                format="$%.1f",
                width="small",
                help="Old strike price",
            ),
            "old_expiration": st.column_config.TextColumn(
                "Old Exp",
                width="small",
            ),
            "old_exit_price": st.column_config.NumberColumn(
                "Old Exit",
                min_value=0.0,
                format="$%.2f",
                width="small",
                help="Short: buyback price (0=expired). Long: sale price of old LEAP",
            ),
            "new_strike": st.column_config.NumberColumn(
                "New K",
                min_value=1.0,
                format="$%.1f",
                width="small",
                help="New strike price",
            ),
            "new_expiration": st.column_config.TextColumn(
                "New Exp",
                width="small",
            ),
            "new_credit": st.column_config.NumberColumn(
                "New Entry",
                format="$%.2f",
                width="small",
                help="Short: credit received. Long: negative of LEAP cost",
            ),
            "roll_credit": st.column_config.NumberColumn(
                "Net",
                format="$%.2f",
                disabled=True,  # Computed field
                width="small",
                help="Net credit (+) or debit (-). Auto-computed.",
            ),
            "underlying_price": st.column_config.NumberColumn(
                "UVXY",
                min_value=0.0,
                format="$%.2f",
                width="small",
            ),
            "regime": st.column_config.TextColumn(
                "Regime",
                width="small",
            ),
            "notes": st.column_config.TextColumn(
                "Notes",
                width="medium",
            ),
        }
        
        # Editable dataframe
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",  # Don't allow adding/deleting rows here
            key=f"roll_editor_{pos.position_id}",
        )
        
        # Show computed totals
        if len(edited_df) > 0:
            # Recompute roll_credit for display
            edited_df["roll_credit_computed"] = edited_df["new_credit"] - edited_df["old_exit_price"]
            total_roll_credit = (edited_df["roll_credit_computed"] * edited_df["contracts"]).sum()
            
            # Count short vs long rolls
            short_rolls = len(edited_df[edited_df["roll_type"] == "short"])
            long_rolls = len(edited_df[edited_df["roll_type"] == "long"])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rolls in History", len(edited_df), help=f"Short: {short_rolls}, Long: {long_rolls}")
            with col2:
                # Show discrepancy if any
                if pos.total_rolls != short_rolls:
                    st.metric("total_rolls Field", pos.total_rolls, delta=f"Should be {short_rolls}", delta_color="off")
                else:
                    st.metric("total_rolls Field", pos.total_rolls)
            with col3:
                st.metric("Total Roll Credits", f"${total_roll_credit:.2f}")
            with col4:
                avg_credit = edited_df["roll_credit_computed"].mean()
                st.metric("Avg Credit/Roll", f"${avg_credit:.2f}")
            
            # Warning if mismatch
            if pos.total_rolls != short_rolls:
                st.warning(f"⚠️ `total_rolls` ({pos.total_rolls}) doesn't match short roll count ({short_rolls}). Click 'Recalculate Totals' to fix.")
        
        # Action buttons for spreadsheet
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save All Changes", key=f"save_rolls_{pos.position_id}", type="primary"):
                try:
                    # Update each roll record
                    changes_made = 0
                    for idx, row in edited_df.iterrows():
                        roll_id = row["roll_id"]
                        
                        # Recompute roll_credit
                        computed_credit = row["new_credit"] - row["old_exit_price"]
                        
                        trade_log.update_roll_record(
                            pos.position_id,
                            roll_id,
                            roll_date=row["roll_date"],
                            roll_type=row["roll_type"],
                            old_strike=row["old_strike"],
                            old_expiration=row["old_expiration"],
                            old_exit_price=row["old_exit_price"],
                            new_strike=row["new_strike"],
                            new_expiration=row["new_expiration"],
                            new_credit=row["new_credit"],
                            underlying_price=row["underlying_price"],
                            regime=row["regime"],
                            notes=row["notes"],
                            contracts=int(row["contracts"]),
                        )
                        changes_made += 1
                    
                    st.success(f"✅ Saved {changes_made} roll records!")
                    st.session_state[f"editing_rolls_{pos.position_id}"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving: {e}")
        
        with col2:
            if st.button("❌ Cancel", key=f"cancel_rolls_{pos.position_id}"):
                st.session_state[f"editing_rolls_{pos.position_id}"] = False
                st.rerun()
        
        with col3:
            if st.button("🔄 Recalculate Totals", key=f"recalc_rolls_{pos.position_id}"):
                try:
                    pos.recalc_roll_totals()  # Fixed method name
                    trade_log._save()
                    st.success(f"✅ Totals recalculated! Rolls: {pos.total_rolls}, Credits: ${pos.total_roll_credits:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Delete individual roll section
        st.markdown("---")
        st.markdown("**Delete a Roll Record**")
        
        roll_options = [f"{r.roll_id} ({r.roll_date})" for r in pos.roll_history]
        selected_for_delete = st.selectbox(
            "Select roll to delete",
            options=roll_options,
            key=f"delete_roll_select_{pos.position_id}",
        )
        
        delete_col1, delete_col2 = st.columns([1, 3])
        with delete_col1:
            delete_confirm = st.checkbox("Confirm", key=f"delete_confirm_{pos.position_id}")
        with delete_col2:
            if st.button(
                "🗑️ Delete Selected Roll",
                key=f"delete_roll_btn_{pos.position_id}",
                disabled=not delete_confirm,
            ):
                roll_id_to_delete = selected_for_delete.split(" ")[0]
                if trade_log.delete_roll_record(pos.position_id, roll_id_to_delete):
                    st.success(f"✅ Deleted {roll_id_to_delete}")
                    st.session_state[f"editing_rolls_{pos.position_id}"] = False
                    st.rerun()
                else:
                    st.error("Failed to delete")
    
    else:
        # No roll history yet
        st.info("📭 No roll history yet. Add the first roll record below.")
    
    # ═══════════════════════════════════════════════════════════════
    # ADD NEW ROLL RECORD - Insert at any position (sorted by date)
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**➕ Add Roll Record**")
    st.caption("Add a roll at any date - records will be re-sorted and renumbered automatically")
    
    # Select roll type FIRST (outside form so it can change the form displayed)
    add_roll_type = st.radio(
        "Roll Type",
        options=["Short Roll", "Long Roll"],
        horizontal=True,
        key=f"add_roll_type_radio_{pos.position_id}",
        help="Short = roll weekly call, Long = roll LEAP"
    )
    
    if add_roll_type == "Short Roll":
        # ─────────────────────────────────────────────────────────
        # SHORT ROLL FORM
        # ─────────────────────────────────────────────────────────
        with st.form(key=f"add_short_roll_form_{pos.position_id}"):
            st.markdown("**Short Roll** (sell new weekly, buy back old)")
            
            add_col1, add_col2, add_col3 = st.columns(3)
            
            with add_col1:
                add_date = st.date_input(
                    "Roll Date",
                    value=dt.date.today(),
                )
                add_contracts = st.number_input(
                    "Contracts",
                    min_value=1,
                    max_value=100,
                    value=pos.contracts,
                )
            
            with add_col2:
                st.markdown("**Old Short (Closed)**")
                add_old_strike = st.number_input(
                    "Old Strike",
                    min_value=1.0,
                    value=float(pos.current_short_leg.strike if pos.current_short_leg else 40.0),
                    step=0.5,
                )
                add_old_exp = st.date_input("Old Expiration")
                add_old_exit = st.number_input(
                    "Buyback Price",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    help="Price paid to close (0 if expired worthless)",
                )
            
            with add_col3:
                st.markdown("**New Short (Opened)**")
                add_new_strike = st.number_input(
                    "New Strike",
                    min_value=1.0,
                    value=38.0,
                    step=0.5,
                )
                add_new_exp = st.date_input("New Expiration")
                add_new_credit = st.number_input(
                    "Credit Received",
                    min_value=0.0,
                    value=0.30,
                    step=0.01,
                )
            
            # Additional fields
            add_col4, add_col5 = st.columns(2)
            with add_col4:
                add_underlying = st.number_input(
                    "UVXY Price",
                    min_value=0.0,
                    value=15.0,
                    step=0.5,
                )
                add_regime = st.selectbox(
                    "Regime",
                    options=["", "CALM", "ELEVATED", "HIGH", "EXTREME"],
                )
            with add_col5:
                add_notes = st.text_input("Notes", value="")
                # Show computed net credit
                computed_net = add_new_credit - add_old_exit
                if computed_net >= 0:
                    st.success(f"Net Credit: ${computed_net:.2f}")
                else:
                    st.warning(f"Net Debit: ${abs(computed_net):.2f}")
            
            add_submitted = st.form_submit_button("➕ Add Short Roll", type="primary")
        
        if add_submitted:
            try:
                from trade_log import RollRecord
                
                new_roll = RollRecord(
                    roll_id=f"{pos.position_id}-TEMP",
                    position_id=pos.position_id,
                    roll_date=add_date.isoformat(),
                    roll_type="short",
                    old_strike=add_old_strike,
                    old_expiration=add_old_exp.isoformat(),
                    old_exit_price=add_old_exit,
                    new_strike=add_new_strike,
                    new_expiration=add_new_exp.isoformat(),
                    new_credit=add_new_credit,
                    roll_credit=add_new_credit - add_old_exit,
                    underlying_price=add_underlying,
                    contracts=add_contracts,
                    regime=add_regime,
                    notes=add_notes,
                )
                
                # Add, sort, renumber
                pos.roll_history.append(new_roll)
                pos.roll_history.sort(key=lambda r: r.roll_date)
                
                short_num = 0
                long_num = 0
                for roll in pos.roll_history:
                    if roll.roll_type == "long":
                        long_num += 1
                        roll.roll_id = f"{pos.position_id}-RL{long_num}"
                    else:
                        short_num += 1
                        roll.roll_id = f"{pos.position_id}-R{short_num}"
                
                pos.recalc_roll_totals()
                trade_log._save()
                
                st.success(f"✅ Added short roll! Total rolls: {len(pos.roll_history)}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        # ─────────────────────────────────────────────────────────
        # LONG ROLL FORM
        # ─────────────────────────────────────────────────────────
        with st.form(key=f"add_long_roll_form_{pos.position_id}"):
            st.markdown("**Long Roll** (sell old LEAP, buy new LEAP)")
            
            add_col1, add_col2, add_col3 = st.columns(3)
            
            with add_col1:
                add_date = st.date_input(
                    "Roll Date",
                    value=dt.date.today(),
                )
                add_contracts = st.number_input(
                    "Contracts",
                    min_value=1,
                    max_value=100,
                    value=pos.contracts,
                )
            
            with add_col2:
                st.markdown("**Old Long (Sold)**")
                add_old_strike = st.number_input(
                    "Old Strike",
                    min_value=1.0,
                    value=float(pos.long_strike),
                    step=0.5,
                )
                add_old_exp = st.date_input(
                    "Old Expiration",
                    value=dt.datetime.strptime(pos.long_expiration, "%Y-%m-%d").date() if pos.long_expiration else dt.date.today(),
                )
                add_old_exit = st.number_input(
                    "Sale Price",
                    min_value=0.0,
                    value=float(pos.long_current_price) if pos.long_current_price else 5.0,
                    step=0.05,
                    help="Price received for selling old LEAP",
                )
            
            with add_col3:
                st.markdown("**New Long (Bought)**")
                add_new_strike = st.number_input(
                    "New Strike",
                    min_value=1.0,
                    value=float(pos.long_strike),
                    step=0.5,
                )
                add_new_exp = st.date_input("New Expiration")
                add_new_entry = st.number_input(
                    "Purchase Price",
                    min_value=0.0,
                    value=8.0,
                    step=0.05,
                    help="Price paid for new LEAP",
                )
            
            # Additional fields
            add_col4, add_col5 = st.columns(2)
            with add_col4:
                add_underlying = st.number_input(
                    "UVXY Price",
                    min_value=0.0,
                    value=15.0,
                    step=0.5,
                )
                add_regime = st.selectbox(
                    "Regime",
                    options=["", "CALM", "ELEVATED", "HIGH", "EXTREME"],
                )
            with add_col5:
                add_notes = st.text_input("Notes", value="")
                # Show computed net debit/credit
                roll_debit = add_new_entry - add_old_exit
                if roll_debit > 0:
                    st.warning(f"Net Debit: ${roll_debit:.2f} (paid to roll)")
                else:
                    st.success(f"Net Credit: ${abs(roll_debit):.2f} (received)")
            
            add_submitted = st.form_submit_button("➕ Add Long Roll", type="primary")
        
        if add_submitted:
            try:
                from trade_log import RollRecord
                
                # For long rolls: new_credit stores negative of debit
                roll_debit = add_new_entry - add_old_exit
                
                new_roll = RollRecord(
                    roll_id=f"{pos.position_id}-TEMP",
                    position_id=pos.position_id,
                    roll_date=add_date.isoformat(),
                    roll_type="long",
                    old_strike=add_old_strike,
                    old_expiration=add_old_exp.isoformat(),
                    old_exit_price=add_old_exit,
                    new_strike=add_new_strike,
                    new_expiration=add_new_exp.isoformat(),
                    new_credit=-roll_debit,  # Negative if debit
                    roll_credit=-roll_debit,
                    underlying_price=add_underlying,
                    contracts=add_contracts,
                    regime=add_regime,
                    notes=add_notes,
                )
                
                # Add, sort, renumber
                pos.roll_history.append(new_roll)
                pos.roll_history.sort(key=lambda r: r.roll_date)
                
                short_num = 0
                long_num = 0
                for roll in pos.roll_history:
                    if roll.roll_type == "long":
                        long_num += 1
                        roll.roll_id = f"{pos.position_id}-RL{long_num}"
                    else:
                        short_num += 1
                        roll.roll_id = f"{pos.position_id}-R{short_num}"
                
                pos.recalc_roll_totals()
                trade_log._save()
                
                st.success(f"✅ Added long roll! Total rolls: {len(pos.roll_history)}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Debug section
    st.markdown("---")
    with st.expander("🔧 Debug: View Raw Roll Data"):
        st.markdown(f"**Position:** `{pos.position_id}`")
        st.markdown(f"**total_rolls field:** `{pos.total_rolls}`")
        st.markdown(f"**total_roll_credits field:** `${pos.total_roll_credits:.2f}`")
        st.markdown(f"**roll_history length:** `{len(pos.roll_history)}`")
        st.markdown(f"**short_legs count:** `{len(pos.short_legs)}`")
        
        # Short legs info
        if len(pos.short_legs) > 1:
            st.info(f"💡 You have {len(pos.short_legs)} short legs, which implies {len(pos.short_legs) - 1} rolls occurred. "
                   f"If roll_history shows fewer, some rolls may have been done before roll tracking was added.")
        
        st.markdown("**Short Legs:**")
        for i, leg in enumerate(pos.short_legs):
            st.code(f"""
Short Leg {i+1}: {leg.leg_id}
  status: {leg.status}
  strike: {leg.strike}
  expiration: {leg.expiration_date}
  entry_credit: {leg.entry_credit}
  entry_date: {leg.entry_date}
""")
        
        st.markdown("**Raw roll_history:**")
        if not pos.roll_history:
            st.warning("⚠️ roll_history is empty! Rolls may have been done before tracking was added.")
        for i, roll in enumerate(pos.roll_history):
            roll_type = getattr(roll, "roll_type", "NOT SET")
            st.code(f"""
Roll {i+1}: {roll.roll_id}
  roll_type: {roll_type}
  roll_date: {roll.roll_date}
  old_strike: {roll.old_strike} -> new_strike: {roll.new_strike}
  old_exit_price: {roll.old_exit_price}, new_credit: {roll.new_credit}
  roll_credit: {roll.roll_credit}
  contracts: {getattr(roll, 'contracts', 'NOT SET')}
""")
        
        # Fix all positions button
        if st.button("🔧 Fix ALL Position Roll Counts", key=f"fix_all_rolls_{pos.position_id}"):
            fixed = 0
            for pid, p in trade_log.diagonal_positions.items():
                old_total = p.total_rolls
                p.recalc_roll_totals()
                if p.total_rolls != old_total:
                    fixed += 1
            trade_log._save()
            st.success(f"✅ Recalculated totals for all positions. Fixed {fixed} discrepancies.")
            st.rerun()
        
        # Info about short_legs vs roll_history
        if len(pos.short_legs) > 1 and len(pos.roll_history) < len(pos.short_legs) - 1:
            st.markdown("---")
            st.warning(f"⚠️ **Missing Roll Records**: You have {len(pos.short_legs)} short legs but only {len(pos.roll_history)} roll records. "
                      f"This likely means {len(pos.short_legs) - 1 - len(pos.roll_history)} rolls were done before roll tracking was added.")
            st.info("Use the '➕ Add Roll Record' form above to add missing roll records.")



def _render_roll_long_form(trade_log, pos):
    """Form to roll the long leg (LEAP) to a new strike/expiration."""
    st.markdown("##### 🔄 Roll Long Leg (LEAP)")
    
    # Get current underlying price
    try:
        import yfinance as yf
        ticker = yf.Ticker("UVXY")
        current_price = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice', 38.0)
    except:
        current_price = 38.0
    
    # Current long leg info
    long_dte = pos.days_to_long_expiry()
    
    st.info(f"""
    **Current Long Leg:**
    - Strike: ${pos.long_strike:.0f}
    - Expiration: {pos.long_expiration} ({long_dte} DTE)
    - Entry Price: ${pos.long_entry_price:.2f}
    - Current Price: ${pos.long_current_price:.2f}
    
    **Current UVXY:** ${current_price:.2f}
    """)
    
    # Suggestions for new long leg
    st.markdown("##### 💡 Roll Suggestions")
    
    # Suggest strikes based on current price
    suggested_strikes = [
        round(current_price * 0.85, 0),  # 15% ITM
        round(current_price * 0.90, 0),  # 10% ITM
        round(current_price * 0.95, 0),  # 5% ITM
        round(current_price, 0),          # ATM
    ]
    
    # Suggest 6-month out expiration
    from datetime import datetime, timedelta
    today = datetime.now()
    suggested_exp = today + timedelta(days=180)
    # Adjust to nearest Friday
    days_until_friday = (4 - suggested_exp.weekday()) % 7
    suggested_exp = suggested_exp + timedelta(days=days_until_friday)
    
    st.info(f"""
    **Suggested Strikes:**
    - Deep ITM (15%): ${suggested_strikes[0]:.0f}
    - ITM (10%): ${suggested_strikes[1]:.0f}
    - Slight ITM (5%): ${suggested_strikes[2]:.0f}
    - ATM: ${suggested_strikes[3]:.0f}
    
    **Suggested Expiration:** {suggested_exp.strftime('%Y-%m-%d')} (~6 months out)
    """)
    
    st.markdown("---")
    
    # Simplified Roll UI - matches IB's single net price format
    with st.form(key=f"roll_long_form_{pos.position_id}"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**New Long Position:**")
            new_strike = st.number_input(
                "New Strike",
                min_value=1.0, value=float(suggested_strikes[1]), step=1.0,
                key=f"roll_long_new_strike_{pos.position_id}"
            )
            new_exp = st.date_input(
                "New Expiration", 
                value=suggested_exp.date(), 
                key=f"roll_long_new_exp_{pos.position_id}"
            )
        
        with col2:
            st.markdown("**Roll Transaction:**")
            roll_type = st.radio(
                "Roll Type",
                ["Net Debit", "Net Credit", "Even"],
                key=f"roll_long_type_{pos.position_id}",
                horizontal=True
            )
            
            net_roll_input = st.number_input(
                "Net Roll Price ($)",
                min_value=0.0, value=0.50, step=0.05,
                key=f"roll_long_net_{pos.position_id}",
                help="Net price from IB roll order (single transaction)"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("✅ Execute Long Roll", type="primary")
        with col2:
            cancelled = st.form_submit_button("❌ Cancel")
    
    # Process after form submission
    if submitted:
        # Calculate net price based on roll type
        if roll_type == "Even":
            net_roll_price = 0.0
        elif roll_type == "Net Credit":
            net_roll_price = -net_roll_input
        else:
            net_roll_price = net_roll_input
        
        # For backend compatibility
        estimated_exit = pos.long_current_price
        estimated_entry = pos.long_current_price + net_roll_price
        
        try:
            roll_info = trade_log.roll_diagonal_long(
                position_id=pos.position_id,
                exit_price=estimated_exit,
                new_strike=new_strike,
                new_expiration=new_exp.isoformat(),
                new_entry_price=estimated_entry,
                underlying_price=current_price,
                regime="CALM",  # TODO: Get current regime
                notes=f"Rolled from ${pos.long_strike} to ${new_strike} (net {'debit' if net_roll_price > 0 else 'credit'}: ${abs(net_roll_price):.2f})",
            )
            st.success(f"✅ Long leg rolled! Net {'debit' if net_roll_price > 0 else 'credit'}: ${abs(net_roll_price):.2f}")
            st.session_state[f"rolling_long_{pos.position_id}"] = False
            st.rerun()
        except Exception as e:
            st.error(f"Error rolling long: {e}")
    
    if cancelled:
        st.session_state[f"rolling_long_{pos.position_id}"] = False
        st.rerun()

def _render_close_short_form(trade_log, pos):
    """Form to close just the short leg."""
    st.markdown("##### 📕 Close Short Leg")
    
    short = pos.current_short_leg
    if not short or short.status != "open":
        st.warning("No active short leg to close.")
        if st.button("Cancel", key=f"close_short_cancel_{pos.position_id}"):
            st.session_state[f"closing_short_{pos.position_id}"] = False
            st.rerun()
        return
    
    st.info(f"""
    **Current Short Leg:**
    - Strike: ${short.strike:.0f}
    - Expiration: {short.expiration_date}
    - Entry Credit: ${short.entry_credit:.2f}
    - Current Price: ${short.current_price:.2f}
    - Contracts: {short.contracts}
    """)
    
    with st.form(key=f"close_short_form_{pos.position_id}"):
        col1, col2 = st.columns(2)
        with col1:
            buyback_price = st.number_input(
                "Buyback Price ($)",
                min_value=0.0,
                value=float(short.current_price) if short.current_price else 0.05,
                step=0.01,
                key=f"cs_buyback_{pos.position_id}"
            )
        with col2:
            close_reason = st.selectbox(
                "Reason",
                ["closed_manual", "expired_worthless", "expired_itm", "stop_loss", "take_profit"],
                key=f"cs_reason_{pos.position_id}"
            )
        
        # Show P&L preview
        pnl = (short.entry_credit - buyback_price) * 100 * short.contracts
        pnl_color = "green" if pnl >= 0 else "red"
        st.markdown(f"**Estimated P&L:** :{pnl_color}[${pnl:,.0f}]")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("✅ Close Short"):
                trade_log.close_short_leg(
                    pos.position_id,
                    exit_price=buyback_price,
                    exit_reason=close_reason
                )
                st.success("✅ Short leg closed!")
                st.session_state[f"closing_short_{pos.position_id}"] = False
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state[f"closing_short_{pos.position_id}"] = False
                st.rerun()


def _render_close_long_form(trade_log, pos):
    """Form to close the long leg (and entire position)."""
    st.markdown("##### 📘 Close Long Leg")
    
    st.warning("⚠️ Closing the long leg will close the entire diagonal position!")
    
    st.info(f"""
    **Current Long Leg:**
    - Strike: ${pos.long_strike:.0f}
    - Expiration: {pos.long_expiration}
    - Entry Price: ${pos.long_entry_price:.2f}
    - Current Price: ${pos.long_current_price:.2f}
    - Contracts: {pos.contracts}
    """)
    
    short = pos.current_short_leg
    if short and short.status == "open":
        st.warning(f"⚠️ Active short leg (${short.strike} @ ${short.current_price:.2f}) will also be closed!")
    
    with st.form(key=f"close_long_form_{pos.position_id}"):
        col1, col2 = st.columns(2)
        with col1:
            sell_price = st.number_input(
                "Sell Price ($)",
                min_value=0.0,
                value=float(pos.long_current_price) if pos.long_current_price else 1.0,
                step=0.05,
                key=f"cl_sell_{pos.position_id}"
            )
        with col2:
            close_reason = st.selectbox(
                "Reason",
                ["closed_manual", "expired_worthless", "expired_itm", "stop_loss", "take_profit", "regime_change"],
                key=f"cl_reason_{pos.position_id}"
            )
        
        # Show P&L preview
        long_pnl = (sell_price - pos.long_entry_price) * 100 * pos.contracts
        short_pnl = pos.short_pnl
        total_pnl = long_pnl + short_pnl - pos.total_commissions
        
        st.markdown(f"""
        **P&L Preview:**
        - Long Leg: ${long_pnl:,.0f}
        - Short Legs (total): ${short_pnl:,.0f}
        - Commissions: -${pos.total_commissions:,.0f}
        - **Total: ${total_pnl:,.0f}**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("✅ Close Position"):
                # Close any open short first
                if short and short.status == "open":
                    trade_log.close_short_leg(pos.position_id, exit_price=0.0, exit_reason="closed_with_long")
                # Close the position
                trade_log.close_diagonal(
                    pos.position_id,
                    long_exit_price=sell_price,
                    exit_reason=close_reason
                )
                st.success("✅ Position closed!")
                st.session_state[f"closing_long_{pos.position_id}"] = False
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state[f"closing_long_{pos.position_id}"] = False
                st.rerun()


def _render_sell_short_form(trade_log, pos):
    """Form to sell a new short leg when position has none."""
    st.markdown("##### 📈 Sell New Short Leg")
    
    # Get current underlying price for suggestions
    try:
        import yfinance as yf
        ticker = yf.Ticker("UVXY")
        current_price = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice', 38.0)
    except:
        current_price = 38.0
    
    # Roll suggestions based on current price
    st.markdown("##### 💡 Suggestions")
    suggested_strikes = [
        round(current_price * 1.02, 0),  # 2% OTM
        round(current_price * 1.05, 0),  # 5% OTM  
        round(current_price * 1.10, 0),  # 10% OTM
    ]
    
    # Suggest next Friday expiration
    from datetime import datetime, timedelta
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7  # Next week if today is Friday
    suggested_exp = today + timedelta(days=days_until_friday)
    
    st.info(f"""
    **Current UVXY:** ${current_price:.2f}
    
    **Suggested Strikes (OTM):**
    - Conservative (2% OTM): ${suggested_strikes[0]:.0f}
    - Moderate (5% OTM): ${suggested_strikes[1]:.0f}  
    - Aggressive (10% OTM): ${suggested_strikes[2]:.0f}
    
    **Suggested Expiration:** {suggested_exp.strftime('%Y-%m-%d')} (next Friday)
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        new_strike = st.number_input(
            "Strike",
            min_value=1.0, value=float(suggested_strikes[1]), step=0.5,
            key=f"sell_short_strike_{pos.position_id}"
        )
    with col2:
        new_exp = st.date_input(
            "Expiration", 
            value=suggested_exp.date(),
            key=f"sell_short_exp_{pos.position_id}"
        )
    with col3:
        new_credit = st.number_input(
            "Credit ($)",
            min_value=0.01, value=0.20, step=0.05,
            key=f"sell_short_credit_{pos.position_id}"
        )
    
    total_credit = new_credit * pos.contracts * 100
    st.success(f"💰 Total Credit: ${total_credit:.2f} for {pos.contracts} contracts")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Sell Short", key=f"sell_short_submit_{pos.position_id}"):
            try:
                pos.add_short_leg(new_strike, new_exp.isoformat(), new_credit)
                trade_log._save()
                st.success(f"✅ Sold short ${new_strike} @ ${new_credit:.2f}")
                st.session_state[f"selling_short_{pos.position_id}"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("❌ Cancel", key=f"sell_short_cancel_{pos.position_id}"):
            st.session_state[f"selling_short_{pos.position_id}"] = False
            st.rerun()


def _render_expire_confirm(trade_log, pos):
    """Confirmation dialog for expiring short worthless."""
    st.markdown("##### 🎉 Short Expired Worthless - Profit!")
    
    short = pos.current_short_leg
    if not short:
        st.warning("No open short leg to expire")
        if st.button("❌ Cancel", key=f"expire_cancel_no_short_{pos.position_id}"):
            st.session_state[f"expiring_{pos.position_id}"] = False
            st.rerun()
        return
    
    credit_received = short.entry_credit * pos.contracts * 100
    
    st.success(f"""
    **Short leg expired OTM (out-of-the-money)**
    
    - Strike: ${short.strike:.2f}
    - Expiration: {short.expiration_date}
    - Credit received: ${credit_received:.2f} (now realized profit!)
    
    This will mark the short as expired at $0. Your LEAP stays open.
    You can then roll into a new short or wait.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Expiration", key=f"expire_confirm_{pos.position_id}", type="primary"):
            trade_log.expire_diagonal_short(pos.position_id)
            st.success(f"✅ Short expired! Credit of ${credit_received:.2f} locked in.")
            st.session_state[f"expiring_{pos.position_id}"] = False
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"expire_cancel_{pos.position_id}"):
            st.session_state[f"expiring_{pos.position_id}"] = False
            st.rerun()


def _render_delete_confirm(trade_log, pos):
    """Confirmation dialog for deleting a position."""
    st.markdown("##### 🗑️ Delete Position")
    st.warning(f"⚠️ Are you sure you want to delete **{pos.variant_name}** ({pos.position_id})?")
    st.error("This action cannot be undone!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", key=f"delete_confirm_{pos.position_id}", type="primary"):
            if trade_log.delete_diagonal(pos.position_id):
                st.success("✅ Position deleted")
                st.session_state[f"deleting_{pos.position_id}"] = False
                st.rerun()
            else:
                st.error("Failed to delete position")
    
    with col2:
        if st.button("❌ Cancel", key=f"delete_cancel_{pos.position_id}"):
            st.session_state[f"deleting_{pos.position_id}"] = False
            st.rerun()



def _render_roll_analytics(trade_log):
    """Render roll analytics and statistics."""
    st.subheader("📊 Roll Analytics")
    
    roll_stats = trade_log.get_roll_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rolls", roll_stats["total_rolls"])
    with col2:
        st.metric("Positions with Rolls", roll_stats["positions_with_rolls"])
    with col3:
        st.metric("Total Roll Credits", f"${roll_stats['total_roll_credits']:,.2f}")
    with col4:
        st.metric("Avg Roll Credit", f"${roll_stats['avg_roll_credit']:.2f}")
    
    # Roll history across all positions
    st.markdown("---")
    st.markdown("**All Rolls**")
    
    all_rolls = []
    for pos in trade_log.get_all_diagonals():
        for roll in pos.roll_history:
            all_rolls.append({
                "Date": roll.roll_date,
                "Position": pos.variant_name,
                "Old Strike": f"${roll.old_strike}",
                "New Strike": f"${roll.new_strike}",
                "Buy Back": f"${roll.old_exit_price:.2f}",
                "New Credit": f"${roll.new_credit:.2f}",
                "Net Credit": f"${roll.roll_credit:.2f}",
                "Underlying": f"${roll.underlying_price:.2f}",
            })
    
    if all_rolls:
        import pandas as pd
        df = pd.DataFrame(all_rolls)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Roll History",
            csv,
            "roll_history.csv",
            "text/csv",
        )
    else:
        st.info("No rolls recorded yet.")




def render_trade_log(trade_log=None):
    """Trade Log - View and manage all paper trades."""
    st.title("📒 Trade Log")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = trade_log or get_trade_log()

    summary = trade_log.get_summary()
    
    # Summary metrics (includes both simple trades and diagonal positions)
    diagonals = trade_log.get_all_diagonals()
    open_diagonals = trade_log.get_open_diagonals()
    closed_diagonals = [d for d in diagonals if d.status != "open"]
    diagonal_pnl = sum(d.total_pnl for d in diagonals)
    
    total_trades = summary["total_trades"] + len(diagonals)
    total_open = summary["open_trades"] + len(open_diagonals)
    total_closed = summary["closed_trades"] + len(closed_diagonals)
    total_pnl = summary["combined_pnl"] + diagonal_pnl
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Open", total_open)
    with col3:
        st.metric("Closed", total_closed)
    with col4:
        st.metric("Win Rate", f"{summary['win_rate']:.0%}" if total_closed > 0 else "N/A")
    with col5:
        pnl_delta = "↑" if total_pnl > 0 else "↓" if total_pnl < 0 else ""
        st.metric("Total P&L", f"${total_pnl:,.0f}")
    
    st.markdown("---")
    
    # Tabs for different views (Diagonal Positions is default/first)
    tab1, tab2, tab3, tab_real = st.tabs(["🔄 Diagonal Positions", "📋 Simple Trades", "📊 Roll Analytics", "💵 Real Trades"])
    
    with tab1:
        _render_diagonal_positions(trade_log)

    with tab_real:
        from real_trade_log import get_real_trade_log
        rtl = get_real_trade_log()
        # ── Summary bar
        real_open = rtl.open_positions()
        summary   = rtl.summary()
        st.markdown(f"""
        <div style="background:#1a0a00;border:2px solid #3d1f00;border-radius:8px;
                    padding:14px 20px;margin-bottom:16px">
          <span style="font-size:16px;font-weight:800;color:#ff6b35">
            💵 REAL MONEY TRADES</span>
          <span style="font-size:11px;color:#664422;margin-left:16px">
            {summary['open_count']} open &nbsp;·&nbsp;
            P&L: <b style="color:{'#00e5a0' if summary['total_pnl']>=0 else '#ff3366'}">
            ${summary['total_pnl']:+,.0f}</b> &nbsp;·&nbsp;
            Commissions: ${summary['total_commissions']:,.2f} &nbsp;·&nbsp;
            Slippage: ${summary['total_slippage']:+,.2f}
          </span>
        </div>
        """, unsafe_allow_html=True)
        # Real trade entry and position list
        from real_trade_ui import render_real_trade_section
        render_real_trade_section()

    with tab3:
        _render_roll_analytics(trade_log)
    
    with tab2:
        # Filters for simple trades
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "Open", "Closed"],
                key="trade_log_status_filter"
            )
        with col2:
            variant_filter = st.selectbox(
                "Filter by Variant",
                ["All"] + [role.value for role in VariantRole],
                key="trade_log_variant_filter"
            )
        
        # Multi-Leg Trade Entry Form
    with st.expander("➕ Add Trade Manually", expanded=False):
        st.markdown("Record a diagonal spread (Long LEAP + Short Weekly) executed outside the system.")
        
        manual_variant = st.selectbox(
            "Variant",
            options=[role.value for role in VariantRole],
            key="manual_trade_variant"
        )
        
        st.markdown("---")
        
        # LONG LEG
        st.markdown("##### 📈 Long Leg (LEAP Call)")
        long_col1, long_col2, long_col3 = st.columns(3)
        with long_col1:
            long_strike = st.number_input(
                "Long Strike",
                min_value=1.0, max_value=200.0, value=40.0, step=0.5,
                key="manual_long_strike"
            )
        with long_col2:
            long_expiration = st.date_input(
                "Long Expiration",
                key="manual_long_expiration"
            )
        with long_col3:
            long_debit = st.number_input(
                "Long Debit ($)",
                min_value=0.01, max_value=50.0, value=3.50, step=0.05,
                key="manual_long_debit",
                help="Price paid per contract for LEAP"
            )
        
        st.markdown("---")
        
        # SHORT LEG
        long_only_mode = st.checkbox(
            "📌 Long Only — short leg not sold yet",
            value=False,
            key="rtl_long_only_mode",
            help="Use when you've bought the LEAP but haven't sold the short call yet"
        )
        if long_only_mode:
            st.info("✅ Long-only position. Add short leg later via '📈 Add Short Leg' button.")
        if not long_only_mode:
            st.markdown("##### 📉 Short Leg (Weekly Call)")
        short_col1, short_col2, short_col3 = st.columns(3)
        with short_col1:
            short_strike = st.number_input(
                "Short Strike",
                min_value=1.0, max_value=200.0, value=38.0, step=0.5,
                key="manual_short_strike"
            )
        with short_col2:
            short_expiration = st.date_input(
                "Short Expiration",
                key="manual_short_expiration"
            )
        with short_col3:
            short_credit = st.number_input(
                "Short Credit ($)",
                min_value=0.01, max_value=20.0, value=0.80, step=0.05,
                key="manual_short_credit",
                help="Credit received per contract for weekly"
            )
        
        st.markdown("---")
        
        # POSITION INFO
        pos_col1, pos_col2 = st.columns(2)
        with pos_col1:
            manual_contracts = st.number_input(
                "Contracts",
                min_value=1, max_value=100, value=5, step=1,
                key="manual_trade_contracts"
            )
        with pos_col2:
            manual_notes = st.text_input(
                "Notes (optional)",
                key="manual_trade_notes"
            )
        
        # Calculate net debit/credit
        net_position = short_credit - long_debit
        net_type = "CREDIT" if net_position > 0 else "DEBIT"
        total_cost = abs(net_position) * manual_contracts * 100
        
        st.markdown(f"""
        **Position Summary:**
        - Net {net_type}: **${abs(net_position):.2f}** per spread
        - Total {'Credit' if net_position > 0 else 'Cost'}: **${total_cost:.2f}** for {manual_contracts} contracts
        - Max Risk: ${long_debit * manual_contracts * 100:.2f} (if LEAP expires worthless)
        """)
        
        if st.button("📥 Record Diagonal Spread", key="manual_trade_submit"):
            try:
                variant_names = {
                    "v1_income_harvester": "V1 Income Harvester",
                    "v2_mean_reversion": "V2 Mean Reversion",
                    "v3_shock_absorber": "V3 Shock Absorber",
                    "v4_tail_hunter": "V4 Tail Hunter",
                    "v5_regime_allocator": "V5 Regime Allocator",
                }
                variant_name = variant_names.get(manual_variant, manual_variant)
                
                # Calculate entry commission
                fee_per_contract = 0.65
                entry_commission = fee_per_contract * manual_contracts * 2  # Long buy + Short sell
                
                # Store as diagonal position with proper roll tracking
                pos = trade_log.open_diagonal(
                    variant_id=manual_variant.upper(),
                    variant_name=variant_name,
                    contracts=manual_contracts,
                    long_strike=long_strike,
                    long_expiration=long_expiration.isoformat(),
                    long_price=long_debit,
                    short_strike=short_strike,
                    short_expiration=short_expiration.isoformat(),
                    short_credit=short_credit,
                    entry_regime="CALM",  # Default
                    entry_vix_level=0.0,  # Not specified
                    fee_per_contract=fee_per_contract,
                    notes=manual_notes,
                )
                st.success(f"✅ Recorded {variant_name} diagonal spread! (ID: {pos.position_id})")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    
    # Get trades based on filter
    if status_filter == "Open":
        trades = trade_log.get_open_trades()
    elif status_filter == "Closed":
        trades = trade_log.get_closed_trades()
    else:
        trades = trade_log.get_all_trades()
    
    # Apply variant filter
    if variant_filter != "All":
        trades = [t for t in trades if t.variant_role.value == variant_filter]
    
    # Display trades
    if not trades:
        st.info("No trades found. Execute signals to create trades.")
        return
    
    st.subheader(f"Trades ({len(trades)})")
    
    for trade in sorted(trades, key=lambda t: t.entry_date, reverse=True):
        status_icon = "🟢" if trade.status.value == "open" else "🔴"
        pnl_color = "green" if trade.total_pnl >= 0 else "red"
        
        with st.expander(
            f"{status_icon} {trade.variant_name} | {trade.entry_date.strftime('%Y-%m-%d')} | "
            f"${trade.total_pnl:+,.0f}",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Trade ID:** {trade.position_id}")
                st.write(f"**Signal ID:** {trade.signal_id}")
                st.write(f"**Entry:** {trade.entry_date.strftime('%Y-%m-%d %H:%M')}")
            with col2:
                st.write(f"**Regime:** {trade.entry_regime.value}")
                st.write(f"**Contracts:** {trade.total_contracts}")
                st.write(f"**Days Held:** {trade.days_held}")
            with col3:
                st.write(f"**Entry Debit:** ${trade.entry_debit:,.2f}")
                st.markdown(f"**P&L:** <span style='color:{pnl_color}'>${trade.total_pnl:+,.2f}</span>", unsafe_allow_html=True)


def render_real_trade_log_page():
    """Trade Log Real — mirrors Trade Log page but uses real_trade_log.json."""
    from real_trade_log import get_real_trade_log, reset_real_trade_log_cache
    import pandas as pd

    rtl = get_real_trade_log()
    # Auto-fetch live prices on page load if any long_current_price is 0
    _needs_px = any(float(p.long_current_price or 0) <= 0
                    for p in rtl.diagonal_positions.values())
    if _needs_px:
        try:
            update_diagonal_live_prices(rtl, symbol="UVXY")
            reset_real_trade_log_cache()
            rtl = get_real_trade_log()
        except Exception:
            pass
    open_pos   = rtl.open_positions()
    all_pos    = rtl.diagonal_positions
    summary    = rtl.summary()
    total_pnl  = summary["total_pnl"]
    total_comm = summary["total_commissions"]
    total_slip = summary["total_slippage"]
    closed     = [p for p in all_pos.values() if p.status == "closed"]
    win_rate   = (sum(1 for p in closed if p.total_pnl > 0) / len(closed) * 100
                  if closed else 0.0)

    # ── Header
    st.markdown("""
    <div style="background:#1a0a00;border:2px solid #3d1f00;border-radius:8px;
                padding:16px 20px;margin-bottom:20px">
      <span style="font-size:22px;font-weight:800;color:#ff6b35">
        💵 Trade Log — Real Money</span>
      <span style="font-size:11px;color:#664422;margin-left:12px">
        Fidelity / IB · Live capital · Separate from paper trades</span>
    </div>""", unsafe_allow_html=True)

    # ── Stats bar
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Positions", len(all_pos))
    c2.metric("Open",            len(open_pos))
    c3.metric("Closed",          len(closed))
    c4.metric("Win Rate",        f"{win_rate:.0f}%")
    c5.metric("Total P&L",       f"${total_pnl:+,.0f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Commissions Paid", f"${total_comm:.2f}")
    c7.metric("Total Slippage",   f"${total_slip:+.2f}")
    c8.metric("Net After Costs",  f"${summary['net_after_costs']:+,.0f}")

    st.divider()

    tab_open, tab_new, tab_history, tab_analytics = st.tabs([
        "📋 Open Positions", "➕ New Entry",
        "📊 History", "📈 Analytics"
    ])

    # ══ OPEN POSITIONS ══════════════════════════════════════
    with tab_open:
        if not open_pos:
            st.info("No open real money positions. Use 'New Entry' to add one.")
        else:
            # Live P&L table
            if st.button("🔄 Refresh Prices", key="rtl_refresh"):
                try:
                    update_diagonal_live_prices(rtl, symbol="UVXY")
                    reset_real_trade_log_cache()
                except Exception as e:
                    st.warning(f"Price fetch: {e}")
                st.rerun()

            rows = []
            for pid, pos in sorted(open_pos.items(),
                                   key=lambda x: x[1].variant_id):
                short = pos.current_short_leg
                dte   = pos.days_to_expiry()
                rows.append({
                    "Variant":     pos.variant_name,
                    "Broker":      f"{pos.broker} {pos.account_id}",
                    "Contracts":   pos.contracts,
                    "Long":        f"${pos.long_strike:.0f} exp {pos.long_expiration}",
                    "Long Fill":   f"${pos.long_fill_price:.2f}",
                    "Short":       f"${short.strike:.0f} exp {short.expiration_date}" if short else "—",
                    "Short Fill":  f"${short.fill_price:.2f}" if short else "—",
                    "DTE":         dte,
                    "Net Credits": f"${pos.net_short_credits:,.0f}",
                    "Coverage%":   f"{pos.short_coverage_pct:.0f}%",
                    "Commission":  f"${pos.total_commissions:.2f}",
                    "Slippage":    f"${pos.total_slippage:+.2f}",
                    "Total P&L":   f"${pos.total_pnl:+,.0f}",
                    "Rolls":       len(pos.roll_history),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            st.divider()

            # Position cards with roll forms
            for pid, pos in sorted(open_pos.items(),
                                   key=lambda x: x[1].variant_id):
                short = pos.current_short_leg
                dte   = pos.days_to_expiry()
                label = (f"⚠️ ROLL NOW" if dte <= 0 else
                         f"📋 ORDER ROLL" if dte == 1 else "✓ HOLD")
                color = "#ff3366" if dte <= 0 else "#ff9800" if dte == 1 else "#00e5a0"

                with st.expander(
                    f"💵 {pos.variant_name}  |  {pos.broker}  |  "
                    f"P&L: ${pos.total_pnl:+,.0f}  |  DTE: {dte}d  |  {label}",
                    expanded=(dte <= 1)
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"**Entry** {pos.entry_date}")
                        st.markdown(f"**Regime** {pos.entry_regime}")
                        st.markdown(f"**Contracts** {pos.contracts}")
                    with col2:
                        st.markdown(f"**Long** ${pos.long_strike:.0f} exp {pos.long_expiration}")
                        st.markdown(f"**Long fill** ${pos.long_fill_price:.2f}  "
                                    f"*(cost: ${pos.long_cost:,.0f})*")
                        _lc = pos.long_current_price
                        if _lc > 0:
                            _lpnl = (_lc - pos.long_fill_price) * pos.contracts * 100
                            st.markdown(f"**Long current** ${_lc:.2f}  "
                                        f"*(P&L: ${_lpnl:+,.0f})*")
                        else:
                            st.markdown("**Long current** *(pending refresh)*")
                    with col3:
                        if short:
                            st.markdown(f"**Short** ${short.strike:.0f} exp {short.expiration_date}")
                            st.markdown(f"**Short fill** ${short.fill_price:.2f}")
                            st.markdown(f"**Short mid** ${short.entry_credit:.2f}")
                        else:
                            st.warning("No active short leg")
                    with col4:
                        st.metric("Net Credits",  f"${pos.net_short_credits:,.0f}")
                        st.metric("Coverage",     f"{pos.short_coverage_pct:.0f}%")
                        st.metric("Slippage",     f"${pos.total_slippage:+.2f}")

                    # Roll form
                    if short and short.is_open():
                        st.markdown("---")
                        st.markdown("#### 🔄 Roll Short Leg")
                        try:
                            import yfinance as yf
                            uvxy_px = float(yf.Ticker("UVXY")
                                .history(period="1d",interval="1m")
                                ["Close"].iloc[-1])
                        except:
                            uvxy_px = 0.0

                        from datetime import datetime, timedelta
                        today_dt = datetime.now()
                        days_fri = (4 - today_dt.weekday()) % 7 or 7
                        next_fri = (today_dt + timedelta(days=days_fri)).date()

                        with st.form(f"rtl_roll_{pid}"):
                            rc1, rc2, rc3 = st.columns(3)
                            with rc1:
                                bb_mid  = st.number_input("Buy-back mid",  value=0.10, step=0.01, key=f"rbb_mid_{pid}")
                                bb_fill = st.number_input("Buy-back fill", value=0.10, step=0.01, key=f"rbb_fill_{pid}")
                            with rc2:
                                ns = st.number_input("New strike", value=float(short.strike+1), step=1.0, key=f"rns_{pid}")
                                ne = st.date_input("New expiry", value=next_fri, key=f"rne_{pid}")
                            with rc3:
                                nc_mid  = st.number_input("New credit mid",  value=1.50, step=0.01, key=f"rnc_mid_{pid}")
                                nc_fill = st.number_input("New credit fill", value=1.50, step=0.01, key=f"rnc_fill_{pid}")
                            reason = st.selectbox("Reason",
                                ["order_roll","delta_trigger","itm_threat","manual"],
                                key=f"rreason_{pid}")
                            notes = st.text_input("Notes", key=f"rnotes_{pid}")
                            submitted = st.form_submit_button("✅ Execute Roll", type="primary")

                        if submitted:
                            try:
                                rtl.roll_short(
                                    position_id      = pid,
                                    old_exit_price   = bb_mid,
                                    old_fill_price   = bb_fill,
                                    new_strike       = ns,
                                    new_expiration   = ne.isoformat(),
                                    new_credit       = nc_mid,
                                    new_fill_price   = nc_fill,
                                    underlying_price = uvxy_px,
                                    roll_reason      = reason,
                                    notes            = notes,
                                )
                                reset_real_trade_log_cache()
                                st.success(
                                    f"✅ Rolled → ${ns:.0f} exp {ne}  "
                                    f"Net credit: ${nc_fill-bb_fill:.2f}/c  "
                                    f"Slippage: ${(nc_fill-nc_mid)+(bb_mid-bb_fill):.2f}")
                                st.rerun()
                            except Exception as e:
                                import traceback
                                st.error(f"Roll failed: {e}")
                                st.code(traceback.format_exc())

                    # ── Action buttons ─────────────────────────────
                    st.markdown("---")
                    has_active_short = short and short.is_open()

                    btn1, btn2, btn3, btn4, btn5, btn6 = st.columns(6)

                    with btn1:
                        if has_active_short:
                            _expire = st.button("🎉 Expire Worthless",
                                                key=f"rexpire_{pid}",
                                                help="Mark short expired at $0")
                        else:
                            _expire = False
                            st.caption("No active short")

                    with btn2:
                        _add_short = st.button("📈 Add Short Leg",
                                               key=f"radd_short_{pid}",
                                               help="Sell new short after expiry")

                    with btn3:
                        _refresh = st.button("💰 Refresh Prices",
                                             key=f"rpx_{pid}")

                    with btn4:
                        _edit = st.button("✏️ Edit Long",
                                          key=f"redit_{pid}",
                                          help="Update long leg price/strike")

                    with btn5:
                        _close_all = st.button("🚪 Close All",
                                               key=f"rclose_{pid}",
                                               help="Close entire position")

                    with btn6:
                        _delete = st.button("🗑️ Delete",
                                            key=f"rdel_{pid}",
                                            help="Permanently delete position")

                    # ── Expire worthless ──
                    if _expire:
                        try:
                            rtl.close_short_leg(pid, exit_price=0.0,
                                                exit_reason="expired_worthless")
                            reset_real_trade_log_cache()
                            st.success("✅ Short marked expired worthless.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                    # ── Add new short leg ──
                    if _add_short or st.session_state.get(f"rshowing_add_short_{pid}"):
                        st.session_state[f"rshowing_add_short_{pid}"] = True
                        st.markdown("##### 📈 Add New Short Leg")
                        from datetime import datetime, timedelta
                        today_dt = datetime.now()
                        days_fri = (4 - today_dt.weekday()) % 7 or 7
                        next_fri = (today_dt + timedelta(days=days_fri)).date()
                        with st.form(f"radd_short_form_{pid}"):
                            as1, as2, as3 = st.columns(3)
                            with as1:
                                as_strike = st.number_input("Strike", value=float(short.strike if short else pos.long_strike), step=1.0, key=f"ras_k_{pid}")
                            with as2:
                                as_exp = st.date_input("Expiry", value=next_fri, key=f"ras_exp_{pid}")
                            with as3:
                                as_mid  = st.number_input("Mid price", value=1.50, step=0.01, key=f"ras_mid_{pid}")
                                as_fill = st.number_input("Fill price", value=1.50, step=0.01, key=f"ras_fill_{pid}")
                                as_comm = st.number_input("Commission", value=0.65, step=0.01, key=f"ras_comm_{pid}")
                            as_sub = st.form_submit_button("✅ Add Short Leg", type="primary")
                        if as_sub:
                            try:
                                leg = rtl.add_short_leg(pid, as_strike, as_exp.isoformat(), as_fill)
                                if leg:
                                    leg.entry_credit = as_mid
                                    leg.commission = as_comm
                                    leg.slippage = round(as_fill - as_mid, 4)
                                    rtl._save()
                                    reset_real_trade_log_cache()
                                    st.session_state[f"rshowing_add_short_{pid}"] = False
                                    st.success(f"✅ Added short ${as_strike:.0f} exp {as_exp}")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Refresh prices ──
                    if _refresh:
                        try:
                            update_diagonal_live_prices(rtl, symbol="UVXY")
                            reset_real_trade_log_cache()
                            st.success("✅ Prices refreshed.")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Price fetch: {e}")

                    # ── Edit long leg ──
                    if _edit or st.session_state.get(f"rshowing_edit_{pid}"):
                        st.session_state[f"rshowing_edit_{pid}"] = True
                        st.markdown("##### ✏️ Edit Long Leg")
                        with st.form(f"redit_form_{pid}"):
                            e1, e2 = st.columns(2)
                            with e1:
                                e_strike = st.number_input("Strike", value=float(pos.long_strike), step=0.5, key=f"re_k_{pid}")
                                e_exp    = st.date_input("Expiry", value=__import__('datetime').date.fromisoformat(pos.long_expiration), key=f"re_exp_{pid}")
                            with e2:
                                e_fill    = st.number_input("Fill price", value=float(pos.long_fill_price), step=0.05, key=f"re_fill_{pid}")
                                e_current = st.number_input("Current price", value=float(pos.long_current_price or 0), step=0.05, key=f"re_cur_{pid}")
                            e_sub = st.form_submit_button("💾 Save", type="primary")
                        if e_sub:
                            try:
                                pos.long_strike = e_strike
                                pos.long_expiration = e_exp.isoformat()
                                pos.long_fill_price = e_fill
                                pos.long_current_price = e_current
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state[f"rshowing_edit_{pid}"] = False
                                st.success("✅ Long leg updated.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Close all ──
                    if _close_all or st.session_state.get(f"rshowing_close_{pid}"):
                        st.session_state[f"rshowing_close_{pid}"] = True
                        st.markdown("##### 🚪 Close Entire Position")
                        with st.form(f"rclose_form_{pid}"):
                            cl1, cl2 = st.columns(2)
                            with cl1:
                                cl_long_exit  = st.number_input("Long exit price", value=float(pos.long_current_price or 0), step=0.05, key=f"rcl_long_{pid}")
                            with cl2:
                                if short and short.is_open():
                                    cl_short_exit = st.number_input("Short buy-back price", value=0.05, step=0.01, key=f"rcl_short_{pid}")
                                cl_reason = st.selectbox("Reason", ["take_profit","stop_loss","manual","long_expiring"], key=f"rcl_reason_{pid}")
                            cl_sub = st.form_submit_button("🚪 Confirm Close", type="primary")
                        if cl_sub:
                            try:
                                if short and short.is_open():
                                    rtl.close_short_leg(pid, exit_price=cl_short_exit, exit_reason=cl_reason)
                                pos.status = "closed"
                                import datetime
                                pos.close_date = datetime.date.today().isoformat()
                                pos.close_reason = cl_reason
                                pos.long_current_price = cl_long_exit
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state[f"rshowing_close_{pid}"] = False
                                st.success(f"✅ Position closed. Final P&L: ${pos.total_pnl:+,.0f}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Delete ──
                    if _delete:
                        st.session_state[f"rconfirm_del_{pid}"] = True
                    if st.session_state.get(f"rconfirm_del_{pid}"):
                        st.warning(f"⚠️ Delete **{pos.variant_name}** permanently?")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("✅ Yes, Delete", key=f"rdelconfirm_{pid}"):
                                del rtl.diagonal_positions[pid]
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state.pop(f"rconfirm_del_{pid}", None)
                                st.success("Deleted.")
                                st.rerun()
                        with dc2:
                            if st.button("❌ Cancel", key=f"rdelcancel_{pid}"):
                                st.session_state.pop(f"rconfirm_del_{pid}", None)
                                st.rerun()

                    # ── Roll history ──
                    st.markdown("---")
                    if True:  # always show roll section
                        rh_col1, rh_col2, rh_col3 = st.columns([4, 1, 1])
                        with rh_col1:
                            st.markdown("**🔄 Roll History**")
                        with rh_col2:
                            if st.button("✏️ Edit Rolls", key=f"redit_rolls_{pid}"):
                                st.session_state[f"rediting_rolls_{pid}"] = not st.session_state.get(f"rediting_rolls_{pid}", False)
                        with rh_col3:
                            if st.button("🔄 Recalc", key=f"rrecalc_{pid}"):
                                try:
                                    pos.recalc_roll_totals()
                                    rtl._save()
                                    reset_real_trade_log_cache()
                                    st.success("✅ Recalculated")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"{e}")
                        # Spreadsheet view
                        import pandas as pd
                        roll_rows = []
                        for r in pos.roll_history:
                            roll_rows.append({
                                "Date":       getattr(r, "roll_date", ""),
                                "Old Strike": f"${getattr(r, 'old_strike', 0):.0f}",
                                "Old Exp":    getattr(r, "old_expiration", ""),
                                "Old Exit":   f"${getattr(r, 'old_exit_price', 0):.2f}",
                                "Old Fill":   f"${getattr(r, 'old_fill_price', getattr(r, 'old_exit_price', 0)):.2f}",
                                "New Strike": f"${getattr(r, 'new_strike', 0):.0f}",
                                "New Exp":    getattr(r, "new_expiration", ""),
                                "New Credit": f"${getattr(r, 'new_credit', 0):.2f}",
                                "New Fill":   f"${getattr(r, 'new_fill_price', getattr(r, 'new_credit', 0)):.2f}",
                                "Net Credit": f"${getattr(r, 'roll_credit', 0):.2f}",
                                "BB Slip":    f"${getattr(r,'old_fill_price',getattr(r,'old_exit_price',0)) - getattr(r,'old_exit_price',0):+.2f}",
                                "Cr Slip":    f"${getattr(r,'new_fill_price',getattr(r,'new_credit',0)) - getattr(r,'new_credit',0):+.2f}",
                                "UVXY":       f"${getattr(r, 'underlying_price', 0):.2f}",
                                "Reason":     getattr(r, "roll_reason", ""),
                                "Notes":      getattr(r, "notes", ""),
                            })
                        st.dataframe(pd.DataFrame(roll_rows),
                                     use_container_width=True, hide_index=True)
                        # Totals
                        total_net = sum(getattr(r, "roll_credit", 0) for r in pos.roll_history)
                        total_bb_slip = sum(
                            getattr(r,'old_fill_price',getattr(r,'old_exit_price',0)) - getattr(r,'old_exit_price',0)
                            for r in pos.roll_history)
                        total_cr_slip = sum(
                            getattr(r,'new_fill_price',getattr(r,'new_credit',0)) - getattr(r,'new_credit',0)
                            for r in pos.roll_history)
                        ts1, ts2, ts3, ts4 = st.columns(4)
                        ts1.metric("Total Net Credits", f"${total_net:,.2f}")
                        ts2.metric("Total BB Slippage", f"${total_bb_slip:+.2f}")
                        ts3.metric("Total Cr Slippage", f"${total_cr_slip:+.2f}")
                        ts4.metric("Total Rolls", len(pos.roll_history))

                        # Edit mode
                        if st.session_state.get(f"rediting_rolls_{pid}"):
                            _render_real_roll_edit_form(rtl, pos)
                    if pos.roll_history:
                        pass  # already handled above
                    if False:  # placeholder to avoid empty block
                        st.markdown("---")
                        st.markdown("**Roll History**")
                        rh = [{
                            "Date":       r.roll_date,
                            "Old Strike": f"${r.old_strike:.0f}",
                            "BB Mid":     f"${getattr(r, 'old_exit_price', getattr(r, 'exit_price', 0.0)):.2f}",
                            "BB Fill":    f"${getattr(r, 'old_fill_price', getattr(r, 'exit_price', 0.0)):.2f}",
                            "BB Slip":    f"${getattr(r,'old_fill_price',0)-getattr(r,'old_exit_price',0):+.2f}",
                            "New Strike": f"${r.new_strike:.0f}",
                            "Cr Mid":     f"${getattr(r, 'new_credit', r.new_credit):.2f}",
                            "Cr Fill":    f"${getattr(r, 'new_fill_price', r.new_credit):.2f}",
                            "Cr Slip":    f"${getattr(r,'new_fill_price',r.new_credit)-r.new_credit:+.2f}",
                            "Net Credit": f"${r.roll_credit:.2f}",
                            "Reason":     r.roll_reason,
                        } for r in pos.roll_history]
                        st.dataframe(pd.DataFrame(rh),
                                     use_container_width=True, hide_index=True)

                    # Close position
                    if st.button(f"❌ Close Position", key=f"rtl_close_{pid}"):
                        rtl.close_position(pid, reason="manual")
                        reset_real_trade_log_cache()
                        st.rerun()

    # ══ NEW ENTRY ════════════════════════════════════════════
    with tab_new:
        from real_trade_ui import _render_new_entry
        _render_new_entry(rtl)

    # ══ HISTORY ══════════════════════════════════════════════
    with tab_history:
        if not all_pos:
            st.info("No trade history yet.")
        else:
            rows = []
            for pid, pos in sorted(all_pos.items(),
                                   key=lambda x: x[1].entry_date, reverse=True):
                rows.append({
                    "ID":          pid[-12:],
                    "Variant":     pos.variant_name,
                    "Status":      pos.status.upper(),
                    "Entry":       pos.entry_date,
                    "Broker":      pos.broker,
                    "Contracts":   pos.contracts,
                    "Long Strike": f"${pos.long_strike:.0f}",
                    "Long Fill":   f"${pos.long_fill_price:.2f}",
                    "Short Fills": len(pos.short_legs),
                    "Net Credits": f"${pos.net_short_credits:,.0f}",
                    "Total P&L":   f"${pos.total_pnl:+,.0f}",
                    "Commission":  f"${pos.total_commissions:.2f}",
                    "Slippage":    f"${pos.total_slippage:+.2f}",
                    "Coverage%":   f"{pos.short_coverage_pct:.0f}%",
                    "Rolls":       len(pos.roll_history),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # ══ ANALYTICS ════════════════════════════════════════════
    with tab_analytics:
        if not all_pos:
            st.info("No data yet.")
            return

        st.markdown("#### Credit Efficiency vs Long Duration")
        from datetime import date
        today = date.today()
        analytics_rows = []
        for pid, pos in all_pos.items():
            try:
                long_entry  = date.fromisoformat(pos.entry_date)
                long_expiry = date.fromisoformat(pos.long_expiration)
                long_total  = (long_expiry - long_entry).days
                long_elapsed = (today - long_entry).days
                long_frac   = long_elapsed / long_total if long_total > 0 else 0
                cpd = pos.net_short_credits / long_elapsed if long_elapsed > 0 else 0
                proj = cpd * long_total
                be_days = pos.long_cost / cpd if cpd > 0 else 9999
                analytics_rows.append({
                    "Variant":      pos.variant_name,
                    "Status":       pos.status,
                    "%Long Used":   f"{long_frac*100:.0f}%",
                    "Long Cost":    f"${pos.long_cost:,.0f}",
                    "Net Credits":  f"${pos.net_short_credits:,.0f}",
                    "Recovery%":    f"{pos.short_coverage_pct:.0f}%",
                    "$/day":        f"${cpd:.2f}",
                    "Proj Total":   f"${proj:,.0f}",
                    "BE days":      int(be_days),
                    "BB Drag%":     f"{pos.total_buybacks/pos.gross_short_credits*100:.0f}%" if pos.gross_short_credits > 0 else "—",
                    "Commission":   f"${pos.total_commissions:.2f}",
                    "Slippage":     f"${pos.total_slippage:+.2f}",
                })
            except Exception:
                continue
        if analytics_rows:
            st.dataframe(pd.DataFrame(analytics_rows),
                         use_container_width=True, hide_index=True)


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
        ["📊 Research", "📈 Paper Trading", "💵 Real Trading"],
        index=0,
        key="app_mode",
    )
    
    st.sidebar.markdown("---")
    
    if "Real Trading" in mode:
        from real_trade_log import get_real_trade_log
        rtl = get_real_trade_log()
        # ── Real Trading sidebar
        st.sidebar.markdown("## 💵 Real Trading")
        page = st.sidebar.radio(
            "Real Trading Pages",
            [
                "Trade Log Real",
                "Signal Dashboard",
                "Active Trades",
                "Post-Mortem Review",
                "Variant Analytics",
                "System Health",
            ],
            index=0,
            key="real_page",
        )
        st.sidebar.markdown("---")
        # Real Trading context (bottom sidebar)
        real_summary = rtl.summary()
        st.sidebar.markdown("**Real Trading**")
        st.sidebar.markdown(f"Open Positions: **{real_summary['open_count']}**")
        st.sidebar.markdown(
            f"Total P&L: **{'$+' if real_summary['total_pnl'] >= 0 else '$'}"
            f"{real_summary['total_pnl']:,.0f}**")
        st.sidebar.markdown(
            f"Commissions: **${real_summary['total_commissions']:.2f}**")
        st.sidebar.markdown(
            f"Slippage: **${real_summary['total_slippage']:+.2f}**")
        # Dispatch with real trade log
        if page == "Trade Log Real":
            render_real_trade_log_page()
        elif page == "Signal Dashboard":
            render_signal_dashboard(trade_log=rtl)
        elif page == "Active Trades":
            render_active_trades(trade_log=rtl)
        elif page == "Post-Mortem Review":
            render_post_mortem(trade_log=rtl)
        elif page == "Variant Analytics":
            render_variant_analytics(trade_log=rtl)
        elif page == "System Health":
            render_system_health(trade_log=rtl)

    elif "Research" in mode:
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
                "Trade Log",
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
        _ptl = get_trade_log()
        if page == "Signal Dashboard":
            render_signal_dashboard(trade_log=_ptl)
        elif page == "Execution Window":
            render_execution_window(trade_log=_ptl)
        elif page == "Active Trades":
            render_active_trades(trade_log=_ptl)
        elif page == "Post-Mortem Review":
            render_post_mortem(trade_log=_ptl)
        elif page == "Variant Analytics":
            render_variant_analytics(trade_log=_ptl)
        elif page == "Trade Log":
            render_trade_log(trade_log=_ptl)
        elif page == "System Health":
            render_system_health(trade_log=_ptl)


if __name__ == "__main__":
    main()