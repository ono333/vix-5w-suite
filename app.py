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



def send_signal_email_smtp(batch, regime, recipient: str = "onoshin333@gmail.com"):
    """Send email notification showing ALL 5 variants with contract sizes."""
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
    
    # Constants
    ACCOUNT_SIZE = 250_000

    # Count active vs paper test
    recommended_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    paper_only_count = len(batch.variants) - recommended_count
    
    # Regime emoji
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    # Email subject
    subject = f"{regime_emoji} [PAPER TEST] VIX Signal: {regime.regime.value.upper()} ({regime.vix_percentile:.0%}) — {recommended_count} Rec / {paper_only_count} Test"
    
    # Build HTML
    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:14px;background:#fff;color:#333;padding:20px;max-width:850px;margin:0 auto;">
    
    <!-- Header -->
    <div style="text-align:center;border-bottom:3px solid #1f77b4;padding-bottom:15px;margin-bottom:20px;">
        <span style="font-size:24px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="font-size:16px;color:#666;background:#fff3cd;padding:4px 12px;border-radius:4px;display:inline-block;margin-top:8px;">
            📋 PAPER TESTING MODE
        </span>
    </div>
    
    <!-- Market State -->
    <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;color:#495057;margin-bottom:10px;">📈 Market State</div>
        <table style="width:100%;border-collapse:collapse;">
            <tr>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">Regime</div>
                    <div style="font-size:20px;font-weight:bold;">{regime_emoji} {regime.regime.value.upper()}</div>
                </td>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">UVXY Price</div>
                    <div style="font-size:20px;font-weight:bold;">${regime.vix_level:.2f}</div>
                </td>
                <td style="padding:8px;text-align:center;border-right:1px solid #dee2e6;">
                    <div style="font-size:12px;color:#6c757d;">52w Percentile</div>
                    <div style="font-size:20px;font-weight:bold;">{regime.vix_percentile:.0%}</div>
                </td>
                <td style="padding:8px;text-align:center;">
                    <div style="font-size:12px;color:#6c757d;">Confidence</div>
                    <div style="font-size:20px;font-weight:bold;">{regime.confidence:.0%}</div>
                </td>
            </tr>
        </table>
    </div>
    
    <!-- Variant Summary -->
    <div style="display:flex;gap:15px;margin-bottom:20px;">
        <div style="flex:1;background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#155724;">{recommended_count}</div>
            <div style="font-size:12px;color:#155724;">🟢 RECOMMENDED<br>(Live-Ready)</div>
        </div>
        <div style="flex:1;background:#cce5ff;border:1px solid #b8daff;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#004085;">{paper_only_count}</div>
            <div style="font-size:12px;color:#004085;">🔵 PAPER TEST<br>(Observe Only)</div>
        </div>
        <div style="flex:1;background:#e2e3e5;border:1px solid #d6d8db;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#383d41;">5</div>
            <div style="font-size:12px;color:#383d41;">📊 TOTAL<br>(All Generated)</div>
        </div>
    </div>
    
    <!-- Section: RECOMMENDED Variants -->
    <div style="margin-bottom:25px;">
        <div style="font-size:16px;font-weight:bold;color:#155724;background:#d4edda;padding:10px 15px;border-radius:6px 6px 0 0;border:1px solid #c3e6cb;border-bottom:none;">
            🟢 RECOMMENDED VARIANTS (Would Execute in Live Mode)
        </div>
        <div style="border:1px solid #c3e6cb;border-radius:0 0 6px 6px;padding:10px;">
    """
    
    # RECOMMENDED variants first
    for variant in batch.variants:
        is_recommended = regime.regime in variant.active_in_regimes
        if not is_recommended:
            continue
        
        # Calculate contract size
        alloc_dollars = ACCOUNT_SIZE * variant.alloc_pct
        est_risk = variant.long_strike_offset * 100  # $100 per point
        contracts = max(1, min(50, int(alloc_dollars / est_risk))) if est_risk > 0 else 1
        total_risk = contracts * est_risk
        
        # Get roll DTE if exists
        roll_dte = getattr(variant, 'roll_dte_days', 3)
        
        # Fetch real market prices
        try:
            market = get_diagonal_prices(
                symbol="UVXY",
                spot_price=regime.vix_level,
                long_offset=variant.long_strike_offset,
                short_offset=variant.short_strike_offset,
                long_dte_weeks=variant.long_dte_weeks,
            )
            long_strike = market["long_strike"]
            short_strike = market["short_strike"]
            est_credit = market["short_mid"] if market["short_mid"] > 0 else estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
            long_cost = market["long_mid"]
        except:
            long_strike = round(regime.vix_level + variant.long_strike_offset)
            short_strike = round(regime.vix_level + variant.short_strike_offset)
            est_credit = estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
            long_cost = 0
        
        price_targets = compute_price_targets(est_credit, variant.tp_pct, variant.sl_pct)
        
        html += f"""
            <div style="border:2px solid #28a745;margin-bottom:10px;border-radius:6px;overflow:hidden;">
                <div style="background:#28a745;color:#fff;padding:10px 15px;font-weight:bold;font-size:15px;">
                    ✅ {get_variant_display_name(variant.role)}
                </div>
                <div style="padding:12px;background:#f8fff8;">
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr>
                            <td style="padding:5px;width:50%;"><b>Long Strike:</b> ${long_strike:.0f}</td>
                            <td style="padding:5px;"><b>Short Strike:</b> ${short_strike:.0f}</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Short DTE:</b> {variant.short_dte_weeks}w (roll {roll_dte}d)</td>
                        </tr>
                        <tr style="background:#d4edda;">
                            <td style="padding:8px;font-size:14px;"><b>💵 Est. Credit:</b> ${est_credit:.2f}/contract</td>
                            <td style="padding:8px;font-size:14px;"><b>📦 Contracts:</b> {contracts}</td>
                        </tr>
                        <tr style="background:#d4edda;">
                            <td style="padding:8px;"><b>🎯 Target:</b> ${price_targets['target']:.2f} (+${price_targets['profit_per_contract']:.0f})</td>
                            <td style="padding:8px;"><b>🛑 Stop:</b> ${price_targets['stop']:.2f} (-${price_targets['loss_per_contract']:.0f})</td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding:8px;color:#666;font-size:12px;">
                                <b>Max Risk:</b> ${total_risk:,.0f} ({total_risk/ACCOUNT_SIZE:.1%} of ${ACCOUNT_SIZE:,})
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    
    html += """
        </div>
    </div>
    
    <!-- Section: PAPER TEST Variants -->
    <div style="margin-bottom:25px;">
        <div style="font-size:16px;font-weight:bold;color:#004085;background:#cce5ff;padding:10px 15px;border-radius:6px 6px 0 0;border:1px solid #b8daff;border-bottom:none;">
            🔵 PAPER TEST VARIANTS (Observe & Compare — Not Live-Ready)
        </div>
        <div style="border:1px solid #b8daff;border-radius:0 0 6px 6px;padding:10px;">
    """
    
    # PAPER TEST variants
    for variant in batch.variants:
        is_recommended = regime.regime in variant.active_in_regimes
        if is_recommended:
            continue
        
        # Calculate contract size
        alloc_dollars = ACCOUNT_SIZE * variant.alloc_pct
        est_risk = variant.long_strike_offset * 100
        contracts = max(1, min(50, int(alloc_dollars / est_risk))) if est_risk > 0 else 1
        total_risk = contracts * est_risk
        
        # Get roll DTE if exists
        roll_dte = getattr(variant, 'roll_dte_days', 3)
        
        # Active regimes
        active_regimes = ", ".join([r.value.upper() for r in variant.active_in_regimes])
        
        html += f"""
            <div style="border:2px solid #6c757d;margin-bottom:10px;border-radius:6px;overflow:hidden;">
                <div style="background:#6c757d;color:#fff;padding:10px 15px;font-weight:bold;font-size:15px;">
                    🔬 {get_variant_display_name(variant.role)}
                    <span style="float:right;font-size:12px;font-weight:normal;background:#495057;padding:2px 8px;border-radius:3px;">
                        Paper Test Only
                    </span>
                </div>
                <div style="padding:12px;background:#f8f9fa;">
                    <div style="background:#fff3cd;border:1px solid #ffeeba;border-radius:4px;padding:8px;margin-bottom:10px;font-size:12px;color:#856404;">
                        ⚠️ <b>Why not recommended:</b> This variant activates in <b>{active_regimes}</b> regime(s), not {regime.regime.value.upper()}.
                        <br>Track it to validate this filtering logic.
                    </div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr>
                            <td style="padding:5px;width:50%;"><b>Long Strike:</b> ${long_strike:.0f}</td>
                            <td style="padding:5px;"><b>Short Strike:</b> ${short_strike:.0f}</td>
                        </tr>
                        <tr>
                            <td style="padding:5px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                            <td style="padding:5px;"><b>Short DTE:</b> {variant.short_dte_weeks}w (roll {roll_dte}d)</td>
                        </tr>
                        <tr style="background:#e9ecef;">
                            <td style="padding:8px;font-size:14px;"><b>💵 Est. Credit:</b> ${est_credit:.2f}/contract</td>
                            <td style="padding:8px;font-size:14px;"><b>📦 Contracts:</b> {contracts}</td>
                        </tr>
                        <tr style="background:#e9ecef;">
                            <td style="padding:8px;"><b>🎯 Target:</b> ${price_targets['target']:.2f} (+${price_targets['profit_per_contract']:.0f})</td>
                            <td style="padding:8px;"><b>🛑 Stop:</b> ${price_targets['stop']:.2f} (-${price_targets['loss_per_contract']:.0f})</td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding:8px;color:#666;font-size:12px;">
                                <b>Max Risk:</b> ${total_risk:,.0f} ({total_risk/ACCOUNT_SIZE:.1%}) | <b>Activates in:</b> {active_regimes}
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    
    html += f"""
        </div>
    </div>
    
    <!-- Paper Testing Notes -->
    <div style="background:#e7f3ff;border:1px solid #b6d4fe;border-radius:8px;padding:15px;margin-bottom:20px;">
        <div style="font-weight:bold;color:#084298;margin-bottom:8px;">📋 Paper Testing Protocol</div>
        <ul style="margin:0;padding-left:20px;color:#084298;font-size:13px;">
            <li><b>Execute ALL 5 variants</b> in paper trading to collect data</li>
            <li><b>RECOMMENDED variants</b> = what live mode would trade</li>
            <li><b>PAPER TEST variants</b> = observe to validate regime filtering</li>
            <li>Track performance to confirm regime logic is correct</li>
            <li>After 8-12 weeks, compare results to decide graduation</li>
        </ul>
    </div>
    
    <!-- Footer -->
    <div style="text-align:center;padding:15px;border-top:1px solid #dee2e6;color:#6c757d;font-size:12px;">
        VIX 5% Weekly Suite — Paper Testing Signal<br>
        Generated: {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | Account Basis: ${ACCOUNT_SIZE:,}
    </div>
    
    </body>
    </html>
    """
    
    # Send email
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        return True, f"Email sent to {recipient}! (5 variants: {recommended_count} recommended, {paper_only_count} paper test)"
    except Exception as e:
        return False, f"Email failed: {e}"

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
        
        # Variant cards
        for variant in batch.variants:
            # Check if variant is active in current regime
            is_active = regime.regime in variant.active_in_regimes
            
            with st.expander(
                f"{'✅' if is_active else '⛔'} {get_variant_display_name(variant.role)} "
                f"({variant.variant_id})",
                expanded=is_active
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
                    
                    # Use market mid if available, else estimate
                    if market["short_mid"] > 0:
                        est_credit = market["short_mid"]
                    else:
                        est_credit = estimate_entry_credit(regime.vix_level, variant.long_strike_offset, variant.long_dte_weeks)
                    
                    long_cost = market["long_mid"] if market["long_mid"] > 0 else 0
                except Exception:
                    # Fallback to estimates
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





def _render_diagonal_positions(trade_log):
    """Render diagonal positions with roll tracking."""
    st.subheader("🔄 Diagonal Positions with Roll Tracking")
    
    # Get diagonal positions
    diagonals = trade_log.get_all_diagonals()
    open_diagonals = trade_log.get_open_diagonals()
    needing_roll = trade_log.get_diagonals_needing_roll(dte_threshold=3)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Diagonals", len(diagonals))
    with col2:
        st.metric("Open", len(open_diagonals))
    with col3:
        st.metric("Need Roll", len(needing_roll), delta="⚠️" if needing_roll else None)
    with col4:
        roll_stats = trade_log.get_roll_summary()
        st.metric("Total Roll Credits", f"${roll_stats['total_roll_credits']:,.2f}")
    
    # Alert for positions needing roll
    if needing_roll:
        st.warning(f"⚠️ {len(needing_roll)} position(s) need rolling (DTE ≤ 3 days)")
        for pos in needing_roll:
            short = pos.current_short_leg
            st.error(f"🔴 {pos.variant_name}: Short ${short.strike} expires in {short.days_to_expiry()} days!")
    
    st.markdown("---")
    
    # Add new diagonal position
    with st.expander("➕ Open New Diagonal Position", expanded=False):
        _render_diagonal_entry_form(trade_log)
    
    # Display existing positions
    if not diagonals:
        st.info("No diagonal positions. Use the form above to create one.")
        return
    
    for pos in sorted(diagonals, key=lambda p: p.entry_date, reverse=True):
        status_icon = "🟢" if pos.status == "open" else "🔴"
        short = pos.current_short_leg
        pnl_color = "green" if pos.total_pnl >= 0 else "red"
        
        header = f"{status_icon} {pos.variant_name} | {pos.entry_date} | "
        header += f"L${pos.long_strike} / S${short.strike if short else 'N/A'} | "
        header += f"Rolls: {pos.total_rolls}"
        
        with st.expander(header, expanded=pos.status == "open"):
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
                st.markdown(f"**Total P&L:** <span style='color:{pnl_color}'>${pos.total_pnl:+,.0f}</span>", unsafe_allow_html=True)
            
            # Roll history
            if pos.roll_history:
                st.markdown("---")
                st.markdown("**🔄 Roll History**")
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
            
            # Action buttons for open positions
            if pos.status == "open":
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Roll Short", key=f"roll_{pos.position_id}"):
                        st.session_state[f"rolling_{pos.position_id}"] = True
                
                with col2:
                    if st.button("💰 Update Prices", key=f"update_{pos.position_id}"):
                        st.session_state[f"updating_{pos.position_id}"] = True
                
                with col3:
                    if st.button("🚪 Close Position", key=f"close_{pos.position_id}"):
                        st.session_state[f"closing_{pos.position_id}"] = True
                
                # Roll form
                if st.session_state.get(f"rolling_{pos.position_id}"):
                    _render_roll_form(trade_log, pos)
                
                # Update prices form
                if st.session_state.get(f"updating_{pos.position_id}"):
                    _render_price_update_form(trade_log, pos)
                
                # Close form
                if st.session_state.get(f"closing_{pos.position_id}"):
                    _render_close_form(trade_log, pos)


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
    
    net = short_credit - long_price
    st.info(f"Net {'Credit' if net > 0 else 'Debit'}: ${abs(net):.2f} per spread | Total: ${abs(net) * contracts * 100:.2f}")
    
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
            )
            st.success(f"✅ Opened diagonal position: {pos.position_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def _render_roll_form(trade_log, pos):
    """Form to roll a short leg."""
    st.markdown("##### 🔄 Roll Short Leg")
    
    short = pos.current_short_leg
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Current Short: ${short.strike} exp {short.expiration_date}")
        exit_price = st.number_input(
            "Buy Back Price ($)",
            min_value=0.0, max_value=20.0, value=0.10, step=0.05,
            key=f"roll_exit_{pos.position_id}"
        )
    
    with col2:
        new_strike = st.number_input(
            "New Strike",
            min_value=1.0, value=short.strike + 1.0, step=0.5,
            key=f"roll_new_strike_{pos.position_id}"
        )
        new_exp = st.date_input("New Expiration", key=f"roll_new_exp_{pos.position_id}")
        new_credit = st.number_input(
            "New Credit ($)",
            min_value=0.01, value=0.85, step=0.05,
            key=f"roll_new_credit_{pos.position_id}"
        )
    
    underlying = st.number_input("Current Underlying Price", min_value=1.0, value=38.0, key=f"roll_underlying_{pos.position_id}")
    
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
                )
                st.success(f"✅ Rolled: Net credit ${roll.roll_credit:.2f}")
                st.session_state[f"rolling_{pos.position_id}"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
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




def render_trade_log():
    """Trade Log - View and manage all paper trades."""
    st.title("📒 Trade Log")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = get_trade_log()
    summary = trade_log.get_summary()
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Trades", summary["total_trades"])
    with col2:
        st.metric("Open", summary["open_trades"])
    with col3:
        st.metric("Closed", summary["closed_trades"])
    with col4:
        st.metric("Win Rate", f"{summary['win_rate']:.0%}" if summary['closed_trades'] > 0 else "N/A")
    with col5:
        st.metric("Total P&L", f"${summary['combined_pnl']:,.0f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Simple Trades", "🔄 Diagonal Positions", "📊 Roll Analytics"])
    
    with tab2:
        _render_diagonal_positions(trade_log)
    
    with tab3:
        _render_roll_analytics(trade_log)
    
    with tab1:
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
                
                # Store as multi-leg trade
                trade_log.create_trade(
                    variant_id=manual_variant.upper(),
                    variant_name=variant_name,
                    entry_price=abs(net_position),  # Net credit/debit
                    contracts=manual_contracts,
                    long_strike=long_strike,
                    long_expiration=long_expiration.isoformat(),
                    long_debit=long_debit,
                    short_strike=short_strike,
                    short_expiration=short_expiration.isoformat(),
                    short_credit=short_credit,
                    notes=manual_notes,
                )
                st.success(f"✅ Recorded {variant_name} diagonal spread!")
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
                st.write(f"**Trade ID:** {trade.trade_id}")
                st.write(f"**Signal ID:** {trade.signal_id}")
                st.write(f"**Entry:** {trade.entry_date.strftime('%Y-%m-%d %H:%M')}")
            with col2:
                st.write(f"**Regime:** {trade.entry_regime.value}")
                st.write(f"**Contracts:** {trade.total_contracts}")
                st.write(f"**Days Held:** {trade.days_held}")
            with col3:
                st.write(f"**Entry Debit:** ${trade.entry_debit:,.2f}")
                st.markdown(f"**P&L:** <span style='color:{pnl_color}'>${trade.total_pnl:+,.2f}</span>", unsafe_allow_html=True)

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
        elif page == "Trade Log":
            render_trade_log()
        elif page == "System Health":
            render_system_health()


if __name__ == "__main__":
    main()
