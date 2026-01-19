#!/usr/bin/env python3
"""
Adaptive VIX 5-Weekly Backtest Engine

This backtester dynamically adjusts parameters based on the current market 
volatility regime. Instead of using fixed parameters throughout the backtest,
it detects the current regime (Ultra Low, Low, Medium, High, Extreme) and 
applies regime-specific optimized parameters.

Key Features:
- Real-time regime detection based on rolling VIX percentile
- Automatic parameter switching at regime transitions
- Uses optimized params from per-regime grid scans when available
- Falls back to static REGIME_CONFIGS defaults
- Tracks regime transitions and per-regime performance

Usage:
    from core.adaptive_backtester import run_adaptive_backtest
    
    results = run_adaptive_backtest(vix_weekly, params)
    
    # Results include:
    # - equity, weekly_returns, realized_weekly, unrealized_weekly
    # - regime_history: list of regime at each week
    # - regime_transitions: list of transition events
    # - per_regime_stats: performance breakdown by regime
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.regime_adapter import RegimeAdapter, REGIME_CONFIGS, RegimeConfig


# ============================================================
# Black-Scholes Pricing (same as backtester.py)
# ============================================================

def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Vanilla Black-Scholes call."""
    try:
        if S <= 0.0 or K <= 0.0:
            return 0.0
        if T <= 0.0:
            return max(S - K, 0.0)
        if sigma <= 0.0:
            return max(S - K * exp(-r * T), 0.0)
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    except Exception:
        return 0.0


# ============================================================
# Position Data Structure
# ============================================================

@dataclass
class OptionPosition:
    quantity: int
    strike: float
    dte_weeks: int
    is_long: bool
    entry_regime: str = ""  # Track which regime we entered in


# ============================================================
# Helper Functions
# ============================================================

def _to_1d_float_array(seq) -> np.ndarray:
    """Convert sequence to 1D float array."""
    clean = []
    for x in seq:
        if isinstance(x, (int, float, np.number)):
            clean.append(float(x))
        elif isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            try:
                clean.append(float(x[0]))
            except Exception:
                pass
    return np.asarray(clean, dtype=float)


def _get_regime_params(
    regime_name: str,
    mode: str,
    base_params: Dict[str, Any],
    use_optimized: bool = True,
) -> Dict[str, Any]:
    """
    Get parameters for a specific regime.
    
    Tries to load optimized params from history first,
    then falls back to static REGIME_CONFIGS.
    """
    params = dict(base_params)
    
    # Try optimized params first
    if use_optimized:
        try:
            from experiments.per_regime_grid_scan import get_optimized_params_for_regime
            optimized = get_optimized_params_for_regime(mode, regime_name)
            if optimized:
                for key in ["entry_percentile", "sigma_mult", "otm_pts", 
                           "long_dte_weeks", "alloc_pct", "target_mult", "exit_mult"]:
                    if key in optimized and optimized[key] is not None:
                        params[key] = optimized[key]
                return params
        except Exception:
            pass
    
    # Fallback to static config
    if regime_name in REGIME_CONFIGS:
        config = REGIME_CONFIGS[regime_name]
        params.update({
            "entry_percentile": config.entry_percentile,
            "otm_pts": config.otm_pts,
            "sigma_mult": config.sigma_mult,
            "long_dte_weeks": config.long_dte_weeks,
            "alloc_pct": config.alloc_pct,
            "target_mult": config.target_mult,
            "exit_mult": config.exit_mult,
            "mode": config.mode,
        })
    
    return params


# ============================================================
# Adaptive Backtester
# ============================================================

def run_adaptive_backtest(
    vix_weekly: pd.Series,
    params: Dict[str, Any],
    use_optimized_params: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Run adaptive backtest with regime-specific parameters.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Weekly VIX/UVXY closes
    params : dict
        Base parameters (initial_capital, risk_free, fee_per_contract, etc.)
    use_optimized_params : bool
        If True, load optimized params from history for each regime
    progress_cb : callable
        Optional progress callback (current_week, total_weeks)
        
    Returns
    -------
    dict with keys:
        - equity: np.ndarray
        - weekly_returns: np.ndarray
        - realized_weekly: np.ndarray
        - unrealized_weekly: np.ndarray
        - trades: int
        - win_rate: float
        - avg_trade_dur: float
        - trade_log: list[dict]
        - regime_history: list[str]  (regime at each week)
        - regime_transitions: list[dict]  (transition events)
        - per_regime_stats: dict  (performance by regime)
    """
    # Initialize regime adapter
    lookback = int(params.get("entry_lookback_weeks", 52))
    adapter = RegimeAdapter(vix_weekly, lookback_weeks=lookback)
    adapter.compute_regime_history()
    
    # Base parameters
    initial_cap = float(params.get("initial_capital", 250_000))
    r = float(params.get("risk_free", params.get("risk_free_rate", 0.03)))
    fee = float(params.get("fee_per_contract", 0.65))
    realism = float(params.get("realism", 1.0))
    base_mode = params.get("mode", "diagonal")
    
    MAX_QTY = 10_000
    
    prices = np.asarray(vix_weekly.values, dtype=float)
    n = len(prices)
    
    if n < 2:
        return {
            "equity": np.asarray([initial_cap], dtype=float),
            "weekly_returns": np.asarray([], dtype=float),
            "realized_weekly": np.asarray([], dtype=float),
            "unrealized_weekly": np.asarray([], dtype=float),
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
            "regime_history": ["MEDIUM"],
            "regime_transitions": [],
            "per_regime_stats": {},
        }
    
    # State tracking
    cash = initial_cap
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]
    
    have_pos = False
    long_pos: Optional[OptionPosition] = None
    short_pos: Optional[OptionPosition] = None
    
    entry_equity: Optional[float] = None
    entry_idx: Optional[int] = None
    entry_price_long: float = 0.0
    
    win_flags: List[bool] = []
    trade_durations: List[int] = []
    trade_log: List[Dict[str, Any]] = []
    
    pos_value_prev = 0.0
    
    # Regime tracking
    regime_history: List[str] = []
    regime_transitions: List[Dict[str, Any]] = []
    per_regime_trades: Dict[str, List[Dict]] = {r: [] for r in REGIME_CONFIGS.keys()}
    per_regime_pnl: Dict[str, float] = {r: 0.0 for r in REGIME_CONFIGS.keys()}
    
    prev_regime = None
    
    # Main loop
    for i in range(n):
        if progress_cb:
            try:
                progress_cb(i, n)
            except Exception:
                pass
        
        S = float(prices[i])
        current_regime = adapter.get_regime_at_index(i)
        regime_history.append(current_regime)
        
        # Track regime transitions
        if prev_regime is not None and current_regime != prev_regime:
            regime_transitions.append({
                "index": i,
                "date": adapter.dates[i] if i < len(adapter.dates) else None,
                "from_regime": prev_regime,
                "to_regime": current_regime,
                "price": S,
            })
        prev_regime = current_regime
        
        if i == 0:
            continue  # Skip first week (no prior history)
        
        # Get regime-specific parameters
        regime_params = _get_regime_params(
            current_regime, base_mode, params, use_optimized_params
        )
        
        # Extract params for this regime
        entry_pct = float(regime_params.get("entry_percentile", 0.10))
        mode = regime_params.get("mode", base_mode)
        alloc_pct_raw = float(regime_params.get("alloc_pct", 0.01))
        if alloc_pct_raw > 1.0:
            alloc_pct = alloc_pct_raw / 100.0
        else:
            alloc_pct = alloc_pct_raw
        
        otm_pts = float(regime_params.get("otm_pts", 10.0))
        sigma_mult = float(regime_params.get("sigma_mult", 1.0))
        long_dte_weeks = int(regime_params.get("long_dte_weeks", 26))
        target_mult = float(regime_params.get("target_mult", 1.20))
        exit_mult = float(regime_params.get("exit_mult", 0.50))
        
        # Advance DTE on existing positions
        if long_pos is not None:
            long_pos.dte_weeks = max(long_pos.dte_weeks - 1, 0)
        if short_pos is not None:
            short_pos.dte_weeks = max(short_pos.dte_weeks - 1, 0)
        
        # Weekly short expiry & roll
        if short_pos is not None and short_pos.dte_weeks == 0:
            intrinsic = max(S - short_pos.strike, 0.0) * 100.0 * abs(short_pos.quantity)
            cash -= intrinsic * realism
            cash -= fee * abs(short_pos.quantity)
            
            if long_pos is not None and mode == "diagonal":
                strike_short = S + otm_pts
                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                if not np.isfinite(sp) or sp < 0.0:
                    sp = 0.0
                cash += sp * 100.0 * abs(short_pos.quantity) * realism
                cash -= fee * abs(short_pos.quantity)
                short_pos = OptionPosition(
                    -abs(short_pos.quantity), strike_short, 1, False,
                    entry_regime=current_regime
                )
            else:
                short_pos = None
        
        # Mark positions to market
        pos_value = 0.0
        long_val = 0.0
        short_val = 0.0
        
        if long_pos is not None:
            T_long = long_pos.dte_weeks / 52.0
            base_sigma = 0.20 * sigma_mult
            sigma_eff = min(max(base_sigma, 0.10), 2.0)
            lv = bs_call_price(S, long_pos.strike, r, sigma_eff, T_long)
            long_val = max(lv, 0.0) * 100.0 * abs(long_pos.quantity)
            pos_value += long_val
        
        if short_pos is not None:
            T_short = short_pos.dte_weeks / 52.0
            short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
            sv = bs_call_price(S, short_pos.strike, r, short_sigma, T_short)
            short_val = max(sv, 0.0) * 100.0 * short_pos.quantity
            pos_value += short_val
        
        eq_prev = equity[-1]
        unreal_pnl = pos_value - pos_value_prev
        pos_value_prev = pos_value
        
        # Exit logic
        exit_trigger = False
        if long_pos is not None:
            if long_pos.dte_weeks <= 0:
                exit_trigger = True
            
            entry_notional = entry_price_long * abs(long_pos.quantity) * 100.0
            if entry_notional > 0:
                if long_val >= entry_notional * target_mult:
                    exit_trigger = True
                if long_val <= entry_notional * exit_mult:
                    exit_trigger = True
        
        if exit_trigger and long_pos is not None:
            cash += long_val * realism
            
            if short_pos is not None:
                cash += short_val * realism
                short_pos = None
            
            dur = i - (entry_idx if entry_idx is not None else i)
            eq_after_close = cash
            
            base_equity = entry_equity if entry_equity is not None else initial_cap
            is_win = eq_after_close > base_equity
            win_flags.append(is_win)
            trade_durations.append(dur)
            
            trade_pnl = eq_after_close - base_equity
            entry_regime = long_pos.entry_regime
            
            trade_rec = {
                "entry_idx": entry_idx,
                "exit_idx": i,
                "duration_weeks": dur,
                "entry_equity": entry_equity,
                "exit_equity": eq_after_close,
                "entry_price_long": entry_price_long,
                "strike_long": long_pos.strike,
                "entry_regime": entry_regime,
                "exit_regime": current_regime,
                "pnl": trade_pnl,
                "is_win": is_win,
            }
            trade_log.append(trade_rec)
            
            # Track per-regime stats
            per_regime_trades[entry_regime].append(trade_rec)
            per_regime_pnl[entry_regime] += trade_pnl
            
            long_pos = None
            have_pos = False
            pos_value_prev = 0.0
            pos_value = 0.0
        
        # Entry logic (regime-aware)
        current_pct = adapter.get_percentile_at_index(i)
        
        if (not have_pos) and np.isfinite(current_pct) and current_pct <= entry_pct:
            capital = cash * alloc_pct
            
            if capital > 0.0:
                strike_long = S + otm_pts
                base_sigma = 0.20 * sigma_mult
                sigma_eff = min(max(base_sigma, 0.10), 2.0)
                
                lp = bs_call_price(S, strike_long, r, sigma_eff, long_dte_weeks / 52.0)
                
                if lp is not None and np.isfinite(lp) and lp > 0.0:
                    denom = lp * 100.0
                    qty_float = capital / denom
                    
                    if np.isfinite(qty_float) and qty_float > 0.0:
                        qty = int(min(max(qty_float, 0.0), MAX_QTY))
                        
                        if qty > 0:
                            cost_long = qty * lp * 100.0 + fee * qty
                            cash -= cost_long
                            
                            short_pos_local: Optional[OptionPosition] = None
                            if mode == "diagonal":
                                strike_short = S + otm_pts
                                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                                if not np.isfinite(sp) or sp < 0.0:
                                    sp = 0.0
                                credit_short = sp * 100.0 * qty
                                cash += credit_short
                                cash -= fee * qty
                                short_pos_local = OptionPosition(
                                    -qty, strike_short, 1, False,
                                    entry_regime=current_regime
                                )
                            
                            long_pos = OptionPosition(
                                qty, strike_long, long_dte_weeks, True,
                                entry_regime=current_regime
                            )
                            short_pos = short_pos_local
                            have_pos = True
                            entry_equity = cash
                            entry_idx = i
                            entry_price_long = lp
                            pos_value_prev = 0.0
                            pos_value = 0.0
        
        # End-of-week bookkeeping
        eq_end = cash + pos_value
        equity.append(eq_end)
        realized_weekly.append(eq_end - eq_prev - unreal_pnl)
        unrealized_weekly.append(unreal_pnl)
        weekly_returns.append((eq_end - eq_prev) / eq_prev if eq_prev > 0 else 0.0)
    
    # Compute per-regime statistics
    per_regime_stats = {}
    for regime_name, trades in per_regime_trades.items():
        if trades:
            wins = sum(1 for t in trades if t.get("is_win", False))
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            avg_dur = np.mean([t.get("duration_weeks", 0) for t in trades])
            
            per_regime_stats[regime_name] = {
                "trades": len(trades),
                "wins": wins,
                "win_rate": wins / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
                "avg_duration": avg_dur,
            }
        else:
            per_regime_stats[regime_name] = {
                "trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_duration": 0.0,
            }
    
    # Final arrays
    equity_arr = _to_1d_float_array(equity)
    weekly_arr = _to_1d_float_array(weekly_returns)
    realized_arr = _to_1d_float_array(realized_weekly)
    unreal_arr = _to_1d_float_array(unrealized_weekly)
    
    trades_count = len(win_flags)
    win_rate = float(np.mean(win_flags)) if trades_count > 0 else 0.0
    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0
    
    return {
        "equity": equity_arr,
        "weekly_returns": weekly_arr,
        "realized_weekly": realized_arr,
        "unrealized_weekly": unreal_arr,
        "trades": trades_count,
        "win_rate": win_rate,
        "avg_trade_dur": avg_dur,
        "trade_log": trade_log,
        "regime_history": regime_history,
        "regime_transitions": regime_transitions,
        "per_regime_stats": per_regime_stats,
    }


# ============================================================
# Comparison Helper
# ============================================================

def compare_adaptive_vs_static(
    vix_weekly: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run both adaptive and static backtests for comparison.
    
    Returns dict with:
        - adaptive_results: results from adaptive backtest
        - static_results: results from standard backtest
        - comparison: dict with delta metrics
    """
    from core.backtester import run_backtest
    
    # Run adaptive
    adaptive = run_adaptive_backtest(vix_weekly, params, use_optimized_params=True)
    
    # Run static
    static = run_backtest(vix_weekly, params)
    
    # Compare
    def _cagr(eq):
        if len(eq) < 2 or eq[0] <= 0:
            return 0.0
        years = (len(eq) - 1) / 52.0
        return (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    
    def _maxdd(eq):
        if len(eq) == 0:
            return 0.0
        cummax = np.maximum.accumulate(eq)
        return float(((eq - cummax) / cummax).min())
    
    adaptive_cagr = _cagr(adaptive["equity"])
    static_cagr = _cagr(static["equity"])
    
    adaptive_dd = _maxdd(adaptive["equity"])
    static_dd = _maxdd(static["equity"])
    
    comparison = {
        "adaptive_cagr": adaptive_cagr,
        "static_cagr": static_cagr,
        "cagr_delta": adaptive_cagr - static_cagr,
        "adaptive_maxdd": adaptive_dd,
        "static_maxdd": static_dd,
        "maxdd_delta": adaptive_dd - static_dd,  # less negative = better
        "adaptive_trades": adaptive["trades"],
        "static_trades": static["trades"],
        "adaptive_win_rate": adaptive["win_rate"],
        "static_win_rate": static["win_rate"],
    }
    
    return {
        "adaptive_results": adaptive,
        "static_results": static,
        "comparison": comparison,
    }
