#!/usr/bin/env python3
"""
Adaptive Backtester with Regime-Based Strategy Switching

This backtester dynamically changes strategy parameters based on VIX regime
DURING the backtest, not just at initialization.

Key Features:
- Regime detection every week
- Parameter adaptation per regime
- Trade execution with proper entry signals
- Comprehensive diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# Black-Scholes Pricing
# ============================================================

def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes call price with safety checks"""
    try:
        if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
            return max(S - K, 0.0)  # Intrinsic value
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    except Exception:
        return max(S - K, 0.0)


# ============================================================
# Position Tracking
# ============================================================

@dataclass
class OptionPosition:
    quantity: int
    strike: float
    dte_weeks: int
    is_long: bool
    entry_value: float  # Track entry value for profit/loss calcs


# ============================================================
# Main Adaptive Backtester
# ============================================================

def run_adaptive_backtest(
    vix_weekly: pd.Series,
    base_params: Dict[str, Any],
    regime_adapter: Any,  # RegimeAdapter instance
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Adaptive backtest that changes strategy based on VIX regime.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Weekly VIX/UVXY closes
    base_params : dict
        Base parameters (will be overridden by regime adapter)
    regime_adapter : RegimeAdapter
        The regime detection and adaptation engine
    verbose : bool
        If True, print diagnostic info during backtest
    
    Returns
    -------
    dict with:
        - equity : np.ndarray
        - weekly_returns : np.ndarray
        - realized_weekly : np.ndarray
        - unrealized_weekly : np.ndarray
        - trades : int
        - win_rate : float
        - avg_trade_dur : float
        - trade_log : list[dict]
        - regime_history : list[dict]
        - entry_signals : list[dict]  # Diagnostic: all entry opportunities
        - no_trade_reasons : list[dict]  # Diagnostic: why trades didn't fire
    """
    
    # Initialize - ensure vix_weekly is 1D
    if isinstance(vix_weekly, pd.DataFrame):
        vix_weekly = vix_weekly.iloc[:, 0]
    vix_series = pd.Series(vix_weekly).astype(float)
    prices = vix_series.values
    n = len(prices)
    
    if n < 2:
        return _empty_backtest_result(base_params)
    
    initial_cap = float(base_params.get("initial_capital", 250_000))
    r = float(base_params.get("risk_free", 0.03))
    fee = float(base_params.get("fee_per_contract", 0.65))
    realism = float(base_params.get("realism", 1.0))
    
    # State tracking
    cash = initial_cap
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]
    
    have_pos = False
    long_pos: Optional[OptionPosition] = None
    short_pos: Optional[OptionPosition] = None
    entry_idx: Optional[int] = None
    pos_value_prev = 0.0
    
    # Trade tracking
    trade_log: List[Dict[str, Any]] = []
    win_flags: List[bool] = []
    durations: List[int] = []
    
    # Diagnostics
    entry_signals: List[Dict[str, Any]] = []
    no_trade_reasons: List[Dict[str, Any]] = []
    
    # -------------------------------------------------------
    # Main loop
    # -------------------------------------------------------
    for i in range(1, n):
        S = float(prices[i])
        prev_eq = float(equity[-1])
        
        # Get regime-adapted parameters for THIS week
        params = regime_adapter.build_adaptive_params(base_params, i)
        
        mode = params["mode"]
        alloc_pct = float(params["alloc_pct"])
        entry_pct_threshold = float(params["entry_percentile"])
        otm_pts = float(params["otm_pts"])
        target_mult = float(params["target_mult"])
        exit_mult = float(params["exit_mult"])
        sigma_mult = float(params["sigma_mult"])
        long_dte_weeks = int(params["long_dte_weeks"])
        
        current_vix_pct = float(regime_adapter.vix_percentile.iloc[i])
        
        # ---------------------------------------------------
        # Advance DTE on existing positions
        # ---------------------------------------------------
        if long_pos is not None:
            long_pos.dte_weeks = max(long_pos.dte_weeks - 1, 0)
        if short_pos is not None:
            short_pos.dte_weeks = max(short_pos.dte_weeks - 1, 0)
        
        # ---------------------------------------------------
        # Roll weekly short (diagonal mode)
        # ---------------------------------------------------
        if short_pos is not None and short_pos.dte_weeks == 0:
            # Pay intrinsic on expiring short
            intrinsic = max(S - short_pos.strike, 0.0) * 100.0 * abs(short_pos.quantity)
            cash -= intrinsic * realism
            cash -= fee * abs(short_pos.quantity)
            
            # Roll into new weekly short if still holding long
            if long_pos is not None and mode == "diagonal":
                strike_short = S + otm_pts
                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                
                if np.isfinite(sp) and sp > 0.0:
                    cash += sp * 100.0 * abs(short_pos.quantity) * realism
                    cash -= fee * abs(short_pos.quantity)
                    short_pos = OptionPosition(
                        quantity=-abs(short_pos.quantity),
                        strike=strike_short,
                        dte_weeks=1,
                        is_long=False,
                        entry_value=sp * 100.0 * abs(short_pos.quantity),
                    )
                else:
                    short_pos = None
            else:
                short_pos = None
        
        # ---------------------------------------------------
        # Mark positions to market
        # ---------------------------------------------------
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
            short_val = max(sv, 0.0) * 100.0 * short_pos.quantity  # negative qty
            pos_value += short_val
        
        unreal_pnl = pos_value - pos_value_prev
        pos_value_prev = pos_value
        
        # ---------------------------------------------------
        # Exit logic
        # ---------------------------------------------------
        exit_trigger = False
        exit_reason = None
        
        if long_pos is not None:
            # Expiration
            if long_pos.dte_weeks <= 0:
                exit_trigger = True
                exit_reason = "expiration"
            
            # Profit target
            if long_pos.entry_value > 0 and long_val >= long_pos.entry_value * target_mult:
                exit_trigger = True
                exit_reason = f"profit_target_{target_mult:.2f}x"
            
            # Stop loss
            if long_pos.entry_value > 0 and long_val <= long_pos.entry_value * exit_mult:
                exit_trigger = True
                exit_reason = f"stop_loss_{exit_mult:.2f}x"
        
        if exit_trigger and long_pos is not None:
            # Save values before clearing position
            entry_value_saved = long_pos.entry_value
            pnl_value = long_val - entry_value_saved
            
            # Close positions
            cash += long_val * realism
            if short_pos is not None:
                cash += short_val * realism
                short_pos = None
            
            dur = i - (entry_idx if entry_idx is not None else i)
            eq_after = cash
            
            win_flag = eq_after > prev_eq
            win_flags.append(win_flag)
            durations.append(dur)
            
            trade_log.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "duration_weeks": dur,
                "entry_vix": float(prices[entry_idx]) if entry_idx is not None else np.nan,
                "exit_vix": S,
                "entry_value": entry_value_saved,
                "exit_value": long_val,
                "pnl": pnl_value,
                "reason": exit_reason,
                "regime": params.get("regime", "unknown"),
            })
            
            long_pos = None
            have_pos = False
            pos_value_prev = 0.0
            
            if verbose:
                print(f"Week {i}: EXIT - {exit_reason}, PnL: {pnl_value:.2f}")
        
        # ---------------------------------------------------
        # Entry logic (only if flat AFTER any exit)
        # ---------------------------------------------------
        if not have_pos:
            # Check if we have a valid percentile
            if not np.isfinite(current_vix_pct):
                no_trade_reasons.append({
                    "week_idx": i,
                    "reason": "no_percentile_data",
                    "vix_level": S,
                })
            elif current_vix_pct > entry_pct_threshold:
                # Not in entry zone
                pass  # Too common to log
            else:
                # ENTRY SIGNAL DETECTED
                entry_signals.append({
                    "week_idx": i,
                    "vix_level": S,
                    "vix_percentile": current_vix_pct,
                    "regime": params.get("regime", "unknown"),
                    "attempted": True,
                })
                
                # Calculate position size
                capital = cash * alloc_pct
                
                if capital <= 0:
                    no_trade_reasons.append({
                        "week_idx": i,
                        "reason": "insufficient_capital",
                        "cash": cash,
                        "required": 0,
                    })
                else:
                    # Price long call
                    strike_long = S + otm_pts
                    base_sigma = 0.20 * sigma_mult
                    sigma_eff = min(max(base_sigma, 0.10), 2.0)
                    
                    lp = bs_call_price(S, strike_long, r, sigma_eff, long_dte_weeks / 52.0)
                    
                    if not np.isfinite(lp) or lp <= 0:
                        no_trade_reasons.append({
                            "week_idx": i,
                            "reason": "invalid_long_price",
                            "price": lp,
                        })
                    else:
                        denom = lp * 100.0
                        qty_float = capital / denom
                        
                        if not np.isfinite(qty_float) or qty_float < 1.0:
                            no_trade_reasons.append({
                                "week_idx": i,
                                "reason": "quantity_too_small",
                                "qty_float": qty_float,
                                "capital": capital,
                                "price": lp,
                            })
                        else:
                            qty = int(min(qty_float, 10_000))
                            cost_long = qty * lp * 100.0 + fee * qty
                            
                            if cost_long > cash:
                                no_trade_reasons.append({
                                    "week_idx": i,
                                    "reason": "insufficient_cash",
                                    "cost": cost_long,
                                    "cash": cash,
                                })
                            else:
                                # EXECUTE TRADE
                                cash -= cost_long
                                
                                long_pos = OptionPosition(
                                    quantity=qty,
                                    strike=strike_long,
                                    dte_weeks=long_dte_weeks,
                                    is_long=True,
                                    entry_value=cost_long,
                                )
                                
                                # Short leg for diagonal
                                if mode == "diagonal":
                                    strike_short = S + otm_pts
                                    short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                                    sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                                    
                                    if np.isfinite(sp) and sp > 0:
                                        credit_short = sp * 100.0 * qty
                                        cash += credit_short
                                        cash -= fee * qty
                                        
                                        short_pos = OptionPosition(
                                            quantity=-qty,
                                            strike=strike_short,
                                            dte_weeks=1,
                                            is_long=False,
                                            entry_value=credit_short,
                                        )
                                    else:
                                        short_pos = None
                                else:
                                    short_pos = None
                                
                                have_pos = True
                                entry_idx = i
                                pos_value_prev = 0.0
                                
                                if verbose:
                                    print(f"Week {i}: ENTRY - VIX={S:.2f}, pct={current_vix_pct:.2%}, "
                                          f"mode={mode}, qty={qty}, strike={strike_long:.2f}")
        
        # ---------------------------------------------------
        # End-of-week equity
        # ---------------------------------------------------
        eq_end = cash + pos_value
        equity.append(eq_end)
        realized_weekly.append(eq_end - prev_eq - unreal_pnl)
        unrealized_weekly.append(unreal_pnl)
        weekly_returns.append((eq_end - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
    
    # -------------------------------------------------------
    # Final results
    # -------------------------------------------------------
    equity_arr = np.asarray(equity, dtype=float)
    weekly_arr = np.asarray(weekly_returns, dtype=float)
    realized_arr = np.asarray(realized_weekly, dtype=float)
    unreal_arr = np.asarray(unrealized_weekly, dtype=float)
    
    trades = len(win_flags)
    win_rate = float(np.mean(win_flags)) if trades > 0 else 0.0
    avg_dur = float(np.mean(durations)) if durations else 0.0
    
    return {
        "equity": equity_arr,
        "weekly_returns": weekly_arr,
        "realized_weekly": realized_arr,
        "unrealized_weekly": unreal_arr,
        "trades": trades,
        "win_rate": win_rate,
        "avg_trade_dur": avg_dur,
        "trade_log": trade_log,
        "regime_history": regime_adapter.regime_history,
        "entry_signals": entry_signals,
        "no_trade_reasons": no_trade_reasons,
    }


def _empty_backtest_result(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return empty result structure"""
    initial_cap = float(params.get("initial_capital", 250_000))
    return {
        "equity": np.array([initial_cap]),
        "weekly_returns": np.array([]),
        "realized_weekly": np.array([]),
        "unrealized_weekly": np.array([]),
        "trades": 0,
        "win_rate": 0.0,
        "avg_trade_dur": 0.0,
        "trade_log": [],
        "regime_history": [],
        "entry_signals": [],
        "no_trade_reasons": [],
    }
