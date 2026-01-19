#!/usr/bin/env python3
"""
Unified VIX 5-Weekly Backtest Engine  (realism-focused version)

Position structures
-------------------
    mode == "diagonal":
        - Long LEAP call (OTM, long_dte_weeks)
        - Short weekly OTM call rolled each week

    mode == "long_only":
        - Long LEAP call only (no short leg)

Key realism features
--------------------
- Uses a minimum / maximum effective volatility so option prices are not
  microscopic or absurd.
- Allocation is interpreted sensibly whether you type 1, 0.01, 50, or 0.50:
    * alloc_pct > 1.0  -> treated as a percent (e.g. 50 -> 0.50)
- Contract quantity is capped, and any NaN/inf premium automatically
  skips the trade instead of blowing up the account.
- Per-contract fees and a "realism haircut" multiply all cashflows.

Return value
------------
A dict:
    {
        "equity": np.ndarray,
        "weekly_returns": np.ndarray,
        "realized_weekly": np.ndarray,
        "unrealized_weekly": np.ndarray,
        "trades": int,
        "win_rate": float,
        "avg_trade_dur": float,
        "trade_log": list[dict],
    }
"""

from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------
# Blackâ€“Scholes call pricing
# ---------------------------------------------------------------------


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vanilla Blackâ€“Scholes call.

    Returns 0 on any bad input instead of raising.
    """
    try:
        if S <= 0.0 or K <= 0.0:
            return 0.0

        if T <= 0.0:
            # At expiration -> intrinsic
            return max(S - K, 0.0)

        # If sigma is too small, approximate as almost intrinsic
        if sigma <= 0.0:
            return max(S - K * exp(-r * T), 0.0)

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class OptionPosition:
    quantity: int
    strike: float
    dte_weeks: int
    is_long: bool  # True for long calls, False for short calls


# ---------------------------------------------------------------------
# Helper: make sure series are 1-D float arrays
# ---------------------------------------------------------------------


def _to_1d_float_array(seq, name: str = "series") -> np.ndarray:
    """
    Converts arbitrary sequences (lists, numpy arrays, small dicts) into
    a clean 1-D float numpy array, ignoring non-numeric garbage.
    """
    clean: List[float] = []

    for x in seq:
        # Already numeric
        if isinstance(x, (int, float, np.number)):
            clean.append(float(x))
            continue

        # 1-element containers
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            try:
                clean.append(float(x[0]))
            except Exception:
                pass
            continue

        # Small dicts with a likely numeric field
        if isinstance(x, dict):
            for key in ("equity", "value", "pnl", "y"):
                if key in x:
                    try:
                        clean.append(float(x[key]))
                        break
                    except Exception:
                        pass

    return np.asarray(clean, dtype=float)


# ---------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------


def run_backtest(vix_weekly: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main unified backtest.

    vix_weekly : weekly VIX close series (Date index).
    params     : dict from sidebar / config.
    """

    # ----------------- parameters -----------------
    initial_cap = float(params.get("initial_capital", 250_000))

    # User may type 1.0 for 100% or 0.01 for 1% or 50 for 50%.
    alloc_raw = float(params.get("alloc_pct", 0.01))
    if alloc_raw > 1.0:
        alloc_pct = alloc_raw / 100.0
    else:
        alloc_pct = alloc_raw

    mode = params.get("mode", "diagonal")

    entry_pct = float(params.get("entry_percentile", 0.10))
    entry_lb = int(params.get("entry_lookback_weeks", 52))

    otm_pts = float(params.get("otm_pts", 10.0))
    target_mult = float(params.get("target_mult", 1.20))
    exit_mult = float(params.get("exit_mult", 0.50))
    sigma_mult = float(params.get("sigma_mult", 1.0))
    long_dte_weeks_default = int(params.get("long_dte_weeks", 26))

    fee = float(params.get("fee_per_contract", 0.65))
    r = float(params.get("risk_free", params.get("risk_free_rate", 0.03)))

    # realism haircut: 1.0 = no haircut, 0.5 = half of theoretical edge, etc.
    realism = float(params.get("realism", 1.0))

    # Safety caps
    MAX_QTY = 10_000  # absolute cap on contracts so grid scans can't explode

    prices = np.asarray(vix_weekly.values, dtype=float)
    n = len(prices)
    if n < 2:
        zero = np.asarray([initial_cap], dtype=float)
        empty = np.asarray([], dtype=float)
        return {
            "equity": zero,
            "weekly_returns": empty,
            "realized_weekly": empty,
            "unrealized_weekly": empty,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
        }

    # ----------------- percentile series -----------------
    pct = np.full(n, np.nan, dtype=float)
    lb = max(1, entry_lb)
    for i in range(lb, n):
        window = prices[i - lb : i]
        pct[i] = (window < prices[i]).mean()

    # ----------------- state -----------------
    cash = initial_cap  # cash in account
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

    # track last week's position value for unrealized PnL
    pos_value_prev = 0.0

    # -----------------------------------------------------------------
    # main loop (start from week 1 because week 0 has no prior history)
    # -----------------------------------------------------------------
    for i in range(1, n):
        S = float(prices[i])

        # --- advance DTE on existing positions ---
        if long_pos is not None:
            long_pos.dte_weeks = max(long_pos.dte_weeks - 1, 0)
        if short_pos is not None:
            short_pos.dte_weeks = max(short_pos.dte_weeks - 1, 0)

        # --- weekly short expiry & roll (for diagonal) ---
        if short_pos is not None and short_pos.dte_weeks == 0:
            # pay intrinsic on short
            intrinsic = max(S - short_pos.strike, 0.0) * 100.0 * abs(short_pos.quantity)
            cash -= intrinsic * realism
            cash -= fee * abs(short_pos.quantity)

            # roll into new 1-week short only if we still have a long
            if long_pos is not None and mode == "diagonal":
                strike_short = S + otm_pts
                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                if not np.isfinite(sp) or sp < 0.0:
                    sp = 0.0
                cash += sp * 100.0 * abs(short_pos.quantity) * realism
                cash -= fee * abs(short_pos.quantity)
                short_pos = OptionPosition(-abs(short_pos.quantity), strike_short, 1, False)
            else:
                short_pos = None

        # --- mark current positions to market ---
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
            short_val = max(sv, 0.0) * 100.0 * short_pos.quantity  # quantity negative
            pos_value += short_val

        eq_prev = equity[-1]
        unreal_pnl = pos_value - pos_value_prev
        pos_value_prev = pos_value

        # --- exit logic for the long leg (target / stop / expiry) ---
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
            # close long at current value
            cash += long_val * realism

            # buy back remaining short at model value
            if short_pos is not None:
                cash += short_val * realism
                short_pos = None

            dur = i - (entry_idx if entry_idx is not None else i)
            eq_after_close = cash  # flat, no positions

            base_equity = entry_equity if entry_equity is not None else initial_cap
            win_flags.append(eq_after_close > base_equity)
            trade_durations.append(dur)
            trade_log.append(
                dict(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    duration_weeks=dur,
                    entry_equity=entry_equity,
                    exit_equity=eq_after_close,
                    entry_price_long=entry_price_long,
                    strike_long=long_pos.strike,
                )
            )

            long_pos = None
            have_pos = False
            pos_value_prev = 0.0
            pos_value = 0.0

        # --- entry logic (can re-enter after exit in same week) ---
        if (not have_pos) and np.isfinite(pct[i]) and pct[i] <= entry_pct:
            capital = cash * alloc_pct
            if capital > 0.0:
                strike_long = S + otm_pts
                base_sigma = 0.20 * sigma_mult
                sigma_eff = min(max(base_sigma, 0.10), 2.0)

                lp = bs_call_price(S, strike_long, r, sigma_eff, long_dte_weeks_default / 52.0)
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
                                short_pos_local = OptionPosition(-qty, strike_short, 1, False)

                            long_pos = OptionPosition(qty, strike_long, long_dte_weeks_default, True)
                            short_pos = short_pos_local
                            have_pos = True
                            entry_equity = cash
                            entry_idx = i
                            entry_price_long = lp
                            pos_value_prev = 0.0
                            pos_value = 0.0

        # --- end-of-week bookkeeping ---
        eq_end = cash + pos_value
        equity.append(eq_end)
        realized_weekly.append(eq_end - eq_prev - unreal_pnl)
        unrealized_weekly.append(unreal_pnl)
        weekly_returns.append((eq_end - eq_prev) / eq_prev if eq_prev > 0 else 0.0)

    # -----------------------------------------------------------------
    # Final stats and sanitisation
    # -----------------------------------------------------------------
    equity_arr = _to_1d_float_array(equity, "equity")
    weekly_arr = _to_1d_float_array(weekly_returns, "weekly_returns")
    realized_arr = _to_1d_float_array(realized_weekly, "realized_weekly")
    unreal_arr = _to_1d_float_array(unrealized_weekly, "unrealized_weekly")

    trades = len(win_flags)
    win_rate = float(np.mean(win_flags)) if trades > 0 else 0.0
    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0

    return {
        "equity": equity_arr,
        "weekly_returns": weekly_arr,
        "realized_weekly": realized_arr,
        "unrealized_weekly": unreal_arr,
        "trades": trades,
        "win_rate": win_rate,
        "avg_trade_dur": avg_dur,
        "trade_log": trade_log,
    }