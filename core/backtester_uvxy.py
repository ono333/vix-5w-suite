#!/usr/bin/env python3
"""
UVXY-optimised synthetic backtester (Black-Scholes).

Designed for decaying vol ETPs like UVXY / VXX:

    - Long call only (no short weekly leg for now)
    - DTE is capped at <= 12 weeks (3 months)
    - Max holding period is short (default 8 weeks)
    - Higher baseline volatility for option pricing
    - Realism haircut scales trade PnL

Interface is intentionally similar to the main synthetic VIX engine:

    run_backtest_uvxy(uvxy_weekly: pd.Series, params: Dict[str, Any]) -> Dict

Return dict:
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

Only ONE position at a time is allowed.  No pyramiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------
# Black–Scholes call pricing
# ---------------------------------------------------------------------


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vanilla Black–Scholes call.

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
    entry_cost: float  # total $ paid incl. fee
    entry_week_idx: int


# ---------------------------------------------------------------------
# Core UVXY backtest
# ---------------------------------------------------------------------


def run_backtest_uvxy(uvxy_weekly: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    UVXY-optimised synthetic backtest (long calls only).

    uvxy_weekly : weekly UVXY close series (Date index).
    params      : same schema as synthetic VIX engine where possible.

    Important params (with sensible UVXY defaults):

        initial_capital : float, starting equity (e.g. 250_000)
        alloc_pct       : float, fraction of equity per trade
                          (1.0  => 100%,  0.10 => 10%,  0.01 => 1%)
        entry_percentile: float, 0-1, enter when UVXY is
                          at or below this price percentile
        entry_lookback_weeks : int, lookback window for percentile
        otm_pts         : float, strike = spot + otm_pts
        sigma_mult      : float, scales baseline volatility
        long_dte_weeks  : int, BEFORE clamp (2..12).  Default 8.
        max_hold_weeks  : int, max weeks to hold (default 8)
        target_mult     : float, take profit when option value
                          >= entry_cost * target_mult
        exit_mult       : float, stop when option value
                          <= entry_cost * exit_mult
        fee_per_contract: float, commission per contract
        risk_free       : float, annual risk-free rate
        realism         : float, 0.5–1.0, scales trade PnL only
    """

    # ----------------- parameters -----------------
    initial_cap = float(params.get("initial_capital", 250_000.0))

    # User may type 1.0 for 100% or 0.01 for 1% or 50 for 50%.
    alloc_raw = float(params.get("alloc_pct", 0.10))  # sensible default 10%
    if alloc_raw > 1.0:
        alloc_pct = alloc_raw / 100.0
    else:
        alloc_pct = alloc_raw
    alloc_pct = max(0.0, min(alloc_pct, 1.0))  # clamp 0–1

    # Mode is kept only for compatibility; diagonal == long_only here.
    mode = params.get("mode", "long_only")

    entry_pct = float(params.get("entry_percentile", 0.10))
    entry_lb = int(params.get("entry_lookback_weeks", 52))

    otm_pts = float(params.get("otm_pts", 2.0))
    target_mult = float(params.get("target_mult", 1.50))
    exit_mult = float(params.get("exit_mult", 0.40))
    sigma_mult = float(params.get("sigma_mult", 1.0))

    # Clamp DTE to [2, 12] weeks (UVXY long calls longer than 3 months are nasty)
    long_dte_weeks_raw = int(params.get("long_dte_weeks", 8))
    long_dte_weeks = max(2, min(long_dte_weeks_raw, 12))

    max_hold_weeks = int(params.get("max_hold_weeks", min(8, long_dte_weeks)))

    fee = float(params.get("fee_per_contract", 0.65))
    r = float(params.get("risk_free", params.get("risk_free_rate", 0.03)))

    # realism haircut: 1.0 = no haircut, 0.5 = half of theoretical edge, etc.
    realism = float(params.get("realism", 1.0))
    realism = max(0.3, min(realism, 1.0))

    prices = uvxy_weekly.values.astype(float)
    dates = uvxy_weekly.index.to_list()
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
        }

    # ----------------- percentile series -----------------
    pct = np.full(n, np.nan, dtype=float)
    lb = max(1, entry_lb)
    for i in range(lb, n):
        window = prices[i - lb : i]
        pct[i] = (window < prices[i]).mean()

    # ----------------- state -----------------
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]

    cash = initial_cap
    have_pos = False
    pos: Optional[OptionPosition] = None
    pos_value_prev = 0.0

    trades = 0
    win_flags: List[bool] = []
    durations: List[int] = []
    trade_log: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # main loop (start from week 1 because week 0 has no prior history)
    # -----------------------------------------------------------------
    for i in range(1, n):
        S = float(prices[i])
        prev_eq = float(equity[-1])

        realized_pnl = 0.0
        unrealized_pnl = 0.0

        # ----------------------------------------------------------
        # 1) Update value of existing position & check exit rules
        # ----------------------------------------------------------
        if have_pos and pos is not None:
            weeks_held = i - pos.entry_week_idx
            # Remaining DTE in weeks for pricing
            remaining_weeks = max(long_dte_weeks - weeks_held, 0)
            T = remaining_weeks / 52.0

            # UVXY baseline vol is high, use 80% * sigma_mult and clamp
            base_sigma = 0.80 * sigma_mult
            sigma_eff = max(0.30, min(base_sigma, 3.00))

            if T <= 0.0:
                # treat as expiration, intrinsic only
                call_value = max(S - pos.strike, 0.0)
            else:
                call_value = bs_call_price(S, pos.strike, r, sigma_eff, T)

            pos_value_now = max(call_value, 0.0) * 100.0 * pos.quantity

            unrealized_pnl = pos_value_now - pos_value_prev
            pos_value_prev = pos_value_now

            # --- Exit logic ---
            exit_trigger = False

            # (a) expiration
            if remaining_weeks <= 0:
                exit_trigger = True

            # (b) profit / loss thresholds
            if pos.entry_cost > 0.0:
                if pos_value_now >= pos.entry_cost * target_mult:
                    exit_trigger = True
                if pos_value_now <= pos.entry_cost * exit_mult:
                    exit_trigger = True

            # (c) hard time stop (UVXY decay guard)
            if weeks_held >= max_hold_weeks:
                exit_trigger = True

            if exit_trigger:
                trades += 1

                trade_pnl_raw = pos_value_now - pos.entry_cost
                trade_pnl = trade_pnl_raw * realism

                # Recover original entry cash (entry_cost) + haircut PnL.
                # At this point cash is still: cash0 - entry_cost
                cash = cash + pos.entry_cost + trade_pnl

                realized_pnl = trade_pnl
                unrealized_pnl = 0.0  # flat after exit
                pos_value_prev = 0.0

                win_flags.append(trade_pnl_raw > 0.0)
                durations.append(weeks_held)

                trade_log.append(
                    {
                        "entry_idx": pos.entry_week_idx,
                        "exit_idx": i,
                        "entry_date": dates[pos.entry_week_idx],
                        "exit_date": dates[i],
                        "entry_cost": pos.entry_cost,
                        "exit_value": pos_value_now,
                        "pnl_raw": trade_pnl_raw,
                        "pnl_after_haircut": trade_pnl,
                        "duration_weeks": weeks_held,
                        "strike_long": pos.strike,
                        "mode": mode,
                    }
                )

                have_pos = False
                pos = None

        # ----------------------------------------------------------
        # 2) Entry condition (only if flat AFTER any exit)
        # ----------------------------------------------------------
        if (not have_pos) and np.isfinite(pct[i]) and pct[i] <= entry_pct:
            # capital based on TOTAL equity, not just cash
            current_equity = cash + pos_value_prev
            capital = current_equity * alloc_pct

            if capital > 0.0:
                strike_long = S + otm_pts
                T0 = long_dte_weeks / 52.0

                base_sigma = 0.80 * sigma_mult
                sigma_eff = max(0.30, min(base_sigma, 3.00))

                call_px = bs_call_price(S, strike_long, r, sigma_eff, T0)
                if call_px > 0.0 and np.isfinite(call_px):
                    denom = call_px * 100.0
                    if denom > 0.0 and np.isfinite(denom):
                        qty_float = capital / denom
                        if np.isfinite(qty_float) and qty_float >= 1.0:
                            qty = int(min(qty_float, 10_000))
                            entry_cost = qty * call_px * 100.0 + fee * qty

                            if entry_cost <= cash:
                                # Open position: pay cost from cash, book value as entry_cost
                                cash -= entry_cost
                                pos = OptionPosition(
                                    quantity=qty,
                                    strike=strike_long,
                                    entry_cost=entry_cost,
                                    entry_week_idx=i,
                                )
                                have_pos = True
                                pos_value_prev = entry_cost

                                # Entry is just a swap: equity unchanged, no PnL booked
                                # realized_pnl / unrealized_pnl remain whatever they are
                            # else: not enough cash -> skip entry

        # ----------------------------------------------------------
        # 3) Final equity for this week
        # ----------------------------------------------------------
        eq_now = cash + pos_value_prev
        equity.append(eq_now)
        realized_weekly.append(realized_pnl)
        unrealized_weekly.append(unrealized_pnl)

        if prev_eq != 0.0:
            weekly_returns.append((eq_now - prev_eq) / prev_eq)
        else:
            weekly_returns.append(0.0)

    equity_arr = np.asarray(equity, dtype=float)
    weekly_arr = np.asarray(weekly_returns, dtype=float)
    realized_arr = np.asarray(realized_weekly, dtype=float)
    unreal_arr = np.asarray(unrealized_weekly, dtype=float)

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
    }