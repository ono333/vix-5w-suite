#!/usr/bin/env python3
import io
import datetime as dt
from datetime import datetime
from math import erf, sqrt
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# Core math helpers
# =========================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Simple Black-Scholes call pricer for VIX options."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def _calc_cagr(total_return: float, n_weeks: int) -> float:
    if n_weeks <= 0:
        return np.nan
    if total_return <= -0.9999:
        return np.nan
    return (1.0 + total_return) ** (52.0 / n_weeks) - 1.0


# =========================
# Parsing helpers for grid
# =========================

def parse_float_list(text: str, default_vals):
    """Parse comma-separated floats; fallback to default_vals on error/empty."""
    try:
        parts = [p.strip() for p in text.split(",")]
        vals = [float(p) for p in parts if p != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


def parse_int_list(text: str, default_vals):
    """Parse comma-separated ints; fallback to default_vals on error/empty."""
    try:
        parts = [p.strip() for p in text.split(",")]
        vals = [int(p) for p in parts if p != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


# =========================
# Meta-rule extractor (for diagonal mode)
# =========================

def score_row(row: pd.Series, dd_penalty: float = 2.0) -> float:
    """Score combining CAGR and MaxDD (higher better)."""
    return row["cagr"] - dd_penalty * abs(row["max_dd"])


def extract_meta_rules(
    grid_df: pd.DataFrame, dd_penalty: float = 2.0, top_n: int = 5
) -> pd.DataFrame:
    """
    From grid_df, compute meta-rules per entry_pct.

    Returns DataFrame:
    [entry_pct, long_dte_weeks, otm_pts, target_mult, sigma_mult, avg_cagr, avg_max_dd, avg_trades]
    """
    if grid_df is None or grid_df.empty:
        return pd.DataFrame()

    rows = []
    for ep, group in grid_df.groupby("entry_pct"):
        g = group.copy()
        if g.empty:
            continue
        g["score"] = g.apply(score_row, axis=1, dd_penalty=dd_penalty)
        g_sorted = g.sort_values("score", ascending=False).head(top_n)

        row = {
            "entry_pct": float(ep),
            "long_dte_weeks": int(round(g_sorted["long_dte_weeks"].mean())),
            "otm_pts": float(g_sorted["otm_pts"].median()),
            "target_mult": float(g_sorted["target_mult"].median()),
            "sigma_mult": float(g_sorted["sigma_mult"].median()),
            "avg_cagr": float(g_sorted["cagr"].mean()),
            "avg_max_dd": float(g_sorted["max_dd"].mean()),
            "avg_trades": float(g_sorted["trades"].mean()),
        }
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    if not meta_df.empty:
        meta_df = meta_df.sort_values("entry_pct").reset_index(drop=True)
    return meta_df


# =========================
# Dynamic schedules (used by diagonal dynamic mode)
# =========================

def param_schedules_default(pctl: float) -> dict:
    """
    Fallback mapping from VIX percentile (0–1) to parameters.
    Used if no meta-rules exist yet.
    """

    # Long DTE: long in calm, short in high vol
    long_dte_weeks = int(
        np.interp(pctl, [0.0, 0.5, 1.0], [52, 13, 3])
    )

    # OTM distance: closer in calm, further out in high vol
    otm_points = float(
        np.interp(pctl, [0.0, 1.0], [3.0, 7.0])
    )

    # Target multiple: big TP in calm, tiny TP in high vol
    target_multiple = float(
        np.interp(pctl, [0.0, 1.0], [2.0, 1.15])
    )

    # Sigma multiplier: more conservative in calm, closer to “sweet spot” in high vol
    sigma_mult = float(
        np.interp(pctl, [0.0, 1.0], [1.0, 0.8])
    )

    return {
        "long_dte_weeks": long_dte_weeks,
        "otm_points": otm_points,
        "target_multiple": target_multiple,
        "sigma_mult": sigma_mult,
    }


def param_schedules_from_meta(pctl: float, meta_df: Optional[pd.DataFrame]) -> dict:
    """
    Map VIX percentile -> params using meta_rules_df.
    If meta_df is None/empty, fall back to param_schedules_default.
    """
    if meta_df is None or meta_df.empty:
        return param_schedules_default(pctl)

    xs = meta_df["entry_pct"].values
    p = float(np.clip(pctl, xs.min(), xs.max()))

    params = {}
    # Name in meta_df, whether result should be int
    for col, as_int in [
        ("long_dte_weeks", True),
        ("otm_pts", False),
        ("target_mult", False),
        ("sigma_mult", False),
    ]:
        ys = meta_df[col].values
        val = float(np.interp(p, xs, ys))
        if as_int:
            val = int(round(val))
        # Map otm_pts -> otm_points, rest keep same
        key = "otm_points" if col == "otm_pts" else col
        params[key] = val

    return params


# =========================
# Core backtest: DIAGONAL (LEAP + weekly short OTM calls)
# =========================

def backtest_vix_diagonal(
    vix_weekly: pd.Series,
    vix_percentile_series: pd.Series,
    strategy_mode: str = "Static (fixed params)",
    initial_capital: float = 250_000.0,
    alloc_pct: float = 0.01,
    long_dte_weeks: int = 26,
    entry_lookback_weeks: int = 52,
    entry_percentile: float = 0.10,
    otm_points: float = 3.0,
    target_multiple: float = 2.0,
    sigma_mult: float = 1.0,
    r: float = 0.03,
    fee_per_contract: float = 0.65,
    meta_rules_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Diagonal strategy:
    - Long-dated ATM call (LEAP-style)
    - Weekly short OTM calls overlay
    """
    prices = np.asarray(vix_weekly, dtype=float).reshape(-1)
    dates = vix_weekly.index
    n = len(prices)
    if n < 5:
        return {
            "equity": np.array([initial_capital]),
            "weekly_returns": np.array([0.0]),
            "total_return": 0.0,
            "cagr": np.nan,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "dates": dates,
            "weekly_pnl_pct": np.array([0.0]),
            "realized_pnl_cum": np.array([0.0]),
            "realized_pnl_weekly": np.array([0.0]),
            "unrealized_pnl_weekly": np.array([0.0]),
            "trade_count": 0,
            "avg_trade_duration": np.nan,
            "min_trade_duration": np.nan,
            "max_trade_duration": np.nan,
        }

    equity = np.zeros(n, dtype=float)
    weekly_ret = np.zeros(n, dtype=float)
    weekly_pnl_pct = np.zeros(n, dtype=float)

    realized_pnl_cum = np.zeros(n, dtype=float)
    realized_pnl_weekly = np.zeros(n, dtype=float)
    unrealized_pnl_weekly = np.zeros(n, dtype=float)

    cash = initial_capital
    equity[0] = initial_capital
    long_value = 0.0
    has_long = False
    long_ttm_weeks = 0
    long_entry_price = 0.0
    long_contracts = 0
    long_cost_basis = 0.0

    trade_count = 0
    current_trade_open_idx = None
    durations = []

    # Per-trade active parameters
    long_dte_weeks_active = long_dte_weeks
    otm_points_active = otm_points
    target_multiple_active = target_multiple
    sigma_mult_active = sigma_mult

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]

        # current VIX percentile
        try:
            current_pct = float(vix_percentile_series.iloc[i])
        except Exception:
            current_pct = np.nan

        # Equity at start of week i
        equity[i] = cash + long_value
        realized_this_week = 0.0

        # ----- ENTRY LOGIC -----
        if not has_long:
            look_start = max(0, i - entry_lookback_weeks + 1)
            look_prices = prices[look_start : i + 1]

            allow_entry = False
            dyn_params = {
                "long_dte_weeks": long_dte_weeks,
                "otm_points": otm_points,
                "target_multiple": target_multiple,
                "sigma_mult": sigma_mult,
            }

            if strategy_mode == "Static (fixed params)":
                if len(look_prices) >= 4:
                    threshold = np.quantile(look_prices, entry_percentile)
                    if S <= threshold and S > 0:
                        allow_entry = True

            elif strategy_mode == "Static 90%-only":
                if not np.isnan(current_pct) and current_pct >= 0.90 and S > 0:
                    allow_entry = True

            elif strategy_mode == "Dynamic (percentile-based)":
                if not np.isnan(current_pct) and S > 0:
                    allow_entry = True
                    dyn_params = param_schedules_from_meta(current_pct, meta_rules_df)

            if allow_entry and S > 0:
                long_dte_weeks_active = int(dyn_params["long_dte_weeks"])
                otm_points_active = float(dyn_params["otm_points"])
                target_multiple_active = float(dyn_params["target_multiple"])
                sigma_mult_active = float(dyn_params["sigma_mult"])

                T_long = long_dte_weeks_active / 52.0
                sigma_long = max((S / 100.0) * sigma_mult_active, 0.01)
                price_long = black_scholes_call(S, S, T_long, r, sigma_long)

                if price_long > 0:
                    equity_now = cash + long_value
                    max_risk = equity_now * alloc_pct

                    per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
                    contracts = int(max_risk / per_contract_cost)

                    if contracts > 0:
                        has_long = True
                        long_ttm_weeks = long_dte_weeks_active
                        long_entry_price = price_long
                        long_contracts = contracts

                        open_fee_total = fee_per_contract * contracts
                        cash -= price_long * 100.0 * contracts
                        cash -= open_fee_total
                        long_value = price_long * 100.0 * contracts
                        long_cost_basis = long_value + open_fee_total

                        trade_count += 1
                        current_trade_open_idx = i

        # ----- ONGOING POSITION -----
        if has_long and long_contracts > 0:
            # Weekly short call overlay
            T_short = 1.0 / 52.0
            K_short = S + otm_points_active
            sigma_short = max((S / 100.0) * sigma_mult_active, 0.01)

            price_short = black_scholes_call(S, K_short, T_short, r, sigma_short)
            if price_short > 0:
                short_premium = price_short * 100.0 * long_contracts
                open_short_fee = fee_per_contract * long_contracts

                cash += short_premium
                cash -= open_short_fee

                payoff_short = max(S_next - K_short, 0.0) * 100.0 * long_contracts
                close_short_fee = fee_per_contract * long_contracts
                cash -= payoff_short
                cash -= close_short_fee

                realized_short = short_premium - payoff_short - open_short_fee - close_short_fee
                realized_this_week += realized_short

            # Re-price long at next week
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            T_new = long_ttm_weeks / 52.0
            sigma_long_new = max((S_next / 100.0) * sigma_mult_active, 0.01)
            price_long_new = black_scholes_call(S_next, S, T_new, r, sigma_long_new)
            long_value_new = price_long_new * 100.0 * long_contracts if T_new > 0 else 0.0

            # Exit condition
            exit_now = False
            if T_new <= 0:
                exit_now = True
            elif price_long_new >= target_multiple_active * long_entry_price:
                exit_now = True

            if exit_now:
                close_long_fee = fee_per_contract * long_contracts
                cash += long_value_new
                cash -= close_long_fee

                realized_long = long_value_new - long_cost_basis - close_long_fee
                realized_this_week += realized_long

                if current_trade_open_idx is not None:
                    dur_weeks = (i + 1) - current_trade_open_idx
                    if dur_weeks > 0:
                        durations.append(dur_weeks)

                long_value_new = 0.0
                long_cost_basis = 0.0
                has_long = False
                long_contracts = 0
                current_trade_open_idx = None

            long_value = long_value_new

        # ----- Weekly bookkeeping -----
        equity[i + 1] = cash + long_value
        eq_change = equity[i + 1] - equity[i]

        if equity[i] > 0:
            weekly_ret[i + 1] = eq_change / equity[i]
            weekly_pnl_pct[i + 1] = weekly_ret[i + 1] * 100.0
        else:
            weekly_ret[i + 1] = 0.0
            weekly_pnl_pct[i + 1] = 0.0

        realized_pnl_weekly[i + 1] = realized_this_week
        realized_pnl_cum[i + 1] = realized_pnl_cum[i] + realized_this_week
        unrealized_pnl_weekly[i + 1] = eq_change - realized_this_week

        if equity[i + 1] <= 0:
            equity[i + 1] = 0.0
            weekly_ret[i + 1 :] = -1.0
            weekly_pnl_pct[i + 1 :] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_weeks_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = _calc_cagr(total_return, n_weeks_valid)

    valid_returns = weekly_ret[1 : n_weeks_valid + 1]
    if len(valid_returns) > 1 and np.std(valid_returns) > 1e-8:
        sharpe = (np.mean(valid_returns) * 52.0 - r) / (np.std(valid_returns) * np.sqrt(52.0))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    win_rate = float(np.mean(valid_returns > 0.0)) if len(valid_returns) else 0.0

    if len(durations) > 0:
        avg_dur = float(np.mean(durations))
        min_dur = float(np.min(durations))
        max_dur = float(np.max(durations))
    else:
        avg_dur = np.nan
        min_dur = np.nan
        max_dur = np.nan

    return {
        "equity": equity,
        "weekly_returns": weekly_ret,
        "total_return": float(total_return),
        "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "dates": dates,
        "weekly_pnl_pct": weekly_pnl_pct,
        "realized_pnl_cum": realized_pnl_cum,
        "realized_pnl_weekly": realized_pnl_weekly,
        "unrealized_pnl_weekly": unrealized_pnl_weekly,
        "trade_count": int(trade_count),
        "avg_trade_duration": avg_dur,
        "min_trade_duration": min_dur,
        "max_trade_duration": max_dur,
    }


# =========================
# Core backtest: LONG-ONLY VIX CALL
# =========================

def backtest_vix_long_only(
    vix_weekly: pd.Series,
    vix_percentile_series: pd.Series,
    strategy_mode: str = "Static (fixed params)",
    exit_mode: str = "TP only",
    stop_mult: float = 0.5,
    exit_pct_threshold: float = 0.5,
    initial_capital: float = 250_000.0,
    alloc_pct: float = 0.01,
    long_dte_weeks: int = 3,
    entry_lookback_weeks: int = 52,
    entry_percentile: float = 0.10,
    target_multiple: float = 1.15,
    sigma_mult: float = 0.5,
    r: float = 0.03,
    fee_per_contract: float = 0.65,
) -> dict:
    """
    Long-only VIX call strategy:
    - Buy ATM VIX call when entry conditions satisfied
    - No weekly short calls
    - Exit logic controlled by exit_mode
        - "TP only": price >= target_multiple * entry_price OR T=0
        - "TP + stop": TP only + price <= stop_mult * entry_price
        - "Percentile exit": TP only + VIX percentile <= exit_pct_threshold
    """
    prices = np.asarray(vix_weekly, dtype=float).reshape(-1)
    dates = vix_weekly.index
    n = len(prices)
    if n < 5:
        return {
            "equity": np.array([initial_capital]),
            "weekly_returns": np.array([0.0]),
            "total_return": 0.0,
            "cagr": np.nan,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "dates": dates,
            "weekly_pnl_pct": np.array([0.0]),
            "realized_pnl_cum": np.array([0.0]),
            "realized_pnl_weekly": np.array([0.0]),
            "unrealized_pnl_weekly": np.array([0.0]),
            "trade_count": 0,
            "avg_trade_duration": np.nan,
            "min_trade_duration": np.nan,
            "max_trade_duration": np.nan,
        }

    equity = np.zeros(n, dtype=float)
    weekly_ret = np.zeros(n, dtype=float)
    weekly_pnl_pct = np.zeros(n, dtype=float)

    realized_pnl_cum = np.zeros(n, dtype=float)
    realized_pnl_weekly = np.zeros(n, dtype=float)
    unrealized_pnl_weekly = np.zeros(n, dtype=float)

    cash = initial_capital
    equity[0] = initial_capital
    long_value = 0.0
    has_long = False
    long_ttm_weeks = 0
    long_entry_price = 0.0
    long_contracts = 0
    long_cost_basis = 0.0

    trade_count = 0
    current_trade_open_idx = None
    durations = []

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]

        # current VIX percentile
        try:
            current_pct = float(vix_percentile_series.iloc[i])
        except Exception:
            current_pct = np.nan

        equity[i] = cash + long_value
        realized_this_week = 0.0

        # ----- ENTRY LOGIC -----
        if not has_long and S > 0:
            look_start = max(0, i - entry_lookback_weeks + 1)
            look_prices = prices[look_start : i + 1]

            allow_entry = False
            if strategy_mode == "Static (fixed params)":
                if len(look_prices) >= 4:
                    threshold = np.quantile(look_prices, entry_percentile)
                    if S <= threshold:
                        allow_entry = True
            elif strategy_mode == "Static 90%-only":
                if not np.isnan(current_pct) and current_pct >= 0.90:
                    allow_entry = True
            elif strategy_mode == "Dynamic (percentile-based)":
                # For now: simple rule — enter whenever percentile is defined
                if not np.isnan(current_pct):
                    allow_entry = True

            if allow_entry:
                long_ttm_weeks = long_dte_weeks
                T_long = long_ttm_weeks / 52.0
                sigma_long = max((S / 100.0) * sigma_mult, 0.01)
                price_long = black_scholes_call(S, S, T_long, r, sigma_long)

                if price_long > 0:
                    equity_now = cash + long_value
                    max_risk = equity_now * alloc_pct
                    per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
                    contracts = int(max_risk / per_contract_cost)
                    if contracts > 0:
                        has_long = True
                        long_entry_price = price_long
                        long_contracts = contracts

                        open_fee_total = fee_per_contract * contracts
                        cash -= price_long * 100.0 * contracts
                        cash -= open_fee_total
                        long_value = price_long * 100.0 * contracts
                        long_cost_basis = long_value + open_fee_total

                        trade_count += 1
                        current_trade_open_idx = i

        # ----- ONGOING POSITION -----
        if has_long and long_contracts > 0:
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            T_new = long_ttm_weeks / 52.0
            sigma_long_new = max((S_next / 100.0) * sigma_mult, 0.01)
            price_long_new = black_scholes_call(S_next, S, T_new, r, sigma_long_new)
            long_value_new = price_long_new * 100.0 * long_contracts if T_new > 0 else 0.0

            # Exit conditions
            exit_now = False

            # TP-only always applies
            if price_long_new >= target_multiple * long_entry_price:
                exit_now = True

            # Time up
            if T_new <= 0:
                exit_now = True

            # Extra: stop loss
            if exit_mode == "TP + stop":
                if price_long_new <= stop_mult * long_entry_price:
                    exit_now = True

            # Extra: percentile exit
            if exit_mode == "Percentile exit" and not np.isnan(current_pct):
                if current_pct <= exit_pct_threshold:
                    exit_now = True

            if exit_now:
                close_fee_total = fee_per_contract * long_contracts
                cash += long_value_new
                cash -= close_fee_total

                realized_long = long_value_new - long_cost_basis - close_fee_total
                realized_this_week += realized_long

                if current_trade_open_idx is not None:
                    dur_weeks = (i + 1) - current_trade_open_idx
                    if dur_weeks > 0:
                        durations.append(dur_weeks)

                long_value_new = 0.0
                long_cost_basis = 0.0
                has_long = False
                long_contracts = 0
                current_trade_open_idx = None

            long_value = long_value_new

        # ----- Weekly bookkeeping -----
        equity[i + 1] = cash + long_value
        eq_change = equity[i + 1] - equity[i]

        if equity[i] > 0:
            weekly_ret[i + 1] = eq_change / equity[i]
            weekly_pnl_pct[i + 1] = weekly_ret[i + 1] * 100.0
        else:
            weekly_ret[i + 1] = 0.0
            weekly_pnl_pct[i + 1] = 0.0

        realized_pnl_weekly[i + 1] = realized_this_week
        realized_pnl_cum[i + 1] = realized_pnl_cum[i] + realized_this_week
        unrealized_pnl_weekly[i + 1] = eq_change - realized_this_week

        if equity[i + 1] <= 0:
            equity[i + 1] = 0.0
            weekly_ret[i + 1 :] = -1.0
            weekly_pnl_pct[i + 1 :] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_weeks_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = _calc_cagr(total_return, n_weeks_valid)

    valid_returns = weekly_ret[1 : n_weeks_valid + 1]
    if len(valid_returns) > 1 and np.std(valid_returns) > 1e-8:
        sharpe = (np.mean(valid_returns) * 52.0 - r) / (np.std(valid_returns) * np.sqrt(52.0))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    win_rate = float(np.mean(valid_returns > 0.0)) if len(valid_returns) else 0.0

    if len(durations) > 0:
        avg_dur = float(np.mean(durations))
        min_dur = float(np.min(durations))
        max_dur = float(np.max(durations))
    else:
        avg_dur = np.nan
        min_dur = np.nan
        max_dur = np.nan

    return {
        "equity": equity,
        "weekly_returns": weekly_ret,
        "total_return": float(total_return),
        "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "dates": dates,
        "weekly_pnl_pct": weekly_pnl_pct,
        "realized_pnl_cum": realized_pnl_cum,
        "realized_pnl_weekly": realized_pnl_weekly,
        "unrealized_pnl_weekly": unrealized_pnl_weekly,
        "trade_count": int(trade_count),
        "avg_trade_duration": avg_dur,
        "min_trade_duration": min_dur,
        "max_trade_duration": max_dur,
    }


# =========================
# Grid scan (DIAGONAL only)
# =========================

def run_grid_scan_diagonal(
    vix_weekly: pd.Series,
    vix_percentile_series: pd.Series,
    initial_capital: float,
    alloc_pct: float,
    r: float,
    fee_per_contract: float,
    entry_grid,
    otm_grid,
    target_grid,
    sigma_grid,
    dte_grid,
    entry_lookback_weeks: int = 52,
    strategy_mode_for_scan: str = "Static (fixed params)",
):
    rows = []
    for dte in dte_grid:
        for ep in entry_grid:
            for otm in otm_grid:
                for tgt in target_grid:
                    for sig in sigma_grid:
                        res = backtest_vix_diagonal(
                            vix_weekly=vix_weekly,
                            vix_percentile_series=vix_percentile_series,
                            strategy_mode=strategy_mode_for_scan,
                            initial_capital=initial_capital,
                            alloc_pct=alloc_pct,
                            long_dte_weeks=int(dte),
                            entry_lookback_weeks=entry_lookback_weeks,
                            entry_percentile=ep,
                            otm_points=otm,
                            target_multiple=tgt,
                            sigma_mult=sig,
                            r=r,
                            fee_per_contract=fee_per_contract,
                            meta_rules_df=None,
                        )
                        rows.append(
                            {
                                "entry_pct": ep,
                                "otm_pts": otm,
                                "target_mult": tgt,
                                "sigma_mult": sig,
                                "long_dte_weeks": int(dte),
                                "total_return": res["total_return"],
                                "cagr": res["cagr"],
                                "sharpe": res["sharpe"],
                                "max_dd": res["max_dd"],
                                "win_rate": res["win_rate"],
                                "trades": res.get("trade_count", 0),
                                "avg_dur": res.get("avg_trade_duration", np.nan),
                                "min_dur": res.get("min_trade_duration", np.nan),
                                "max_dur": res.get("max_trade_duration", np.nan),
                            }
                        )

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return grid_df, None

    sorted_df = grid_df.sort_values(["cagr", "max_dd"], ascending=[False, True])
    best_row = sorted_df.iloc[0]
    return sorted_df, best_row


# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="VIX 5% Weekly Strategy Backtester", layout="wide")

st.title("VIX 5% Weekly Strategy: Long-Term Backtester")
st.write(
    "Simulate:\n"
    "- **Diagonal**: VIX 5% Weekly Diagonal (LEAP + Weekly Covered Calls)\n"
    "- **Long-only**: ATM VIX Calls without weekly shorts\n"
    "on historical VIX data with dynamic and static modes."
)

# ----- Structure + strategy mode -----

structure_mode = st.radio(
    "Position structure",
    [
        "Diagonal: LEAP + Weekly OTM Calls",
        "Long-only: VIX Call (no weekly shorts)",
    ],
    index=0,
)

strategy_mode = st.selectbox(
    "Entry / regime mode",
    ["Static (fixed params)", "Static 90%-only", "Dynamic (percentile-based)"],
    index=0,
)

# ----- Session-state init for defaults -----

param_keys = [
    "initial_capital",
    "alloc_pct",
    "long_dte_weeks",
    "risk_free",
    "fee_per_contract",
    "entry_percentile",
    "otm_points",
    "target_multiple",
    "sigma_mult",
]

param_defaults = {
    "initial_capital": 250_000.0,
    "alloc_pct": 0.01,
    "long_dte_weeks": 26,
    "risk_free": 0.03,
    "fee_per_contract": 0.65,
    "entry_percentile": 0.10,
    "otm_points": 3.0,
    "target_multiple": 2.0,
    "sigma_mult": 1.0,
}

for k in param_keys:
    if k not in st.session_state:
        st.session_state[k] = param_defaults[k]
    dkey = f"default_{k}"
    if dkey not in st.session_state:
        st.session_state[dkey] = st.session_state[k]

if "grid_results_df" not in st.session_state:
    st.session_state.grid_results_df = None
if "grid_xlsx_bytes" not in st.session_state:
    st.session_state.grid_xlsx_bytes = None
if "meta_rules_df" not in st.session_state:
    st.session_state.meta_rules_df = None
if "last_grid_idx" not in st.session_state:
    st.session_state.last_grid_idx = 0

# ----- Sidebar: dates -----

with st.sidebar:
    st.header("Backtest Settings")

    min_date = dt.date(2004, 1, 1)
    max_date = dt.date.today()

    start_date = st.date_input(
        "Start date",
        value=dt.date(2015, 1, 1),
        min_value=min_date,
        max_value=max_date,
    )

    end_date = st.date_input(
        "End date",
        value=max_date,
        min_value=start_date,
        max_value=max_date,
    )

# ----- Sidebar: capital & params -----

with st.sidebar:
    st.caption("Capital & risk")

    cap_str_default = f'{st.session_state["initial_capital"]:,.0f}'
    cap_str = st.text_input(
        "Initial capital ($)",
        value=cap_str_default,
        key="initial_capital_str",
    )
    try:
        initial_capital = float(cap_str.replace(",", "").strip())
    except ValueError:
        initial_capital = st.session_state["initial_capital"]
    st.session_state["initial_capital"] = initial_capital

    alloc_pct_percent = st.number_input(
        "Fraction of equity allocated to long call (%)",
        min_value=0.1,
        max_value=50.0,
        value=float(st.session_state["alloc_pct"] * 100.0),
        step=0.1,
        format="%.1f",
        key="alloc_pct_percent",
    )
    alloc_pct = alloc_pct_percent / 100.0
    st.session_state["alloc_pct"] = alloc_pct

    long_dte_weeks = st.slider(
        "Long call maturity (weeks) [baseline]",
        1,
        104,
        value=int(st.session_state["long_dte_weeks"]),
        step=1,
        key="long_dte_weeks",
    )

    risk_free = st.slider(
        "Risk-free rate (annual)",
        0.0,
        0.10,
        value=float(st.session_state["risk_free"]),
        step=0.005,
        key="risk_free",
    )

    fee_per_contract = st.number_input(
        "Fee per options contract ($)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["fee_per_contract"]),
        step=0.05,
        format="%.2f",
        key="fee_per_contract",
    )

    entry_percentile = st.slider(
        "Entry percentile (Static mode only)",
        0.0,
        0.30,
        value=float(st.session_state["entry_percentile"]),
        step=0.01,
        key="entry_percentile",
    )

    otm_points = st.slider(
        "Short call OTM distance (pts) [diagonal baseline]",
        1.0,
        20.0,
        value=float(st.session_state["otm_points"]),
        step=0.5,
        key="otm_points",
    )

    target_multiple = st.slider(
        "Exit when long call multiplies by [baseline]",
        1.1,
        5.0,
        value=float(st.session_state["target_multiple"]),
        step=0.05,
        key="target_multiple",
    )

    sigma_mult = st.slider(
        "Volatility multiplier (sigma_mult) baseline",
        0.1,
        3.0,
        value=float(st.session_state["sigma_mult"]),
        step=0.1,
        key="sigma_mult",
    )

    # Extra options for Long-only exit
    if "Long-only" in structure_mode:
        st.markdown("---")
        st.caption("Long-only exit controls")
        exit_mode = st.selectbox(
            "Long-only exit method",
            ["TP only", "TP + stop", "Percentile exit"],
            index=0,
            key="exit_mode_long_only",
        )
        stop_mult = st.slider(
            "Stop-loss multiplier (only for TP + stop)",
            0.1,
            0.9,
            value=0.5,
            step=0.05,
            key="stop_mult_long_only",
        )
        exit_pct_threshold = st.slider(
            "Percentile exit threshold (only for Percentile exit)",
            0.0,
            1.0,
            value=0.5,
            step=0.05,
            key="exit_pct_long_only",
        )
    else:
        exit_mode = "TP only"
        stop_mult = 0.5
        exit_pct_threshold = 0.5

    # Save / reset defaults
    if st.button("Save current as default"):
        for k in param_keys:
            st.session_state[f"default_{k}"] = st.session_state[k]

    if st.button("Reset to saved default"):
        for k in param_keys:
            st.session_state[k] = st.session_state[f"default_{k}"]
        st.experimental_rerun()

    # Grid ranges (diagonal only)
    grid_exp = st.expander("Grid scan: parameter ranges (Diagonal only, incl. DTE)", expanded=False)
    with grid_exp:
        entry_grid_str = st.text_input(
            "Entry percentiles (decimals, e.g. 0.10, 0.30, 0.50, 0.90)",
            value=st.session_state.get("grid_entry_list", "0.10, 0.30, 0.50, 0.70, 0.90"),
            key="grid_entry_list",
        )
        otm_grid_str = st.text_input(
            "OTM distances (points, e.g. 3, 5, 7, 10, 14)",
            value=st.session_state.get("grid_otm_list", "3, 5, 7, 10, 14"),
            key="grid_otm_list",
        )
        target_grid_str = st.text_input(
            "Target multiples (e.g. 1.10, 1.15, 1.20)",
            value=st.session_state.get("grid_target_list", "1.10, 1.15, 1.20"),
            key="grid_target_list",
        )
        sigma_grid_str = st.text_input(
            "Sigma multipliers (e.g. 0.3, 0.5, 0.8, 1.0)",
            value=st.session_state.get("grid_sigma_list", "0.3, 0.5, 0.8, 1.0"),
            key="grid_sigma_list",
        )
        dte_grid_str = st.text_input(
            "Long DTE weeks (e.g. 3, 4, 6, 13, 26)",
            value=st.session_state.get("grid_dte_list", "3, 4, 6, 13, 26"),
            key="grid_dte_list",
        )

# Parse grid ranges
entry_grid_vals = parse_float_list(
    st.session_state.get("grid_entry_list", "0.10, 0.30"), [0.10, 0.30]
)
otm_grid_vals = parse_float_list(
    st.session_state.get("grid_otm_list", "3, 5, 7"), [3.0, 5.0, 7.0]
)
target_grid_vals = parse_float_list(
    st.session_state.get("grid_target_list", "1.10, 1.25"), [1.10, 1.25]
)
sigma_grid_vals = parse_float_list(
    st.session_state.get("grid_sigma_list", "0.8, 1.0"), [0.8, 1.0]
)
dte_grid_vals = parse_int_list(
    st.session_state.get("grid_dte_list", "3, 13, 26"), [3, 13, 26]
)

entry_lookback_weeks = 52

# ----- Load VIX -----

@st.cache_data(show_spinner=False)
def load_vix_weekly(start: datetime, end: datetime) -> pd.Series:
    data = yf.download("^VIX", start=start, end=end)
    if data.empty:
        return pd.Series(dtype=float)
    weekly = data["Close"].resample("W-FRI").last().dropna()
    return weekly

vix_weekly = load_vix_weekly(start_date, end_date)
if vix_weekly.empty:
    st.error("No VIX data was downloaded. Check date range or internet.")
    st.stop()

n_weeks = len(vix_weekly)
st.success(
    f"Loaded {n_weeks} weeks of VIX data "
    f"({vix_weekly.index[0].date()} to {vix_weekly.index[-1].date()})."
)

# ----- VIX percentile -----

vix_vals = np.asarray(vix_weekly, dtype=float).reshape(-1)
vix_pct = np.full(len(vix_vals), np.nan, dtype=float)
for i in range(len(vix_vals)):
    if i >= entry_lookback_weeks - 1:
        window = vix_vals[i - entry_lookback_weeks + 1 : i + 1]
        current_vix = vix_vals[i]
        vix_pct[i] = np.mean(window <= current_vix)
vix_percentile_series = pd.Series(vix_pct, index=vix_weekly.index, name="VIX_52w_percentile")

# ----- Main backtest (structure-dependent) -----

meta_rules_df = st.session_state.get("meta_rules_df", None)

if "Diagonal" in structure_mode:
    results = backtest_vix_diagonal(
        vix_weekly=vix_weekly,
        vix_percentile_series=vix_percentile_series,
        strategy_mode=strategy_mode,
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
        long_dte_weeks=long_dte_weeks,
        entry_lookback_weeks=entry_lookback_weeks,
        entry_percentile=entry_percentile,
        otm_points=otm_points,
        target_multiple=target_multiple,
        sigma_mult=sigma_mult,
        r=risk_free,
        fee_per_contract=fee_per_contract,
        meta_rules_df=meta_rules_df,
    )
else:
    results = backtest_vix_long_only(
        vix_weekly=vix_weekly,
        vix_percentile_series=vix_percentile_series,
        strategy_mode=strategy_mode,
        exit_mode=exit_mode,
        stop_mult=stop_mult,
        exit_pct_threshold=exit_pct_threshold,
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
        long_dte_weeks=long_dte_weeks,
        entry_lookback_weeks=entry_lookback_weeks,
        entry_percentile=entry_percentile,
        target_multiple=target_multiple,
        sigma_mult=sigma_mult,
        r=risk_free,
        fee_per_contract=fee_per_contract,
    )

equity = np.asarray(results["equity"], dtype=float).reshape(-1)
weekly_returns = np.asarray(results["weekly_returns"], dtype=float).reshape(-1)
total_return = results["total_return"]
cagr = results["cagr"]
sharpe = results["sharpe"]
max_dd = results["max_dd"]
win_rate = results["win_rate"]
dates = results["dates"]
weekly_pnl_pct = results["weekly_pnl_pct"]
realized_pnl_cum = results["realized_pnl_cum"]
realized_pnl_weekly = results["realized_pnl_weekly"]
unrealized_pnl_weekly = results["unrealized_pnl_weekly"]
trade_count = results.get("trade_count", 0)
avg_trade_duration = results.get("avg_trade_duration", np.nan)
min_trade_duration = results.get("min_trade_duration", np.nan)
max_trade_duration = results.get("max_trade_duration", np.nan)

dates_1d = np.asarray(dates)
vix_values_weekly = np.asarray(vix_weekly.reindex(dates_1d), dtype=float).reshape(-1)
weekly_pnl_1d = np.asarray(weekly_pnl_pct, dtype=float).reshape(-1)
vix_pct_on_dates = np.asarray(vix_percentile_series.reindex(dates_1d), dtype=float).reshape(-1)

ending_equity = float(equity[-1])
total_return_dollar = ending_equity - initial_capital

current_vix_pct = float(vix_percentile_series.iloc[-1]) if not np.isnan(vix_percentile_series.iloc[-1]) else np.nan

# ----- Execution mode toggle (for checklists / suggestions) -----

execution_mode = st.radio(
    "Execution mode for today's checklist",
    [
        "Entry mode (no long VIX call open)",
        "Management mode (long VIX call already open)",
    ],
    index=0,
)

# ----- Strategy description -----

strategy_expander = st.expander("Strategy Implementation (with your current settings)", expanded=True)
with strategy_expander:
    meta_flag = "meta-rules from last diagonal grid scan" if meta_rules_df is not None and not meta_rules_df.empty else "built-in default schedule"
    struct_label = "Diagonal (LEAP + Weekly Covered Calls)" if "Diagonal" in structure_mode else "Long-only VIX Call"

    extra_exit_text = ""
    if "Long-only" in structure_mode:
        extra_exit_text = f"\n- Long-only exit mode: **{exit_mode}**"

    st.markdown(
        f"""
**Structure:** {struct_label}  
**Entry mode:** `{strategy_mode}`  
{extra_exit_text}

- Weekly VIX data from **{start_date}** to **{end_date}**, 52-week rolling percentile.
- Static: enter when VIX is below **{entry_percentile*100:.1f}%** of last {entry_lookback_weeks} weeks.
- Static 90%-only: enter only if VIX percentile ≥ 90%.
- Dynamic (diagonal): parameters follow a schedule based on current VIX percentile, using **{meta_flag}**.

**Trades:** {trade_count}  
**Avg trade duration:** {avg_trade_duration:.2f} weeks  
**Min / Max duration:** {min_trade_duration:.0f} – {max_trade_duration:.0f} weeks
"""
    )

# ----- Quick Weekly Playbook (simpler checklist) -----

playbook_simple = st.expander("Quick Weekly Playbook (Summary)", expanded=False)
with playbook_simple:
    pct_str = f"{current_vix_pct*100:.1f}%" if not np.isnan(current_vix_pct) else "N/A"

    if "Entry mode" in execution_mode:
        if "Diagonal" in structure_mode:
            st.markdown(
                f"""
**You selected:** `Entry mode` + `Diagonal structure`

1. **Once per week (your chosen execution day):**
   - Check **VIX** on broker.
   - App shows latest 52w percentile ≈ **{pct_str}**.

2. **Entry condition (mode: `{strategy_mode}`):**
   - **Static:** enter if VIX ≤ {entry_percentile*100:.1f}% percentile.
   - **Static 90%-only:** enter only if percentile ≥ 90%.
   - **Dynamic:** enter whenever percentile is defined; DTE/OTM/TP come from meta-rules.

3. **When entry condition is met:**
   - Portfolio equity ≈ **${ending_equity:,.0f}**.
   - Allocation = **{alloc_pct*100:.2f}%** → risk ≈ **${ending_equity*alloc_pct:,.0f}**.
   - Choose a VIX **ATM call** with ≈ **{long_dte_weeks} weeks** to expiry.
   - Buy as many contracts as fit inside that risk.

4. **From next week onward (while long is open):**
   - Each week, sell a **1-week OTM VIX call** per long contract, ≈ (spot + {otm_points:.1f} pts).
   - Let the short call decay, or manage if VIX spikes through strike.

5. **Exit long:**
   - When long call price ≥ **{target_multiple:.2f} × entry price**, or
   - Time-to-expiry is very small.

6. **After exit:**
   - Record PnL and go back to **Entry mode**.
"""
            )
        else:
            st.markdown(
                f"""
**You selected:** `Entry mode` + `Long-only structure`

1. **Once per week (your chosen execution day):**
   - Check **VIX**.
   - App shows latest 52w percentile ≈ **{pct_str}**.

2. **Entry condition (mode: `{strategy_mode}`):**
   - Same percentile logic as diagonal (no weekly short overlay).

3. **When entry condition is met:**
   - Portfolio equity ≈ **${ending_equity:,.0f}**.
   - Allocation = **{alloc_pct*100:.2f}%** → risk ≈ **${ending_equity*alloc_pct:,.0f}**.
   - Buy ATM **VIX call** with ≈ **{long_dte_weeks} weeks** to expiry.

4. **While long is open:**
   - No weekly shorts; you just hold the call.

5. **Exit long (method: `{exit_mode}`):**
   - Always: exit if price ≥ **{target_multiple:.2f} × entry price** or at expiry.
   - If **TP + stop**: also exit if price ≤ **{stop_mult:.2f} × entry price**.
   - If **Percentile exit**: also exit if VIX percentile ≤ **{exit_pct_threshold*100:.1f}%**.

6. **After exit:**
   - Record PnL and return to Entry mode.
"""
            )
    else:
        if "Diagonal" in structure_mode:
            st.markdown(
                f"""
**You selected:** `Management mode` + `Diagonal structure`

1. **This week: manage existing diagonal**

2. **Weekly short call overlay:**
   - Sell ~1-week OTM VIX call per long contract:
     - Strike ≈ spot + {otm_points:.1f} pts.

3. **Monitor:**
   - Let short decay or close early if most premium is captured or VIX spikes.

4. **Check long exit:**
   - Exit if long price ≥ **{target_multiple:.2f} × entry price** or very near expiry.
"""
            )
        else:
            st.markdown(
                f"""
**You selected:** `Management mode` + `Long-only structure`

1. **You already hold a VIX call.**

2. **Each week:**
   - Check whether exit condition for `{exit_mode}` is triggered:
     - TP: price ≥ **{target_multiple:.2f} × entry**.
     - Stop: price ≤ **{stop_mult:.2f} × entry** (if TP + stop).
     - Percentile: VIX percentile ≤ **{exit_pct_threshold*100:.1f}%** (if Percentile exit).

3. **If exit is triggered:**
   - Close the long call.
   - Next week switch back to Entry mode.
"""
            )

# ----- Trade suggestion section -----

trade_suggestion_expander = st.expander("Today's Trade Suggestion (Manual Execution Template)", expanded=False)
with trade_suggestion_expander:
    try:
        last_val = vix_weekly.iloc[-1]
        spot = float(getattr(last_val, "item", lambda: last_val)())
        T_long = long_dte_weeks / 52.0
        sigma_long_guess = max((spot / 100.0) * sigma_mult, 0.01)

        if spot > 0 and T_long > 0:
            d1 = (np.log(spot / spot) + (risk_free + 0.5 * sigma_long_guess**2) * T_long) / (sigma_long_guess * np.sqrt(T_long))
            delta_long = norm_cdf(d1)
            price_long = black_scholes_call(spot, spot, T_long, risk_free, sigma_long_guess)
            per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
            risk_dollars = ending_equity * alloc_pct
            contracts_suggested = int(risk_dollars // per_contract_cost) if per_contract_cost > 0 else 0

            short_strike = spot + otm_points

            if "Entry mode" in execution_mode:
                if "Diagonal" in structure_mode:
                    st.markdown(
                        f"""
**Structure:** Diagonal — new entry suggestion (if conditions met)**

- **Underlying:** VIX options (on VIX futures)  
- **Current VIX (spot):** {spot:.2f}  

**Long leg (LEAP-like):**

- Buy **{contracts_suggested} × VIX call**  
- Expiry ≈ **{long_dte_weeks} weeks** out  
- Strike ≈ **ATM ~ {spot:.0f}**  
- Model premium ≈ **${price_long*100.0:,.2f}** per contract  
- Delta ≈ **{delta_long:.2f}**

**Risk & size:**

- Equity ≈ **${ending_equity:,.0f}**  
- Allocation = **{alloc_pct*100:.2f}%** → risk ≈ **${risk_dollars:,.0f}**  
- Per-contract cost ≈ **${per_contract_cost:,.2f}**  
- Suggested size ≈ **{contracts_suggested}** contracts

**Weekly short overlay (from next week):**

- Sell **1 × VIX call** per long contract  
- 1-week expiry  
- Strike ≈ **spot + {otm_points:.1f} ≈ {short_strike:.1f}**

**Profit target:**

- Exit long when price ≥ **{target_multiple:.2f} × entry**.
"""
                    )
                else:
                    st.markdown(
                        f"""
**Structure:** Long-only — new entry suggestion (if conditions met)**

- **Underlying:** VIX options (on VIX futures)  
- **Current VIX (spot):** {spot:.2f}  

**Long leg:**

- Buy **{contracts_suggested} × VIX call**  
- Expiry ≈ **{long_dte_weeks} weeks** out  
- Strike ≈ **ATM ~ {spot:.0f}**  
- Model premium ≈ **${price_long*100.0:,.2f}** per contract  
- Delta ≈ **{delta_long:.2f}**

**Exit method:** `{exit_mode}`

- Always: TP at **{target_multiple:.2f} × entry** or at expiry.
- If TP + stop: exit if price ≤ **{stop_mult:.2f} × entry**.
- If Percentile exit: exit if percentile ≤ **{exit_pct_threshold*100:.1f}%**.
"""
                    )
            else:
                st.info("Management mode: use the playbook above to adjust or exit existing positions.")
        else:
            st.info("Cannot compute a trade suggestion (spot or T_long invalid).")
    except Exception as e:
        st.warning(f"Could not compute suggestion: {e}")

# ----- Performance metrics -----

st.subheader("Performance Metrics")
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
col1.metric("Total Return (%)", f"{total_return*100:.1f}%")
col2.metric("Total Return ($)", f"{total_return_dollar:,.0f}")
col3.metric("Ending Equity ($)", f"{ending_equity:,.0f}")
col4.metric("CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "N/A")
col5.metric("Sharpe (ann.)", f"{sharpe:.2f}")
col6.metric("Max Drawdown", f"{max_dd*100:.1f}%")
col7.metric("Win Rate", f"{win_rate*100:.1f}%")
col8.metric("Trades", f"{trade_count}")
col9.metric("Avg Trade Dur (weeks)", f"{avg_trade_duration:.2f}" if not np.isnan(avg_trade_duration) else "N/A")

# ----- Charts -----

st.subheader("Equity Curve with VIX & Weekly P&L + VIX Percentile")

plt.style.use("dark_background")

fig1, ax1 = plt.subplots()
ax1.plot(dates, equity, label="Equity ($)", linewidth=1.0)
ax1.set_ylabel("Equity ($)")
ax2 = ax1.twinx()
ax2.plot(dates, vix_values_weekly, linestyle="--", linewidth=1.0, label="VIX")
ax2.set_ylabel("VIX")
ax1.set_title("Equity Curve and VIX")
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig2, ax2p = plt.subplots()
ax2p.bar(dates, weekly_pnl_pct)
ax2p.set_title("Weekly P&L (%) and VIX 52w Percentile")
ax2p.set_ylabel("P&L (%)")
ax2p.grid(True, alpha=0.3)
ax3 = ax2p.twinx()
ax3.plot(dates, vix_pct_on_dates * 100.0, linewidth=0.8, linestyle="--", color="tab:orange")
ax3.set_ylabel("VIX 52w Percentile (%)")

plt.style.use("default")

col_a, col_b = st.columns(2)
with col_a:
    st.pyplot(fig1, clear_figure=True)
with col_b:
    st.pyplot(fig2, clear_figure=True)

# ----- Weekly table -----

df_weekly = pd.DataFrame(
    {
        "Date": dates_1d,
        "VIX Close": vix_values_weekly,
        "VIX 52w Percentile (%)": vix_pct_on_dates * 100.0,
        "Strategy Weekly P&L (%)": weekly_pnl_1d,
        "Realized PnL this week ($)": realized_pnl_weekly,
        "Unrealized PnL this week ($)": unrealized_pnl_weekly,
        "Cum Realized PnL ($)": realized_pnl_cum,
        "Equity ($)": equity,
    }
)
st.dataframe(df_weekly, use_container_width=True)

# ----- Grid scan & meta-rules & upload (Diagonal only) -----

show_grid = st.checkbox("Show DIAGONAL parameter grid scan, meta-rules & XLSX tools", value=False)

if show_grid:
    st.subheader("Parameter Grid Scan (Diagonal structure only)")

    if "Long-only" in structure_mode:
        st.info("Grid scan & meta-rules are currently implemented only for the Diagonal structure. Switch structure to 'Diagonal' above to use this section.")
    else:
        col_run, col_clear = st.columns(2)
        with col_run:
            run_scan_clicked = st.button("Run / Update grid scan")
        with col_clear:
            clear_scan_clicked = st.button("Clear scan + meta-rules")

        if clear_scan_clicked:
            st.session_state.grid_results_df = None
            st.session_state.grid_xlsx_bytes = None
            st.session_state.meta_rules_df = None
            st.session_state.last_grid_idx = 0

        if run_scan_clicked:
            sorted_df, best_row = run_grid_scan_diagonal(
                vix_weekly=vix_weekly,
                vix_percentile_series=vix_percentile_series,
                initial_capital=initial_capital,
                alloc_pct=alloc_pct,
                r=risk_free,
                fee_per_contract=fee_per_contract,
                entry_grid=entry_grid_vals,
                otm_grid=otm_grid_vals,
                target_grid=target_grid_vals,
                sigma_grid=sigma_grid_vals,
                dte_grid=dte_grid_vals,
                entry_lookback_weeks=entry_lookback_weeks,
                strategy_mode_for_scan="Static (fixed params)",
            )
            st.session_state.grid_results_df = sorted_df
            st.session_state.meta_rules_df = extract_meta_rules(sorted_df)
            st.session_state.last_grid_idx = 0

        grid_df = st.session_state.grid_results_df

        # Upload existing scan
        st.markdown("### Upload existing DIAGONAL scan (XLSX) to reuse meta-rules")
        uploaded_file = st.file_uploader("Upload previous vix_5pct_grid_scan.xlsx", type=["xlsx"])
        if uploaded_file is not None:
            try:
                up_df = pd.read_excel(uploaded_file)
                required_cols = {"entry_pct", "otm_pts", "target_mult", "sigma_mult", "long_dte_weeks"}
                if required_cols.issubset(set(up_df.columns)):
                    st.session_state.grid_results_df = up_df
                    st.session_state.meta_rules_df = extract_meta_rules(up_df)
                    st.success("Uploaded scan loaded. Meta-rules updated for Diagonal Dynamic mode.")
                else:
                    st.error(f"Uploaded file missing required columns: {required_cols - set(up_df.columns)}")
            except Exception as e:
                st.error(f"Failed to read uploaded XLSX: {e}")

        grid_df = st.session_state.grid_results_df
        meta_rules_df = st.session_state.meta_rules_df

        if grid_df is None or grid_df.empty:
            st.info("No grid scan results yet. Run a scan or upload a previous XLSX.")
        else:
            st.markdown(
                "Sorted by **higher CAGR first**, and for ties, **smaller (less negative) Max Drawdown**."
            )
            st.dataframe(grid_df.round(4), use_container_width=True)

            # XLSX export (robust)
            st.markdown("### Export scan results")
            col_gen, col_dl = st.columns(2)
            with col_gen:
                if st.button("Generate XLSX from current grid"):
                    try:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                            grid_df.to_excel(writer, index=False, sheet_name="scan")
                        buffer.seek(0)
                        st.session_state.grid_xlsx_bytes = buffer.read()
                        st.success("XLSX generated and stored in session.")
                    except Exception as e:
                        st.error(f"Failed to generate XLSX: {e}")

            with col_dl:
                if st.session_state.grid_xlsx_bytes:
                    st.download_button(
                        label="Download scan results as XLSX",
                        data=st.session_state.grid_xlsx_bytes,
                        file_name="vix_5pct_grid_scan.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_vix_5pct_grid_scan",
                    )
                else:
                    st.info("Click **Generate XLSX from current grid** first.")

            # Meta-rules
            st.subheader("Meta-rules learned from DIAGONAL grid (per entry_pct)")
            if meta_rules_df is not None and not meta_rules_df.empty:
                st.dataframe(meta_rules_df.round(4), use_container_width=True)
            else:
                st.info("Run a grid scan or upload a scan to extract meta-rules.")

st.markdown("---")
st.caption(
    "Educational use only. Simplified approximation of a VIX 5% Weekly Diagonal (LEAP + Weekly Covered Calls) "
    "and a Long-only VIX call strategy, with static / 90%-only / dynamic modes, grid scan for diagonal, trade durations, "
    "meta-rules, XLSX export, and XLSX upload for reusing previous scans."
)