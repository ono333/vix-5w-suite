#!/usr/bin/env python3
"""
VIX 5% Weekly – D3 Engine v12
Diagonal (LEAP + Weekly OTM) & Long-only Backtester
- Realistic VIX option pricing (haircut on BS to mimic futures / contango / IV crush / spreads)
- Single backtest: Diagonal or Long-only
- Grid scans:
    * Diagonal: entry_pct, otm_pts, target_mult, sigma_mult, long_dte_weeks (DTE test!)
    * Long-only: entry_pct, target_mult, sigma_mult, long_dte_weeks, exit_mode
- Regime Engine:
    * Percentile-based parameter schedule for Diagonal
"""

import io
import datetime as dt
from math import erf, sqrt
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# --------------------------------------------------------
# Basic math helpers
# --------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Plain BS call (used as theoretical base for VIX calls)."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def calc_cagr(total_return: float, n_weeks: int) -> float:
    if n_weeks <= 0:
        return np.nan
    if total_return <= -0.9999:
        return np.nan
    return (1.0 + total_return) ** (52.0 / n_weeks) - 1.0


# --------------------------------------------------------
# Realistic VIX option pricing layer
# --------------------------------------------------------

def realism_haircut_factor(pct: float, realism_level: str = "Normal") -> float:
    """
    Haircut factor for option prices to approximate:
    - VIX futures term structure
    - contango bleed
    - IV crush
    - spreads / slippage (aggregated)
    pct: VIX 52w percentile in [0,1]; NaN -> 0.5.
    Return factor in [0.2, 0.9].
    """
    if np.isnan(pct):
        pct = 0.5

    if realism_level == "Optimistic":
        base = 0.75
    elif realism_level == "Conservative":
        base = 0.45
    else:
        base = 0.60

    # In calm (low percentile) we want heavier haircut (more contango bleed).
    calm_boost = np.interp(pct, [0.0, 0.5, 1.0], [1.4, 1.0, 0.7])
    factor = base / calm_boost
    return float(max(0.2, min(factor, 0.9)))


def price_vix_call(
    S_spot: float,
    K: float,
    T: float,
    r: float,
    sigma_base: float,
    vix_pct: float,
    realistic_mode: bool,
    realism_level: str = "Normal",
) -> float:
    """
    Wrapper around Black-Scholes:
    - If realistic_mode=False → pure BS on spot VIX.
    - If realistic_mode=True  → apply haircut factor to mimic
      futures-based VIX option pricing and frictions.
    """
    bs_price = black_scholes_call(S_spot, K, T, r, sigma_base)
    if not realistic_mode or T <= 0:
        return bs_price

    factor = realism_haircut_factor(vix_pct, realism_level)
    return bs_price * factor


# --------------------------------------------------------
# Percentile-based Diagonal schedule (Regime Engine)
# --------------------------------------------------------

def default_regime_schedule() -> pd.DataFrame:
    """
    Default diagonal meta-schedule as a function of VIX percentile bucket.
    Buckets are defined by 'p_min' and 'p_max', in [0,1].
    """
    rows = [
        # Calm
        {"p_min": 0.00, "p_max": 0.40, "long_dte_weeks": 40, "otm_pts": 7.0,
         "target_mult": 1.30, "sigma_mult": 1.2},
        # Neutral
        {"p_min": 0.40, "p_max": 0.70, "long_dte_weeks": 26, "otm_pts": 5.0,
         "target_mult": 1.20, "sigma_mult": 1.0},
        # Elevated
        {"p_min": 0.70, "p_max": 0.90, "long_dte_weeks": 20, "otm_pts": 4.0,
         "target_mult": 1.15, "sigma_mult": 0.8},
        # High
        {"p_min": 0.90, "p_max": 0.97, "long_dte_weeks": 13, "otm_pts": 3.0,
         "target_mult": 1.10, "sigma_mult": 0.6},
        # Extreme
        {"p_min": 0.97, "p_max": 1.00, "long_dte_weeks": 8, "otm_pts": 2.0,
         "target_mult": 1.05, "sigma_mult": 0.5},
    ]
    return pd.DataFrame(rows)


def params_from_regime(pct: float, regime_df: pd.DataFrame) -> Dict[str, float]:
    """
    Given current VIX percentile in [0,1], pick diag params from regime_df.
    If pct lies between rows, we pick the first row whose [p_min,p_max] contains pct.
    """
    if regime_df is None or regime_df.empty:
        regime_df = default_regime_schedule()

    if np.isnan(pct):
        pct = 0.5

    row = regime_df[(regime_df["p_min"] <= pct) & (pct < regime_df["p_max"])]
    if row.empty:
        # Fallback: use closest regime by p_min
        idx = np.argmin(np.abs(regime_df["p_min"].values - pct))
        row = regime_df.iloc[[idx]]

    r0 = row.iloc[0]
    return {
        "long_dte_weeks": int(r0["long_dte_weeks"]),
        "otm_pts": float(r0["otm_pts"]),
        "target_mult": float(r0["target_mult"]),
        "sigma_mult": float(r0["sigma_mult"]),
    }


# --------------------------------------------------------
# Helper: parse lists from sidebar
# --------------------------------------------------------

def parse_float_list(text: str, default_vals):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


def parse_int_list(text: str, default_vals):
    try:
        vals = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


# --------------------------------------------------------
# Backtest: DIAGONAL (LEAP + weekly short OTM)
# --------------------------------------------------------

def backtest_diagonal(
    vix_weekly: pd.Series,
    vix_pct_series: pd.Series,
    mode: str,
    initial_capital: float,
    alloc_pct: float,
    entry_lookback_weeks: int,
    entry_percentile: float,
    long_dte_weeks: int,
    otm_pts: float,
    target_mult: float,
    sigma_mult: float,
    r: float,
    fee_per_contract: float,
    realistic_mode: bool,
    realism_level: str,
    regime_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Diagonal strategy:
    - Long leg: long_dte_weeks or dynamic from regime schedule.
    - Short leg: weekly OTM calls.
    - mode: "Static", "Static 90%-only", "Dynamic (regime)".
    """
    prices = np.asarray(vix_weekly, dtype=float).reshape(-1)
    dates = vix_weekly.index
    n = len(prices)
    if n < 5:
        return {}

    equity = np.zeros(n)
    weekly_ret = np.zeros(n)
    weekly_pnl_pct = np.zeros(n)

    realized_pnl_cum = np.zeros(n)
    realized_pnl_weekly = np.zeros(n)
    unrealized_pnl_weekly = np.zeros(n)

    cash = initial_capital
    equity[0] = initial_capital

    has_long = False
    long_contracts = 0
    long_value = 0.0
    long_ttm_weeks = 0
    long_entry_price = 0.0
    long_cost_basis = 0.0

    trade_count = 0
    durations = []
    current_trade_open_idx = None

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]
        pct = float(vix_pct_series.iloc[i]) if not np.isnan(vix_pct_series.iloc[i]) else np.nan

        # dynamic parameters for this step (if mode == Dynamic)
        if mode == "Dynamic (regime)":
            dyn = params_from_regime(pct, regime_df)
            long_dte_active = dyn["long_dte_weeks"]
            otm_active = dyn["otm_pts"]
            tgt_active = dyn["target_mult"]
            sig_active = dyn["sigma_mult"]
        else:
            long_dte_active = long_dte_weeks
            otm_active = otm_pts
            tgt_active = target_mult
            sig_active = sigma_mult

        equity[i] = cash + long_value
        realized_this_week = 0.0

        # ----- ENTRY -----
        if not has_long and S > 0:
            look_start = max(0, i - entry_lookback_weeks + 1)
            look_window = prices[look_start : i + 1]

            allow_entry = False
            if mode == "Static":
                if len(look_window) >= 4:
                    thr = np.quantile(look_window, entry_percentile)
                    if S <= thr:
                        allow_entry = True
            elif mode == "Static 90%-only":
                if not np.isnan(pct) and pct >= 0.90:
                    allow_entry = True
            elif mode == "Dynamic (regime)":
                if not np.isnan(pct):
                    allow_entry = True

            if allow_entry:
                T_long = long_dte_active / 52.0
                sigma_long = max((S / 100.0) * sig_active, 0.01)
                price_long = price_vix_call(
                    S_spot=S, K=S, T=T_long, r=r,
                    sigma_base=sigma_long,
                    vix_pct=pct, realistic_mode=realistic_mode,
                    realism_level=realism_level,
                )
                if price_long > 0:
                    equity_now = cash + long_value
                    risk_dollars = equity_now * alloc_pct
                    per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
                    contracts = int(risk_dollars // per_contract_cost)
                    if contracts > 0:
                        has_long = True
                        long_contracts = contracts
                        long_ttm_weeks = long_dte_active
                        long_entry_price = price_long
                        open_fee = fee_per_contract * contracts
                        cash -= price_long * 100.0 * contracts
                        cash -= open_fee
                        long_value = price_long * 100.0 * contracts
                        long_cost_basis = long_value + open_fee
                        trade_count += 1
                        current_trade_open_idx = i

        # ----- POSITION MANAGEMENT -----
        if has_long and long_contracts > 0:
            # weekly short call (always 1-week)
            T_short = 1.0 / 52.0
            K_short = S + otm_active
            sigma_short = max((S / 100.0) * sig_active, 0.01)
            price_short = price_vix_call(
                S_spot=S, K=K_short, T=T_short, r=r,
                sigma_base=sigma_short,
                vix_pct=pct, realistic_mode=realistic_mode,
                realism_level=realism_level,
            )
            if price_short > 0:
                short_prem = price_short * 100.0 * long_contracts
                short_open_fee = fee_per_contract * long_contracts
                cash += short_prem
                cash -= short_open_fee

                payoff_short = max(S_next - K_short, 0.0) * 100.0 * long_contracts
                short_close_fee = fee_per_contract * long_contracts
                cash -= payoff_short
                cash -= short_close_fee

                realized_this_week += short_prem - payoff_short - short_open_fee - short_close_fee

            # reprice long
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            T_new = long_ttm_weeks / 52.0
            sigma_long_new = max((S_next / 100.0) * sig_active, 0.01)
            price_long_new = price_vix_call(
                S_spot=S_next, K=S, T=T_new, r=r,
                sigma_base=sigma_long_new,
                vix_pct=pct, realistic_mode=realistic_mode,
                realism_level=realism_level,
            )
            new_val = price_long_new * 100.0 * long_contracts if T_new > 0 else 0.0

            exit_now = False
            if T_new <= 0:
                exit_now = True
            elif price_long_new >= tgt_active * long_entry_price:
                exit_now = True

            if exit_now:
                close_fee = fee_per_contract * long_contracts
                cash += new_val
                cash -= close_fee
                realized_long = new_val - long_cost_basis - close_fee
                realized_this_week += realized_long

                if current_trade_open_idx is not None:
                    dur_weeks = (i + 1) - current_trade_open_idx
                    if dur_weeks > 0:
                        durations.append(dur_weeks)

                has_long = False
                long_contracts = 0
                long_value = 0.0
                long_cost_basis = 0.0
                current_trade_open_idx = None
            else:
                long_value = new_val

        # ----- Weekly bookkeeping -----
        equity[i + 1] = cash + long_value
        eq_change = equity[i + 1] - equity[i]
        if equity[i] > 0:
            weekly_ret[i + 1] = eq_change / equity[i]
            weekly_pnl_pct[i + 1] = weekly_ret[i + 1] * 100.0
        realized_pnl_weekly[i + 1] = realized_this_week
        realized_pnl_cum[i + 1] = realized_pnl_cum[i] + realized_this_week
        unrealized_pnl_weekly[i + 1] = eq_change - realized_this_week

        if equity[i + 1] <= 0:
            equity[i + 1] = 0.0
            weekly_ret[i + 1 :] = -1.0
            weekly_pnl_pct[i + 1 :] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = calc_cagr(total_return, n_valid)
    valid_rets = weekly_ret[1 : n_valid + 1]
    if len(valid_rets) > 1 and np.std(valid_rets) > 1e-8:
        sharpe = (np.mean(valid_rets) * 52.0 - r) / (np.std(valid_rets) * np.sqrt(52.0))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(np.min(dd))
    win_rate = float(np.mean(valid_rets > 0.0)) if len(valid_rets) else 0.0

    if durations:
        avg_dur = float(np.mean(durations))
        min_dur = float(np.min(durations))
        max_dur = float(np.max(durations))
    else:
        avg_dur = min_dur = max_dur = np.nan

    return dict(
        structure="Diagonal",
        dates=dates,
        equity=equity,
        weekly_returns=weekly_ret,
        weekly_pnl_pct=weekly_pnl_pct,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        realized_pnl_cum=realized_pnl_cum,
        realized_pnl_weekly=realized_pnl_weekly,
        unrealized_pnl_weekly=unrealized_pnl_weekly,
        trade_count=trade_count,
        avg_trade_duration=avg_dur,
        min_trade_duration=min_dur,
        max_trade_duration=max_dur,
    )


# --------------------------------------------------------
# Backtest: LONG-ONLY
# --------------------------------------------------------

def backtest_long_only(
    vix_weekly: pd.Series,
    vix_pct_series: pd.Series,
    mode: str,
    exit_mode: str,
    initial_capital: float,
    alloc_pct: float,
    entry_lookback_weeks: int,
    entry_percentile: float,
    long_dte_weeks: int,
    target_mult: float,
    sigma_mult: float,
    stop_mult: float,
    exit_pct_threshold: float,
    r: float,
    fee_per_contract: float,
    realistic_mode: bool,
    realism_level: str,
) -> Dict[str, Any]:
    """
    Long-only ATM VIX call strategy.
    exit_mode:
        - "TP only"
        - "TP + stop"
        - "Percentile exit"
    """
    prices = np.asarray(vix_weekly, dtype=float).reshape(-1)
    dates = vix_weekly.index
    n = len(prices)
    if n < 5:
        return {}

    equity = np.zeros(n)
    weekly_ret = np.zeros(n)
    weekly_pnl_pct = np.zeros(n)

    realized_pnl_cum = np.zeros(n)
    realized_pnl_weekly = np.zeros(n)
    unrealized_pnl_weekly = np.zeros(n)

    cash = initial_capital
    equity[0] = initial_capital

    has_long = False
    long_contracts = 0
    long_value = 0.0
    long_ttm_weeks = 0
    long_entry_price = 0.0
    long_cost_basis = 0.0

    trade_count = 0
    current_trade_open_idx = None
    durations = []

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]
        pct = float(vix_pct_series.iloc[i]) if not np.isnan(vix_pct_series.iloc[i]) else np.nan

        equity[i] = cash + long_value
        realized_this_week = 0.0

        # ENTRY
        if not has_long and S > 0:
            look_start = max(0, i - entry_lookback_weeks + 1)
            look_window = prices[look_start : i + 1]

            allow_entry = False
            if mode == "Static":
                if len(look_window) >= 4:
                    thr = np.quantile(look_window, entry_percentile)
                    if S <= thr:
                        allow_entry = True
            elif mode == "Static 90%-only":
                if not np.isnan(pct) and pct >= 0.90:
                    allow_entry = True
            elif mode == "Dynamic (regime)":
                if not np.isnan(pct):
                    allow_entry = True

            if allow_entry:
                long_ttm_weeks = long_dte_weeks
                T_long = long_ttm_weeks / 52.0
                sigma_long = max((S / 100.0) * sigma_mult, 0.01)
                price_long = price_vix_call(
                    S_spot=S, K=S, T=T_long, r=r,
                    sigma_base=sigma_long,
                    vix_pct=pct, realistic_mode=realistic_mode,
                    realism_level=realism_level,
                )
                if price_long > 0:
                    equity_now = cash + long_value
                    risk_dollars = equity_now * alloc_pct
                    per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
                    contracts = int(risk_dollars // per_contract_cost)
                    if contracts > 0:
                        has_long = True
                        long_contracts = contracts
                        long_entry_price = price_long
                        open_fee = fee_per_contract * contracts
                        cash -= price_long * 100.0 * contracts
                        cash -= open_fee
                        long_value = price_long * 100.0 * contracts
                        long_cost_basis = long_value + open_fee
                        trade_count += 1
                        current_trade_open_idx = i

        # MANAGEMENT
        if has_long and long_contracts > 0:
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            T_new = long_ttm_weeks / 52.0
            sigma_long_new = max((S_next / 100.0) * sigma_mult, 0.01)
            price_long_new = price_vix_call(
                S_spot=S_next, K=S, T=T_new, r=r,
                sigma_base=sigma_long_new,
                vix_pct=pct, realistic_mode=realistic_mode,
                realism_level=realism_level,
            )
            new_val = price_long_new * 100.0 * long_contracts if T_new > 0 else 0.0

            exit_now = False
            if price_long_new >= target_mult * long_entry_price:
                exit_now = True
            if T_new <= 0:
                exit_now = True
            if exit_mode == "TP + stop":
                if price_long_new <= stop_mult * long_entry_price:
                    exit_now = True
            if exit_mode == "Percentile exit" and not np.isnan(pct):
                if pct <= exit_pct_threshold:
                    exit_now = True

            if exit_now:
                close_fee = fee_per_contract * long_contracts
                cash += new_val
                cash -= close_fee
                realized_long = new_val - long_cost_basis - close_fee
                realized_this_week += realized_long

                if current_trade_open_idx is not None:
                    dur_weeks = (i + 1) - current_trade_open_idx
                    if dur_weeks > 0:
                        durations.append(dur_weeks)

                has_long = False
                long_contracts = 0
                long_value = 0.0
                long_cost_basis = 0.0
                current_trade_open_idx = None
            else:
                long_value = new_val

        # WEEKLY BOOKKEEPING
        equity[i + 1] = cash + long_value
        eq_change = equity[i + 1] - equity[i]
        if equity[i] > 0:
            weekly_ret[i + 1] = eq_change / equity[i]
            weekly_pnl_pct[i + 1] = weekly_ret[i + 1] * 100.0
        realized_pnl_weekly[i + 1] = realized_this_week
        realized_pnl_cum[i + 1] = realized_pnl_cum[i] + realized_this_week
        unrealized_pnl_weekly[i + 1] = eq_change - realized_this_week

        if equity[i + 1] <= 0:
            equity[i + 1] = 0.0
            weekly_ret[i + 1 :] = -1.0
            weekly_pnl_pct[i + 1 :] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = calc_cagr(total_return, n_valid)
    valid_rets = weekly_ret[1 : n_valid + 1]
    if len(valid_rets) > 1 and np.std(valid_rets) > 1e-8:
        sharpe = (np.mean(valid_rets) * 52.0 - r) / (np.std(valid_rets) * np.sqrt(52.0))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(np.min(dd))
    win_rate = float(np.mean(valid_rets > 0.0)) if len(valid_rets) else 0.0

    if durations:
        avg_dur = float(np.mean(durations))
        min_dur = float(np.min(durations))
        max_dur = float(np.max(durations))
    else:
        avg_dur = min_dur = max_dur = np.nan

    return dict(
        structure="Long-only",
        dates=dates,
        equity=equity,
        weekly_returns=weekly_ret,
        weekly_pnl_pct=weekly_pnl_pct,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        realized_pnl_cum=realized_pnl_cum,
        realized_pnl_weekly=realized_pnl_weekly,
        unrealized_pnl_weekly=unrealized_pnl_weekly,
        trade_count=trade_count,
        avg_trade_duration=avg_dur,
        min_trade_duration=min_dur,
        max_trade_duration=max_dur,
    )


# --------------------------------------------------------
# Grid Scan: DIAGONAL (with long DTE!)
# --------------------------------------------------------

def score_row(row: pd.Series, dd_penalty: float = 2.0) -> float:
    return row["cagr"] - dd_penalty * abs(row["max_dd"])


def grid_scan_diagonal(
    vix_weekly: pd.Series,
    vix_pct_series: pd.Series,
    initial_capital: float,
    alloc_pct: float,
    r: float,
    fee_per_contract: float,
    entry_grid,
    otm_grid,
    target_grid,
    sigma_grid,
    dte_grid,
    entry_lookback_weeks: int,
    realistic_mode: bool,
    realism_level: str,
) -> pd.DataFrame:
    rows = []
    for dte in dte_grid:
        for ep in entry_grid:
            for otm in otm_grid:
                for tgt in target_grid:
                    for sig in sigma_grid:
                        res = backtest_diagonal(
                            vix_weekly=vix_weekly,
                            vix_pct_series=vix_pct_series,
                            mode="Static",
                            initial_capital=initial_capital,
                            alloc_pct=alloc_pct,
                            entry_lookback_weeks=entry_lookback_weeks,
                            entry_percentile=ep,
                            long_dte_weeks=int(dte),
                            otm_pts=otm,
                            target_mult=tgt,
                            sigma_mult=sig,
                            r=r,
                            fee_per_contract=fee_per_contract,
                            realistic_mode=realistic_mode,
                            realism_level=realism_level,
                            regime_df=None,
                        )
                        rows.append(
                            dict(
                                entry_pct=ep,
                                otm_pts=otm,
                                target_mult=tgt,
                                sigma_mult=sig,
                                long_dte_weeks=int(dte),
                                total_return=res["total_return"],
                                cagr=res["cagr"],
                                sharpe=res["sharpe"],
                                max_dd=res["max_dd"],
                                win_rate=res["win_rate"],
                                trades=res["trade_count"],
                                avg_dur=res["avg_trade_duration"],
                            )
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df.apply(score_row, axis=1)
    df = df.sort_values(["score", "cagr"], ascending=[False, False])
    return df


# --------------------------------------------------------
# Grid Scan: LONG-ONLY
# --------------------------------------------------------

def grid_scan_long_only(
    vix_weekly: pd.Series,
    vix_pct_series: pd.Series,
    initial_capital: float,
    alloc_pct: float,
    r: float,
    fee_per_contract: float,
    entry_grid,
    target_grid,
    sigma_grid,
    dte_grid,
    exit_modes,
    entry_lookback_weeks: int,
    stop_mult: float,
    exit_pct_threshold: float,
    realistic_mode: bool,
    realism_level: str,
) -> pd.DataFrame:
    rows = []
    for emode in exit_modes:
        for dte in dte_grid:
            for ep in entry_grid:
                for tgt in target_grid:
                    for sig in sigma_grid:
                        res = backtest_long_only(
                            vix_weekly=vix_weekly,
                            vix_pct_series=vix_pct_series,
                            mode="Static",
                            exit_mode=emode,
                            initial_capital=initial_capital,
                            alloc_pct=alloc_pct,
                            entry_lookback_weeks=entry_lookback_weeks,
                            entry_percentile=ep,
                            long_dte_weeks=int(dte),
                            target_mult=tgt,
                            sigma_mult=sig,
                            stop_mult=stop_mult,
                            exit_pct_threshold=exit_pct_threshold,
                            r=r,
                            fee_per_contract=fee_per_contract,
                            realistic_mode=realistic_mode,
                            realism_level=realism_level,
                        )
                        rows.append(
                            dict(
                                exit_mode=emode,
                                entry_pct=ep,
                                long_dte_weeks=int(dte),
                                target_mult=tgt,
                                sigma_mult=sig,
                                total_return=res["total_return"],
                                cagr=res["cagr"],
                                sharpe=res["sharpe"],
                                max_dd=res["max_dd"],
                                win_rate=res["win_rate"],
                                trades=res["trade_count"],
                                avg_dur=res["avg_trade_duration"],
                            )
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df.apply(score_row, axis=1)
    df = df.sort_values(["score", "cagr"], ascending=[False, False])
    return df


# --------------------------------------------------------
# Data loading
# --------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_vix_weekly(start_date: dt.date, end_date: dt.date) -> pd.Series:
    df = yf.download("^VIX", start=start_date, end=end_date)
    if df.empty:
        return pd.Series(dtype=float)
    weekly = df["Close"].resample("W-FRI").last().dropna()
    return weekly


def compute_vix_percentile(vix_weekly: pd.Series, window_weeks: int = 52) -> pd.Series:
    vals = np.asarray(vix_weekly, dtype=float).reshape(-1)
    pct = np.full(len(vals), np.nan, dtype=float)
    for i in range(len(vals)):
        if i >= window_weeks - 1:
            w = vals[i - window_weeks + 1 : i + 1]
            cur = vals[i]
            pct[i] = np.mean(w <= cur)
    return pd.Series(pct, index=vix_weekly.index, name="VIX_52w_percentile")


# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------

st.set_page_config(page_title="VIX 5% Weekly – D3 Engine", layout="wide")

st.title("VIX 5% Weekly – D3 Engine (Diagonal + Long-only)")

# --- Sidebar: global settings ---
st.sidebar.header("Global Settings")

min_date = dt.date(2004, 1, 1)
today = dt.date.today()
start_date = st.sidebar.date_input("Start date", value=dt.date(2015, 1, 1),
                                   min_value=min_date, max_value=today)
end_date = st.sidebar.date_input("End date", value=today,
                                 min_value=start_date, max_value=today)

initial_capital_str = st.sidebar.text_input(
    "Initial Capital ($)", value="250,000", help="You like 250,000 as baseline."
)
try:
    initial_capital = float(initial_capital_str.replace(",", ""))
except ValueError:
    initial_capital = 250_000.0

alloc_pct_percent = st.sidebar.number_input(
    "Fraction of equity allocated to long leg (%)",
    min_value=0.1, max_value=50.0, value=1.0, step=0.1
)
alloc_pct = alloc_pct_percent / 100.0

risk_free = st.sidebar.slider("Risk-free (annual)", 0.0, 0.10, 0.03, 0.005)
fee_per_contract = st.sidebar.number_input("Fee per contract ($)", 0.0, 10.0, 0.65, 0.05)

realistic_mode = st.sidebar.checkbox(
    "Realistic VIX options mode", value=True,
    help="ON = haircut BS prices to mimic futures / contango / IV crush / spreads."
)
realism_level = st.sidebar.selectbox("Realism Strength", ["Optimistic", "Normal", "Conservative"], index=1)

page = st.sidebar.radio(
    "Page",
    ["Dashboard", "Single Backtest", "Grid Scans", "Regime Engine", "Playbook / Help"],
    index=0,
)

# --- Load data ---
vix_weekly = load_vix_weekly(start_date, end_date)
if vix_weekly.empty:
    st.error("No VIX data downloaded. Check dates or connection.")
    st.stop()

vix_pct_series = compute_vix_percentile(vix_weekly, window_weeks=52)
current_vix = float(vix_weekly.iloc[-1])
current_pct = float(vix_pct_series.iloc[-1]) if not np.isnan(vix_pct_series.iloc[-1]) else np.nan

st.info(
    f"Loaded {len(vix_weekly)} weekly VIX observations "
    f"({vix_weekly.index[0].date()} → {vix_weekly.index[-1].date()})."
)

# --- Regime schedule in session_state ---
if "regime_df" not in st.session_state:
    st.session_state.regime_df = default_regime_schedule()
regime_df = st.session_state.regime_df


# ========================================================
# PAGE: DASHBOARD
# ========================================================

if page == "Dashboard":
    st.subheader("Dashboard – quick view")

    # Run a baseline diagonal with Dynamic regime
    baseline_res = backtest_diagonal(
        vix_weekly=vix_weekly,
        vix_pct_series=vix_pct_series,
        mode="Dynamic (regime)",
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
        entry_lookback_weeks=52,
        entry_percentile=0.10,
        long_dte_weeks=26,   # not used in dynamic
        otm_pts=3.0,
        target_mult=1.2,
        sigma_mult=1.0,
        r=risk_free,
        fee_per_contract=fee_per_contract,
        realistic_mode=realistic_mode,
        realism_level=realism_level,
        regime_df=regime_df,
    )

    equity = baseline_res["equity"]
    total_return = baseline_res["total_return"]
    cagr = baseline_res["cagr"]
    sharpe = baseline_res["sharpe"]
    max_dd = baseline_res["max_dd"]
    win_rate = baseline_res["win_rate"]
    trade_count = baseline_res["trade_count"]
    avg_dur = baseline_res["avg_trade_duration"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Return", f"{total_return*100:.1f}%")
    col2.metric("CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "N/A")
    col3.metric("Sharpe", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    col5.metric("Win Rate", f"{win_rate*100:.1f}%")
    col6.metric("Trades", f"{trade_count} (avg {avg_dur:.1f}w)")

    # Charts
    dates = baseline_res["dates"]
    vix_vals = vix_weekly.reindex(dates).values
    weekly_pnl_pct = baseline_res["weekly_pnl_pct"]
    pct_vals = vix_pct_series.reindex(dates).values * 100.0

    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots()
    ax1.plot(dates, equity, linewidth=1.0, label="Equity")
    ax1.set_ylabel("Equity ($)")
    ax2 = ax1.twinx()
    ax2.plot(dates, vix_vals, linestyle="--", linewidth=1.0, label="VIX")
    ax2.set_ylabel("VIX")
    ax1.set_title("Diagonal (Dynamic Regime) – Equity vs VIX")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig2, axp = plt.subplots()
    axp.bar(dates, weekly_pnl_pct, width=3)
    axp.set_ylabel("Weekly PnL (%)")
    axp.set_title("Weekly PnL (%) and VIX Percentile")
    axp.grid(True, alpha=0.3)
    ax3 = axp.twinx()
    ax3.plot(dates, pct_vals, linestyle="--", linewidth=0.8)
    ax3.set_ylabel("VIX 52w Percentile (%)")

    plt.style.use("default")

    c1, c2 = st.columns(2)
    c1.pyplot(fig1, clear_figure=True)
    c2.pyplot(fig2, clear_figure=True)

    st.markdown(
        f"""
**Current VIX:** {current_vix:.2f}  
**Current 52w percentile:** {current_pct*100:.1f}% (if NaN → not enough history)

This dashboard runs your **Diagonal (Dynamic Regime)** as a baseline:
- Long DTE, OTM distance, target, sigma_mult all change with VIX percentile.
- Realistic pricing: `{realistic_mode}` (level `{realism_level}`).
"""
    )

    st.markdown("----")
    st.markdown("### TL;DR")
    st.write(
        "- Dashboard shows your **adaptive diagonal** over this period.\n"
        "- Use this page to quickly feel if the regime engine is behaving.\n"
        "- For DTE tests and deeper analysis, use **Grid Scans** and **Regime Engine** pages."
    )


# ========================================================
# PAGE: SINGLE BACKTEST
# ========================================================

elif page == "Single Backtest":
    st.subheader("Single Backtest – Diagonal or Long-only")

    colA, colB = st.columns(2)
    with colA:
        structure = st.radio(
            "Structure",
            ["Diagonal: LEAP + Weekly OTM", "Long-only: VIX Call"],
            index=0,
        )
    with colB:
        mode = st.radio(
            "Entry / Regime Mode",
            ["Static", "Static 90%-only", "Dynamic (regime)"],
            index=0,
        )

    entry_lookback = 52
    entry_percentile = st.slider("Entry percentile (Static only)", 0.0, 0.3, 0.10, 0.01)

    if "Diagonal" in structure:
        long_dte_weeks = st.slider("Long DTE (weeks – baseline)", 4, 52, 26, 1)
        otm_pts = st.slider("Short call OTM distance (pts)", 1.0, 20.0, 3.0, 0.5)
        target_mult = st.slider("Target multiple for long", 1.05, 5.0, 1.20, 0.05)
        sigma_mult = st.slider("Sigma multiplier", 0.1, 3.0, 1.0, 0.1)

        res = backtest_diagonal(
            vix_weekly=vix_weekly,
            vix_pct_series=vix_pct_series,
            mode=mode,
            initial_capital=initial_capital,
            alloc_pct=alloc_pct,
            entry_lookback_weeks=entry_lookback,
            entry_percentile=entry_percentile,
            long_dte_weeks=long_dte_weeks,
            otm_pts=otm_pts,
            target_mult=target_mult,
            sigma_mult=sigma_mult,
            r=risk_free,
            fee_per_contract=fee_per_contract,
            realistic_mode=realistic_mode,
            realism_level=realism_level,
            regime_df=regime_df,
        )
    else:
        long_dte_weeks = st.slider("Long-only DTE (weeks)", 1, 26, 3, 1)
        target_mult = st.slider("Target multiple (long-only)", 1.05, 3.0, 1.20, 0.05)
        sigma_mult = st.slider("Sigma multiplier (long-only)", 0.1, 2.0, 0.5, 0.1)
        exit_mode = st.selectbox("Exit mode", ["TP only", "TP + stop", "Percentile exit"], index=1)
        stop_mult = st.slider("Stop multiplier (TP + stop)", 0.1, 0.9, 0.5, 0.05)
        exit_pct_threshold = st.slider("Exit percentile (Percentile exit)", 0.0, 1.0, 0.5, 0.05)

        res = backtest_long_only(
            vix_weekly=vix_weekly,
            vix_pct_series=vix_pct_series,
            mode=mode,
            exit_mode=exit_mode,
            initial_capital=initial_capital,
            alloc_pct=alloc_pct,
            entry_lookback_weeks=entry_lookback,
            entry_percentile=entry_percentile,
            long_dte_weeks=long_dte_weeks,
            target_mult=target_mult,
            sigma_mult=sigma_mult,
            stop_mult=stop_mult,
            exit_pct_threshold=exit_pct_threshold,
            r=risk_free,
            fee_per_contract=fee_per_contract,
            realistic_mode=realistic_mode,
            realism_level=realism_level,
        )

    if not res:
        st.warning("Not enough data.")
    else:
        equity = res["equity"]
        dates = res["dates"]
        vix_vals = vix_weekly.reindex(dates).values
        vix_pct_vals = vix_pct_series.reindex(dates).values * 100.0
        weekly_pnl_pct = res["weekly_pnl_pct"]

        total_return = res["total_return"]
        cagr = res["cagr"]
        sharpe = res["sharpe"]
        max_dd = res["max_dd"]
        win_rate = res["win_rate"]
        trade_count = res["trade_count"]
        avg_dur = res["avg_trade_duration"]
        ending_equity = float(equity[-1])
        tr_dollars = ending_equity - initial_capital

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Return", f"{total_return*100:.1f}%")
        c2.metric("Total Return ($)", f"{tr_dollars:,.0f}")
        c3.metric("CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "N/A")
        c4.metric("Sharpe", f"{sharpe:.2f}")
        c5.metric("Max DD", f"{max_dd*100:.1f}%")
        c6.metric("Trades", f"{trade_count} (avg {avg_dur:.1f}w)")

        plt.style.use("dark_background")
        fig1, ax1 = plt.subplots()
        ax1.plot(dates, equity, linewidth=1.0, label="Equity")
        ax1.set_ylabel("Equity ($)")
        ax2 = ax1.twinx()
        ax2.plot(dates, vix_vals, linestyle="--", linewidth=1.0, label="VIX")
        ax2.set_ylabel("VIX")
        ax1.set_title(f"{res['structure']} – Equity vs VIX")
        ax1.grid(True, alpha=0.3)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        fig2, axp = plt.subplots()
        axp.bar(dates, weekly_pnl_pct, width=3)
        axp.set_ylabel("Weekly PnL (%)")
        axp.set_title("Weekly PnL (%) and VIX Percentile")
        axp.grid(True, alpha=0.3)
        ax3 = axp.twinx()
        ax3.plot(dates, vix_pct_vals, linestyle="--", linewidth=0.8)
        ax3.set_ylabel("VIX 52w Percentile (%)")
        plt.style.use("default")

        cc1, cc2 = st.columns(2)
        cc1.pyplot(fig1, clear_figure=True)
        cc2.pyplot(fig2, clear_figure=True)

        st.markdown("----")
        st.markdown("### TL;DR")
        st.write(
            "- This page lets you experiment with **one configuration at a time**.\n"
            "- For Diagonal: vary long DTE, OTM, target, sigma, and (optionally) let regime mode drive them.\n"
            "- For Long-only: use it as a **research lens**, not necessarily as your main trading engine."
        )


# ========================================================
# PAGE: GRID SCANS
# ========================================================

elif page == "Grid Scans":
    st.subheader("Grid Scans – DTE tests and exit methods")

    entry_grid_str = st.text_input("Entry percentiles", "0.10, 0.30, 0.50, 0.90")
    target_grid_str = st.text_input("Target multiples", "1.10, 1.15, 1.20")
    sigma_grid_str = st.text_input("Sigma multipliers", "0.3, 0.5, 0.8, 1.0")
    dte_grid_str = st.text_input("Long DTE weeks", "3, 8, 13, 26, 40, 52")
    otm_grid_str = st.text_input("OTM distances (Diagonal only)", "3, 5, 7, 10")

    entry_grid = parse_float_list(entry_grid_str, [0.1, 0.3])
    target_grid = parse_float_list(target_grid_str, [1.10, 1.20])
    sigma_grid = parse_float_list(sigma_grid_str, [0.5, 1.0])
    dte_grid = parse_int_list(dte_grid_str, [3, 13, 26])
    otm_grid = parse_float_list(otm_grid_str, [3.0, 5.0, 7.0])

    st.markdown("### Diagonal grid (with DTE dimension)")
    col_run_d, col_clear_d = st.columns(2)
    if "diag_grid_df" not in st.session_state:
        st.session_state.diag_grid_df = None
        st.session_state.diag_xlsx = None

    with col_run_d:
        run_diag = st.button("Run Diagonal Grid Scan")
    with col_clear_d:
        clear_diag = st.button("Clear Diagonal Scan")

    if clear_diag:
        st.session_state.diag_grid_df = None
        st.session_state.diag_xlsx = None

    if run_diag:
        with st.spinner("Running diagonal grid scan..."):
            df_diag = grid_scan_diagonal(
                vix_weekly=vix_weekly,
                vix_pct_series=vix_pct_series,
                initial_capital=initial_capital,
                alloc_pct=alloc_pct,
                r=risk_free,
                fee_per_contract=fee_per_contract,
                entry_grid=entry_grid,
                otm_grid=otm_grid,
                target_grid=target_grid,
                sigma_grid=sigma_grid,
                dte_grid=dte_grid,
                entry_lookback_weeks=52,
                realistic_mode=realistic_mode,
                realism_level=realism_level,
            )
            st.session_state.diag_grid_df = df_diag

    df_diag = st.session_state.diag_grid_df
    if df_diag is not None and not df_diag.empty:
        st.dataframe(df_diag.round(4), use_container_width=True)
        # Group by DTE to see pattern
        st.markdown("**Average performance by long DTE (Diagonal):**")
        summary_dte = (
            df_diag.groupby("long_dte_weeks")[["cagr", "max_dd", "sharpe"]]
            .mean()
            .sort_index()
        )
        st.dataframe(summary_dte.round(4), use_container_width=True)

        # XLSX export
        if st.button("Generate Diagonal XLSX"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_diag.to_excel(writer, index=False, sheet_name="diag_scan")
            buf.seek(0)
            st.session_state.diag_xlsx = buf.read()
        if st.session_state.diag_xlsx:
            st.download_button(
                "Download Diagonal Scan XLSX",
                data=st.session_state.diag_xlsx,
                file_name="vix_diagonal_grid_scan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("### TL;DR – Diagonal DTE behavior")
        st.write(
            "- Look at the **summary table by `long_dte_weeks`** to see which DTE gives the best balance of CAGR, Sharpe, and MaxDD.\n"
            "- This is where you answer: *“Should I use the longest DTE or something in the middle?”*\n"
            "- Often, 13–26 weeks is a sweet spot; this table will tell you for your date range."
        )
    else:
        st.info("Run a Diagonal grid scan to see results.")

    st.markdown("---")
    st.markdown("### Long-only grid (exit method testing)")

    exit_modes_str = st.text_input(
        "Exit modes to test (Long-only)",
        "TP only, TP + stop, Percentile exit",
    )
    exit_tokens = [x.strip() for x in exit_modes_str.split(",") if x.strip()]
    valid_exits = {"TP only", "TP + stop", "Percentile exit"}
    exit_modes = [e for e in exit_tokens if e in valid_exits] or ["TP only"]

    stop_mult = st.slider("Stop multiplier for grid (TP + stop)", 0.1, 0.9, 0.5, 0.05)
    exit_pct_threshold = st.slider("Exit percentile for grid (Percentile exit)", 0.0, 1.0, 0.5, 0.05)

    col_run_l, col_clear_l = st.columns(2)
    if "long_grid_df" not in st.session_state:
        st.session_state.long_grid_df = None
        st.session_state.long_xlsx = None

    with col_run_l:
        run_long = st.button("Run Long-only Grid Scan")
    with col_clear_l:
        clear_long = st.button("Clear Long-only Scan")

    if clear_long:
        st.session_state.long_grid_df = None
        st.session_state.long_xlsx = None

    if run_long:
        with st.spinner("Running long-only grid scan..."):
            df_long = grid_scan_long_only(
                vix_weekly=vix_weekly,
                vix_pct_series=vix_pct_series,
                initial_capital=initial_capital,
                alloc_pct=alloc_pct,
                r=risk_free,
                fee_per_contract=fee_per_contract,
                entry_grid=entry_grid,
                target_grid=target_grid,
                sigma_grid=sigma_grid,
                dte_grid=dte_grid,
                exit_modes=exit_modes,
                entry_lookback_weeks=52,
                stop_mult=stop_mult,
                exit_pct_threshold=exit_pct_threshold,
                realistic_mode=realistic_mode,
                realism_level=realism_level,
            )
            st.session_state.long_grid_df = df_long

    df_long = st.session_state.long_grid_df
    if df_long is not None and not df_long.empty:
        st.dataframe(df_long.round(4), use_container_width=True)

        summary_exit = (
            df_long.groupby("exit_mode")[["cagr", "max_dd", "sharpe", "trades"]]
            .mean()
            .sort_values("cagr", ascending=False)
        )
        st.markdown("**Average performance per exit mode (Long-only):**")
        st.dataframe(summary_exit.round(4), use_container_width=True)

        # XLSX export
        if st.button("Generate Long-only XLSX"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_long.to_excel(writer, index=False, sheet_name="long_scan")
            buf.seek(0)
            st.session_state.long_xlsx = buf.read()
        if st.session_state.long_xlsx:
            st.download_button(
                "Download Long-only Scan XLSX",
                data=st.session_state.long_xlsx,
                file_name="vix_longonly_grid_scan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("### TL;DR – Long-only insights")
        st.write(
            "- Use this as a **research tool** to see which exit method (`TP only`, `TP + stop`, `Percentile exit`) behaves best.\n"
            "- You already saw: high percentile (0.9), 3-week DTE, small target (1.15–1.20), sigma ~0.3, and `TP + stop` often dominate.\n"
            "- These insights can then **inform your diagonal regime settings**, especially for high-VIX regimes."
        )
    else:
        st.info("Run a Long-only grid scan to see exit-method behavior.")


# ========================================================
# PAGE: REGIME ENGINE
# ========================================================

elif page == "Regime Engine":
    st.subheader("Regime Engine – percentile-based diagonal schedule")

    st.markdown(
        "This table controls how **Diagonal (Dynamic mode)** chooses long DTE, OTM distance, target, and sigma "
        "based on VIX 52-week percentile."
    )

    edited_regime = st.data_editor(
        regime_df,
        num_rows="dynamic",
        use_container_width=True,
        key="regime_editor",
    )
    st.session_state.regime_df = edited_regime
    regime_df = edited_regime

    st.markdown("**Current schedule interpretation:**")
    for _, row in regime_df.iterrows():
        st.write(
            f"- Percentile {row['p_min']:.2f}–{row['p_max']:.2f} → "
            f"DTE={int(row['long_dte_weeks'])}w, OTM={row['otm_pts']:.1f}pts, "
            f"Target={row['target_mult']:.2f}x, Sigma_mult={row['sigma_mult']:.2f}"
        )

    st.markdown("### TL;DR")
    st.write(
        "- This page defines how the **Diagonal Dynamic Regime** behaves.\n"
        "- You can use **Diagonal grid results** plus **Long-only insights** to refine these buckets.\n"
        "- For example, for VIX percentile ≥ 0.9, you might set DTE=8–13, target=1.10, sigma_mult=0.5–0.8."
    )


# ========================================================
# PAGE: PLAYBOOK / HELP
# ========================================================

else:
    st.subheader("Playbook / Help")

    st.markdown(
        f"""
### Concept

You are testing a **VIX 5% Weekly Diagonal** strategy with:
- Long-dated ATM VIX calls (LEAP-like)  
- Weekly short OTM calls for income  
- Optionally, a **Long-only** engine as a benchmark

### Realistic pricing

- `Realistic VIX options mode` ON applies a haircut to Black–Scholes prices to approximate:
  - VIX futures term structure / contango
  - IV crush after spikes
  - Bid–ask spreads / slippage
- This prevents impossible 10^9–10^19 equity explosions from pure BS modelling.

### Pages

1. **Dashboard**  
   - Runs **Diagonal (Dynamic Regime)** using your current regime schedule.
   - Use this to get a feel for overall performance.

2. **Single Backtest**  
   - Choose `Diagonal` or `Long-only`.  
   - Choose mode: `Static`, `Static 90%-only`, or `Dynamic (regime)`.  
   - Tune parameters and inspect equity curves, PnL, trade count, durations.

3. **Grid Scans**  
   - **Diagonal Grid**: tests entry_pct, OTM, target, sigma_mult, and **long DTE**.  
   - **Long-only Grid**: tests entry_pct, target, sigma_mult, DTE, and **exit methods**.  
   - Use the DTE summary table to see if 13, 26, 40, or 52 weeks is best for diagonal.

4. **Regime Engine**  
   - Edit the **percentile-based schedule** for Diagonal (Dynamic mode).  
   - High percentile ⇒ shorter DTE & smaller target; low percentile ⇒ longer DTE & larger target.

5. **Playbook / Help** (this page)  
   - Human explanation & notes.

### Practical way to use this

1. Run **Diagonal grid scan** with DTE range like `[8,13,26,40,52]`.  
2. Look at **summary by long DTE** to see where CAGR/Sharpe/MaxDD look best.  
3. Run **Long-only grid** to understand:
   - In high VIX regimes, which DTE, target, sigma, and exit method make sense.  
4. Update the **Regime Engine table**:
   - Calm (pct < 0.4) → DTE long (26–40), bigger targets (1.25+).  
   - High (pct > 0.9) → DTE shortish (8–13), small targets (1.05–1.15), lower sigma_mult.
5. Use **Dashboard** (Diagonal Dynamic) to see combined behaviour.

### TL;DR

- Diagonal = your *main* strategy; Long-only = *research lens*.  
- Longest DTE should be **tested**, not **assumed best**.  
- Use grid scans + regime engine to let the diagonal **adapt to VIX percentile**, not stay fixed.
"""
    )

# End of app