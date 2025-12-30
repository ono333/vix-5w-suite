#!/usr/bin/env python3
"""
VIX 5% Weekly – D3 Engine v13
Diagonal (LEAP + Weekly OTM) & Long-only Backtester
with Trade Explorer (long + short legs plotted on VIX)

Key features:
- Realistic VIX option pricing (haircut on BS)
- Diagonal & Long-only backtests
- Short-call modes: Roll / No Roll / ASL (adaptive by percentile)
- Grid scans (Diagonal with long DTE; Long-only with exit modes)
- Percentile-based Regime Engine for Diagonal Dynamic mode
- Trade Explorer: plot entries/exits of long + short on VIX, plus trade tables
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
# Math helpers
# --------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
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
    if np.isnan(pct):
        pct = 0.5
    if realism_level == "Optimistic":
        base = 0.75
    elif realism_level == "Conservative":
        base = 0.45
    else:
        base = 0.60
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
    bs_price = black_scholes_call(S_spot, K, T, r, sigma_base)
    if not realistic_mode or T <= 0:
        return bs_price
    factor = realism_haircut_factor(vix_pct, realism_level)
    return bs_price * factor


# --------------------------------------------------------
# Regime schedule (for Diagonal Dynamic mode)
# --------------------------------------------------------

def default_regime_schedule() -> pd.DataFrame:
    rows = [
        {"p_min": 0.00, "p_max": 0.40, "long_dte_weeks": 40, "otm_pts": 7.0,
         "target_mult": 1.30, "sigma_mult": 1.2},
        {"p_min": 0.40, "p_max": 0.70, "long_dte_weeks": 26, "otm_pts": 5.0,
         "target_mult": 1.20, "sigma_mult": 1.0},
        {"p_min": 0.70, "p_max": 0.90, "long_dte_weeks": 20, "otm_pts": 4.0,
         "target_mult": 1.15, "sigma_mult": 0.8},
        {"p_min": 0.90, "p_max": 0.97, "long_dte_weeks": 13, "otm_pts": 3.0,
         "target_mult": 1.10, "sigma_mult": 0.6},
        {"p_min": 0.97, "p_max": 1.00, "long_dte_weeks": 8, "otm_pts": 2.0,
         "target_mult": 1.05, "sigma_mult": 0.5},
    ]
    return pd.DataFrame(rows)


def params_from_regime(pct: float, regime_df: pd.DataFrame) -> Dict[str, float]:
    if regime_df is None or regime_df.empty:
        regime_df = default_regime_schedule()
    if np.isnan(pct):
        pct = 0.5
    row = regime_df[(regime_df["p_min"] <= pct) & (pct < regime_df["p_max"])]
    if row.empty:
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
# Parse helpers
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
# Diagonal backtest (with short_mode and trade/event logs)
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
    short_mode: str = "ASL (adaptive)",
) -> Dict[str, Any]:
    """
    Diagonal strategy: LEAP-like long + weekly short OTM.
    - mode: "Static", "Static 90%-only", "Dynamic (regime)"
    - short_mode:
        "Roll (always)"         → sell weekly, every week, while long open
        "No Roll (exit & wait)" → sell only in week 0 of each long
        "ASL (adaptive)"        → percentile-based logic:
                                   pct < 0.60: short with otm_pts
                              0.60 ≤ pct < 0.80: short with 1.5 * otm_pts
                                   pct ≥ 0.80: no new short calls
    Returns:
      - Full equity series
      - Performance metrics
      - Trade log (long leg)
      - Event log (long + short entries/exits) for plotting
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
    weeks_since_long_entry = 0

    trades = []   # per-trade (long leg)
    events = []   # timeline events (long+short)

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]
        pct = float(vix_pct_series.iloc[i]) if not np.isnan(vix_pct_series.iloc[i]) else np.nan

        # --- Dynamic regime params (Diagonal engine) ---
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

        # --- Long ENTRY ---
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
                        weeks_since_long_entry = 0

                        events.append({
                            "date": dates[i],
                            "event": "long_entry",
                            "side": "long",
                            "price": float(price_long),
                            "underlying": float(S),
                            "strike": float(S),
                            "contracts": int(contracts),
                            "cash_flow": -(price_long * 100.0 * contracts + open_fee),
                            "reason": "Signal",
                        })

        # --- POSITION MANAGEMENT ---
        if has_long and long_contracts > 0:
            # Decide short-call behavior (Roll / No Roll / ASL)
            allow_short = False
            otm_for_short = None
            pct_for_short = 0.5 if np.isnan(pct) else pct

            if short_mode == "Roll (always)":
                allow_short = True
                otm_for_short = otm_active
            elif short_mode == "No Roll (exit & wait)":
                if weeks_since_long_entry == 0:
                    allow_short = True
                    otm_for_short = otm_active
            elif short_mode == "ASL (adaptive)":
                if pct_for_short < 0.60:
                    allow_short = True
                    otm_for_short = otm_active
                elif pct_for_short < 0.80:
                    allow_short = True
                    otm_for_short = otm_active * 1.5
                else:
                    allow_short = False
                    otm_for_short = None

            # Weekly SHORT call
            if allow_short:
                T_short = 1.0 / 52.0
                K_short = S + otm_for_short
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

                    events.append({
                        "date": dates[i],
                        "event": "short_entry",
                        "side": "short",
                        "price": float(price_short),
                        "underlying": float(S),
                        "strike": float(K_short),
                        "contracts": int(long_contracts),
                        "cash_flow": short_prem - short_open_fee,
                        "reason": f"Weekly income ({short_mode})",
                    })

                    payoff_short = max(S_next - K_short, 0.0) * 100.0 * long_contracts
                    short_close_fee = fee_per_contract * long_contracts
                    cash -= payoff_short
                    cash -= short_close_fee

                    pnl_short = short_prem - payoff_short - short_open_fee - short_close_fee
                    realized_this_week += pnl_short

                    events.append({
                        "date": dates[i+1],
                        "event": "short_exit",
                        "side": "short",
                        "price": 0.0,
                        "underlying": float(S_next),
                        "strike": float(K_short),
                        "contracts": int(long_contracts),
                        "cash_flow": - (payoff_short + short_close_fee),
                        "pnl_leg": float(pnl_short),
                        "reason": "Weekly expiry/exercise",
                    })

            # Reprice long leg
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            weeks_since_long_entry += 1
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
            exit_reason = "Other"
            if T_new <= 0:
                exit_now = True
                exit_reason = "Expiry"
            elif price_long_new >= tgt_active * long_entry_price:
                exit_now = True
                exit_reason = "TP"

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

                pnl_dollars = float(realized_long)
                pnl_pct = pnl_dollars / float(long_cost_basis) if long_cost_basis > 0 else np.nan

                trades.append({
                    "structure": "Diagonal",
                    "entry_date": dates[current_trade_open_idx],
                    "exit_date": dates[i+1],
                    "entry_price": float(long_entry_price),
                    "exit_price": float(price_long_new),
                    "contracts": int(long_contracts),
                    "direction": "long",
                    "reason_exit": exit_reason,
                    "pnl_dollars": pnl_dollars,
                    "pnl_pct": pnl_pct,
                    "duration_weeks": float(dur_weeks),
                })

                events.append({
                    "date": dates[i+1],
                    "event": "long_exit",
                    "side": "long",
                    "price": float(price_long_new),
                    "underlying": float(S_next),
                    "strike": float(S),
                    "contracts": int(long_contracts),
                    "cash_flow": new_val - close_fee,
                    "reason": exit_reason,
                })

                has_long = False
                long_contracts = 0
                long_value = 0.0
                long_cost_basis = 0.0
                current_trade_open_idx = None
                weeks_since_long_entry = 0
            else:
                long_value = new_val

        # --- Weekly bookkeeping ---
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
        trades=trades,
        events=events,
    )


# --------------------------------------------------------
# Long-only backtest (with trade/event logs)
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

    trades = []
    events = []

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

                        events.append({
                            "date": dates[i],
                            "event": "long_entry",
                            "side": "long",
                            "price": float(price_long),
                            "underlying": float(S),
                            "strike": float(S),
                            "contracts": int(contracts),
                            "cash_flow": -(price_long * 100.0 * contracts + open_fee),
                            "reason": "Signal",
                        })

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
            exit_reason = "Other"
            if price_long_new >= target_mult * long_entry_price:
                exit_now = True
                exit_reason = "TP"
            if T_new <= 0:
                exit_now = True
                exit_reason = "Expiry"
            if exit_mode == "TP + stop":
                if price_long_new <= stop_mult * long_entry_price:
                    exit_now = True
                    exit_reason = "Stop"
            if exit_mode == "Percentile exit" and not np.isnan(pct):
                if pct <= exit_pct_threshold:
                    exit_now = True
                    exit_reason = "Pct exit"

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

                pnl_dollars = float(realized_long)
                pnl_pct = pnl_dollars / float(long_cost_basis) if long_cost_basis > 0 else np.nan

                trades.append({
                    "structure": "Long-only",
                    "entry_date": dates[current_trade_open_idx],
                    "exit_date": dates[i+1],
                    "entry_price": float(long_entry_price),
                    "exit_price": float(price_long_new),
                    "contracts": int(long_contracts),
                    "direction": "long",
                    "reason_exit": exit_reason,
                    "pnl_dollars": pnl_dollars,
                    "pnl_pct": pnl_pct,
                    "duration_weeks": float(dur_weeks),
                })

                events.append({
                    "date": dates[i+1],
                    "event": "long_exit",
                    "side": "long",
                    "price": float(price_long_new),
                    "underlying": float(S_next),
                    "strike": float(S),
                    "contracts": int(long_contracts),
                    "cash_flow": new_val - close_fee,
                    "reason": exit_reason,
                })

                has_long = False
                long_contracts = 0
                long_value = 0.0
                long_cost_basis = 0.0
                current_trade_open_idx = None
            else:
                long_value = new_val

        # BOOKKEEPING
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
        trades=trades,
        events=events,
    )


# --------------------------------------------------------
# Grid scans (same logic as previous v12)
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
                            short_mode="ASL (adaptive)",
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
    df["score"] = df.apply(score_row, axis1)


