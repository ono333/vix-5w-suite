#!/usr/bin/env python3
"""
VIX 5% Weekly – Diagonal & Long-only Backtester with Trade Explorer

Features
--------
- Diagonal (LEAP-style long call + weekly short OTM calls)
- Long-only VIX call strategy
- Short-call modes: Roll / No Roll / ASL (adaptive by percentile)
- Realistic pricing (haircut on Black–Scholes)
- Weekly VIX data + 52-week percentile
- Grid scans (basic) for diagonal + long-only
- Trade Explorer:
    * VIX line (dark)
    * Long entries / exits
    * Short entries / exits (win/loss)
    * Date zoom
    * Trade tables
"""

import io
import datetime as dt
from math import erf, sqrt
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf


# =========================================================
# Math & pricing helpers
# =========================================================

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


def realism_haircut_factor(pct: float, realism_level: str = "Normal") -> float:
    """
    Haircut factor on theoretical BS price based on VIX percentile.
    pct: 0–1, realism_level: "Optimistic", "Normal", "Conservative".
    """
    if np.isnan(pct):
        pct = 0.5
    if realism_level == "Optimistic":
        base = 0.75
    elif realism_level == "Conservative":
        base = 0.45
    else:
        base = 0.60

    # Calm regimes: VIX low → overpricing worse, reduce haircut.
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
    """Black–Scholes call with optional realism haircut."""
    bs_price = black_scholes_call(S_spot, K, T, r, sigma_base)
    if not realistic_mode or T <= 0:
        return bs_price
    factor = realism_haircut_factor(vix_pct, realism_level)
    return bs_price * factor


# =========================================================
# Regime table (for Dynamic mode, optional)
# =========================================================

def default_regime_schedule() -> pd.DataFrame:
    rows = [
        {"p_min": 0.00, "p_max": 0.40, "long_dte_weeks": 40, "otm_pts": 7.0, "target_mult": 1.30, "sigma_mult": 1.2},
        {"p_min": 0.40, "p_max": 0.70, "long_dte_weeks": 26, "otm_pts": 5.0, "target_mult": 1.20, "sigma_mult": 1.0},
        {"p_min": 0.70, "p_max": 0.90, "long_dte_weeks": 20, "otm_pts": 4.0, "target_mult": 1.15, "sigma_mult": 0.8},
        {"p_min": 0.90, "p_max": 0.97, "long_dte_weeks": 13, "otm_pts": 3.0, "target_mult": 1.10, "sigma_mult": 0.6},
        {"p_min": 0.97, "p_max": 1.00, "long_dte_weeks": 8,  "otm_pts": 2.0, "target_mult": 1.05, "sigma_mult": 0.5},
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


# =========================================================
# Parsing helpers
# =========================================================

def parse_float_list(text: str, default_vals: List[float]) -> List[float]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


def parse_int_list(text: str, default_vals: List[int]) -> List[int]:
    try:
        vals = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


# =========================================================
# Diagonal backtest (LEAP + weekly short)
# =========================================================

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
    Diagonal strategy with trade + event logs for plotting.
    short_mode: "Roll (always)", "No Roll (exit & wait)", "ASL (adaptive)".
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
    durations: List[int] = []
    current_trade_open_idx: Optional[int] = None
    weeks_since_long_entry = 0

    trades: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]
        pct = float(vix_pct_series.iloc[i]) if not np.isnan(vix_pct_series.iloc[i]) else np.nan

        # Regime-adjusted diagonal parameters
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

        # ----- Long entry -----
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

        # ----- Position management (long + short) -----
        if has_long and long_contracts > 0:
            # Short-call decision
            allow_short = False
            otm_for_short: Optional[float] = None
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

            # Weekly short call
            if allow_short and otm_for_short is not None:
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
                        "cash_flow": -(payoff_short + short_close_fee),
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

        # ----- Bookkeeping -----
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
            weekly_ret[i + 1:] = -1.0
            weekly_pnl_pct[i + 1:] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = calc_cagr(total_return, n_valid)
    valid_rets = weekly_ret[1:n_valid + 1]
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


# =========================================================
# Long-only backtest
# =========================================================

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
    current_trade_open_idx: Optional[int] = None
    durations: List[int] = []

    trades: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

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
            weekly_ret[i + 1:] = -1.0
            weekly_pnl_pct[i + 1:] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = calc_cagr(total_return, n_valid)
    valid_rets = weekly_ret[1:n_valid + 1]
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


# =========================================================
# Grid scans
# =========================================================

def score_row(row: pd.Series, dd_penalty: float = 2.0) -> float:
    return float(row["cagr"] - dd_penalty * abs(row["max_dd"]))


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
    rows: List[Dict[str, Any]] = []
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
                        if not res:
                            continue
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
    return df.sort_values("score", ascending=False)


def grid_scan_long_only(
    vix_weekly: pd.Series,
    vix_pct_series: pd.Series,
    initial_capital: float,
    alloc_pct: float,
    r: float,
    fee_per_contract: float,
    entry_percentiles,
    target_mults,
    sigma_mults,
    long_dtes,
    exit_modes,
    stop_mults,
    exit_pcts,
    entry_lookback_weeks: int,
    realistic_mode: bool,
    realism_level: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dte in long_dtes:
        for ep in entry_percentiles:
            for tgt in target_mults:
                for sig in sigma_mults:
                    for exm in exit_modes:
                        for sm in stop_mults:
                            for ex_pct in exit_pcts:
                                res = backtest_long_only(
                                    vix_weekly=vix_weekly,
                                    vix_pct_series=vix_pct_series,
                                    mode="Static",
                                    exit_mode=exm,
                                    initial_capital=initial_capital,
                                    alloc_pct=alloc_pct,
                                    entry_lookback_weeks=entry_lookback_weeks,
                                    entry_percentile=ep,
                                    long_dte_weeks=int(dte),
                                    target_mult=tgt,
                                    sigma_mult=sig,
                                    stop_mult=sm,
                                    exit_pct_threshold=ex_pct,
                                    r=r,
                                    fee_per_contract=fee_per_contract,
                                    realistic_mode=realistic_mode,
                                    realism_level=realism_level,
                                )
                                if not res:
                                    continue
                                rows.append(
                                    dict(
                                        entry_pct=ep,
                                        target_mult=tgt,
                                        sigma_mult=sig,
                                        long_dte_weeks=int(dte),
                                        exit_mode=exm,
                                        stop_mult=sm,
                                        exit_pct=ex_pct,
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
    return df.sort_values("score", ascending=False)


# =========================================================
# Data loading
# =========================================================

@st.cache_data(show_spinner=False)
def load_vix_weekly() -> pd.Series:
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 20)
    data = yf.download("^VIX", start=start, end=end, auto_adjust=False)
    if data.empty:
        raise RuntimeError("Failed to download ^VIX from yfinance.")
    w = data["Close"].resample("W-FRI").last().dropna()
    return w


@st.cache_data(show_spinner=False)
def compute_vix_percentile(vix_weekly: pd.Series, lookback_weeks: int = 52) -> pd.Series:
    # In case a DataFrame sneaks in, take the first column
    if isinstance(vix_weekly, pd.DataFrame):
        vix_weekly = vix_weekly.iloc[:, 0]

    # Ensure 1-D array
    vals = np.asarray(vix_weekly, dtype=float).reshape(-1)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(len(vals)):
        start = max(0, i - lookback_weeks + 1)
        window = vals[start : i + 1]
        if window.size >= 4:
            out[i] = (window <= vals[i]).mean()

    return pd.Series(out, index=vix_weekly.index, name="VIX_pct_52w")

# =========================================================
# Streamlit App
# =========================================================

st.set_page_config(
    page_title="VIX 5% Weekly – Trade Explorer",
    layout="wide",
)

st.title("VIX 5% Weekly – Diagonal & Long-only with Trade Explorer")

# Sidebar: global settings
st.sidebar.header("Global Settings")

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=10_000.0,
    max_value=10_000_000.0,
    value=250_000.0,
    step=10_000.0,
    format="%.0f",
)
alloc_pct = st.sidebar.number_input(
    "Fraction of equity per trade",
    min_value=0.001,
    max_value=0.50,
    value=0.01,  # 1%
    step=0.001,
    format="%.3f",
)
fee_per_contract = st.sidebar.number_input(
    "Fee per contract ($, per leg)",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.25,
    format="%.2f",
)
risk_free = st.sidebar.number_input(
    "Risk-free annual rate (for Sharpe)",
    min_value=-0.02,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f",
)
realistic_mode = st.sidebar.checkbox(
    "Realistic pricing mode (haircut BS)",
    value=True,
)
realism_level = st.sidebar.selectbox(
    "Realism level",
    ["Optimistic", "Normal", "Conservative"],
    index=1,
)
entry_lookback_weeks = st.sidebar.slider(
    "Percentile lookback window (weeks)",
    min_value=26,
    max_value=156,
    value=52,
    step=1,
)

page = st.sidebar.radio(
    "Page",
    ["Dashboard", "Single Backtest", "Grid Scans", "Regime Engine", "Trade Explorer", "Playbook / Help"],
    index=0,
)

with st.spinner("Loading VIX data..."):
    vix_weekly = load_vix_weekly()
    vix_pct_series = compute_vix_percentile(vix_weekly, lookback_weeks=entry_lookback_weeks)


# =========================================================
# Page: Dashboard
# =========================================================

if page == "Dashboard":
    st.subheader("Dashboard – baseline diagonal (ASL) vs long-only")

    mode = st.selectbox(
        "Entry mode",
        ["Static", "Static 90%-only", "Dynamic (regime)"],
        index=2,
    )
    regime_df = default_regime_schedule() if mode == "Dynamic (regime)" else None

    diag_res = backtest_diagonal(
        vix_weekly=vix_weekly,
        vix_pct_series=vix_pct_series,
        mode=mode,
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
        entry_lookback_weeks=entry_lookback_weeks,
        entry_percentile=0.10,
        long_dte_weeks=26,
        otm_pts=3.0,
        target_mult=1.20,
        sigma_mult=1.0,
        r=risk_free,
        fee_per_contract=fee_per_contract,
        realistic_mode=realistic_mode,
        realism_level=realism_level,
        regime_df=regime_df,
        short_mode="ASL (adaptive)",
    )
    long_res = backtest_long_only(
        vix_weekly=vix_weekly,
        vix_pct_series=vix_pct_series,
        mode=mode,
        exit_mode="TP + stop",
        initial_capital=initial_capital,
        alloc_pct=alloc_pct,
        entry_lookback_weeks=entry_lookback_weeks,
        entry_percentile=0.10,
        long_dte_weeks=3,
        target_mult=1.20,
        sigma_mult=0.3,
        stop_mult=0.5,
        exit_pct_threshold=0.5,
        r=risk_free,
        fee_per_contract=fee_per_contract,
        realistic_mode=realistic_mode,
        realism_level=realism_level,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Diagonal (ASL)")
        st.write(
            f"CAGR: {diag_res['cagr']*100:,.2f}%  \n"
            f"Sharpe: {diag_res['sharpe']:.2f}  \n"
            f"Max DD: {diag_res['max_dd']*100:,.2f}%  \n"
            f"Trades: {diag_res['trade_count']}"
        )
    with c2:
        st.markdown("### Long-only")
        st.write(
            f"CAGR: {long_res['cagr']*100:,.2f}%  \n"
            f"Sharpe: {long_res['sharpe']:.2f}  \n"
            f"Max DD: {long_res['max_dd']*100:,.2f}%  \n"
            f"Trades: {long_res['trade_count']}"
        )

    fig, ax = plt.subplots()
    ax.plot(diag_res["dates"], diag_res["equity"], label="Diagonal (ASL)")
    ax.plot(long_res["dates"], long_res["equity"], label="Long-only")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Equity Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.caption("Use Single Backtest, Grid Scans, and Trade Explorer for deeper analysis.")


# =========================================================
# Page: Single Backtest
# =========================================================

elif page == "Single Backtest":
    st.subheader("Single Backtest")

    mode = st.selectbox(
        "Entry mode",
        ["Static", "Static 90%-only", "Dynamic (regime)"],
        index=0,
    )
    structure = st.radio("Structure", ["Diagonal", "Long-only"], index=0, horizontal=True)
    entry_percentile = st.slider(
        "Static entry percentile (only used in Static mode)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01,
    )
    regime_df = default_regime_schedule() if mode == "Dynamic (regime)" else None

    if structure == "Diagonal":
        short_mode = st.selectbox(
            "Short-call mode",
            ["Roll (always)", "No Roll (exit & wait)", "ASL (adaptive)"],
            index=2,
        )
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
            entry_lookback_weeks=entry_lookback_weeks,
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
            short_mode=short_mode,
        )
    else:
        exit_mode = st.selectbox("Exit mode", ["TP only", "TP + stop", "Percentile exit"], index=1)
        long_dte_weeks = st.slider("Long-only DTE (weeks)", 1, 26, 3, 1)
        target_mult = st.slider("Target multiple", 1.05, 3.0, 1.20, 0.05)
        sigma_mult = st.slider("Sigma multiplier", 0.1, 2.0, 0.3, 0.05)
        stop_mult = st.slider("Stop multiple (for TP + stop)", 0.1, 0.9, 0.50, 0.05)
        exit_pct_threshold = st.slider("Exit percentile (for Percentile exit)", 0.0, 1.0, 0.5, 0.05)

        res = backtest_long_only(
            vix_weekly=vix_weekly,
            vix_pct_series=vix_pct_series,
            mode=mode,
            exit_mode=exit_mode,
            initial_capital=initial_capital,
            alloc_pct=alloc_pct,
            entry_lookback_weeks=entry_lookback_weeks,
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
        st.markdown("### Performance Metrics")
        st.write(
            f"Total return: {res['total_return']*100:,.2f}%  \n"
            f"CAGR: {res['cagr']*100:,.2f}%  \n"
            f"Sharpe: {res['sharpe']:.2f}  \n"
            f"Max drawdown: {res['max_dd']*100:,.2f}%  \n"
            f"Win rate: {res['win_rate']*100:,.1f}%  \n"
            f"Trades: {res['trade_count']}  \n"
            f"Avg duration: {res['avg_trade_duration']:.1f} weeks"
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.plot(res["dates"], res["equity"])
            ax.set_title("Equity Curve")
            ax.set_ylabel("Equity ($)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        with c2:
            fig2, ax2 = plt.subplots()
            ax2.bar(res["dates"], res["realized_pnl_weekly"])
            ax2.set_title("Weekly Realized PnL")
            ax2.set_ylabel("PnL ($)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

        st.caption("Use Trade Explorer to inspect exact timing on the VIX chart.")


# =========================================================
# Page: Grid Scans
# =========================================================

elif page == "Grid Scans":
    st.subheader("Grid Scans (Diagonal ASL & Long-only)")
    tab1, tab2 = st.tabs(["Diagonal (ASL)", "Long-only"])

    with tab1:
        st.markdown("#### Diagonal grid (ASL short mode)")
        entry_grid = parse_float_list(st.text_input("Entry percentiles", "0.05, 0.10, 0.15"), [0.10])
        otm_grid = parse_float_list(st.text_input("OTM distances (pts)", "3, 5, 7"), [3.0])
        target_grid = parse_float_list(st.text_input("Target multiples", "1.15, 1.20, 1.25"), [1.20])
        sigma_grid = parse_float_list(st.text_input("Sigma multipliers", "0.8, 1.0, 1.2"), [1.0])
        dte_grid = parse_int_list(st.text_input("Long DTE (weeks)", "13, 26, 39"), [26])

        if st.button("Run diagonal grid scan"):
            with st.spinner("Running diagonal grid..."):
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
                    entry_lookback_weeks=entry_lookback_weeks,
                    realistic_mode=realistic_mode,
                    realism_level=realism_level,
                )
            if df_diag.empty:
                st.warning("No results.")
            else:
                st.dataframe(df_diag.head(50), use_container_width=True)
                buf = io.BytesIO()
                df_diag.to_excel(buf, index=False)
                st.download_button(
                    "Download Diagonal Grid (.xlsx)",
                    buf.getvalue(),
                    file_name="vix_diagonal_grid.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tab2:
        st.markdown("#### Long-only grid")
        entry_grid_lo = parse_float_list(st.text_input("Entry percentiles (long-only)", "0.05, 0.10, 0.15"), [0.10])
        target_grid_lo = parse_float_list(st.text_input("Target multiples (long-only)", "1.15, 1.20, 1.25"), [1.20])
        sigma_grid_lo = parse_float_list(st.text_input("Sigma multipliers (long-only)", "0.3, 0.5"), [0.3])
        dte_grid_lo = parse_int_list(st.text_input("Long DTE (weeks, long-only)", "2, 3, 4"), [3])
        exit_modes_lo = ["TP + stop"]
        stop_mults_lo = parse_float_list(st.text_input("Stop multiples (long-only)", "0.5, 0.6"), [0.5])
        exit_pcts_lo = parse_float_list(st.text_input("Exit percentiles (for Percentile exit)", "0.3, 0.5"), [0.5])

        if st.button("Run long-only grid scan"):
            with st.spinner("Running long-only grid..."):
                df_lo = grid_scan_long_only(
                    vix_weekly=vix_weekly,
                    vix_pct_series=vix_pct_series,
                    initial_capital=initial_capital,
                    alloc_pct=alloc_pct,
                    r=risk_free,
                    fee_per_contract=fee_per_contract,
                    entry_percentiles=entry_grid_lo,
                    target_mults=target_grid_lo,
                    sigma_mults=sigma_grid_lo,
                    long_dtes=dte_grid_lo,
                    exit_modes=exit_modes_lo,
                    stop_mults=stop_mults_lo,
                    exit_pcts=exit_pcts_lo,
                    entry_lookback_weeks=entry_lookback_weeks,
                    realistic_mode=realistic_mode,
                    realism_level=realism_level,
                )
            if df_lo.empty:
                st.warning("No results.")
            else:
                st.dataframe(df_lo.head(50), use_container_width=True)
                buf = io.BytesIO()
                df_lo.to_excel(buf, index=False)
                st.download_button(
                    "Download Long-only Grid (.xlsx)",
                    buf.getvalue(),
                    file_name="vix_longonly_grid.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# =========================================================
# Page: Regime Engine
# =========================================================

elif page == "Regime Engine":
    st.subheader("Regime Engine (Diagonal Dynamic mode)")
    st.dataframe(default_regime_schedule(), use_container_width=True)
    st.caption(
        "In Dynamic (regime) mode, this table sets long DTE, OTM distance, target, "
        "and sigma multiplier as functions of VIX percentile."
    )


# =========================================================
# Page: Trade Explorer
# =========================================================

elif page == "Trade Explorer":
    st.subheader("Trade Explorer – timing & pattern view")

    structure = st.radio("Structure", ["Diagonal", "Long-only"], index=0, horizontal=True)
    mode = st.selectbox(
        "Entry mode",
        ["Static", "Static 90%-only", "Dynamic (regime)"],
        index=2,
    )
    regime_df = default_regime_schedule() if mode == "Dynamic (regime)" else None
    entry_percentile = st.slider(
        "Static entry percentile (only used in Static mode)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01,
    )

    if structure == "Diagonal":
        short_mode = st.selectbox(
            "Short-call mode",
            ["Roll (always)", "No Roll (exit & wait)", "ASL (adaptive)"],
            index=2,
        )
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
            entry_lookback_weeks=entry_lookback_weeks,
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
            short_mode=short_mode,
        )
    else:
        exit_mode = st.selectbox("Exit mode (long-only)", ["TP only", "TP + stop", "Percentile exit"], index=1)
        long_dte_weeks = st.slider("Long-only DTE (weeks)", 1, 26, 3, 1)
        target_mult = st.slider("Target multiple", 1.05, 3.0, 1.20, 0.05)
        sigma_mult = st.slider("Sigma multiplier", 0.1, 2.0, 0.3, 0.05)
        stop_mult = st.slider("Stop multiple (for TP + stop)", 0.1, 0.9, 0.50, 0.05)
        exit_pct_threshold = st.slider("Exit percentile (for Percentile exit)", 0.0, 1.0, 0.5, 0.05)

        res = backtest_long_only(
            vix_weekly=vix_weekly,
            vix_pct_series=vix_pct_series,
            mode=mode,
            exit_mode=exit_mode,
            initial_capital=initial_capital,
            alloc_pct=alloc_pct,
            entry_lookback_weeks=entry_lookback_weeks,
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
        events = pd.DataFrame(res.get("events", []))
        trades = pd.DataFrame(res.get("trades", []))
        if events.empty:
            st.info("No trades/events recorded for this configuration.")
        else:
            st.markdown("### Zoom window")
            all_dates = events["date"].sort_values().unique()
            min_date = pd.to_datetime(all_dates[0]).date()
            max_date = pd.to_datetime(all_dates[-1]).date()

            c1, c2 = st.columns(2)
            with c1:
                start_zoom = st.date_input("Start date", value=max(min_date, max_date - dt.timedelta(days=365)))
            with c2:
                end_zoom = st.date_input("End date", value=max_date)

            start_zoom_dt = pd.to_datetime(start_zoom)
            end_zoom_dt = pd.to_datetime(end_zoom)

            mask = (events["date"] >= start_zoom_dt) & (events["date"] <= end_zoom_dt)
            ev_zoom = events[mask].copy()
            vix_z = vix_weekly.loc[start_zoom_dt:end_zoom_dt]

            # Plot VIX + trade markers
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(vix_z.index, vix_z.values, linewidth=1.0, label="VIX")

            le = ev_zoom[ev_zoom["event"] == "long_entry"]
            lx = ev_zoom[ev_zoom["event"] == "long_exit"]
            se = ev_zoom[ev_zoom["event"] == "short_entry"]
            sx = ev_zoom[ev_zoom["event"] == "short_exit"]

            if not le.empty:
                ax.scatter(le["date"], le["underlying"], marker="^", s=60, label="Long Entry")
            if not lx.empty:
                ax.scatter(lx["date"], lx["underlying"], marker="v", s=60, label="Long Exit")
            if not se.empty:
                ax.scatter(se["date"], se["underlying"], marker="o", s=35, label="Short Entry")
            if not sx.empty:
                sx = sx.copy()
                sx["pnl_leg"] = sx.get("pnl_leg", 0.0).astype(float)
                winners = sx[sx["pnl_leg"] >= 0]
                losers = sx[sx["pnl_leg"] < 0]
                if not winners.empty:
                    ax.scatter(winners["date"], winners["underlying"], marker="o", s=40, label="Short Exit (win)")
                if not losers.empty:
                    ax.scatter(losers["date"], losers["underlying"], marker="x", s=50, label="Short Exit (loss)")

            ax.set_ylabel("VIX")
            ax.set_title("VIX with long & short entries/exits")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
            st.pyplot(fig, clear_figure=True)
            plt.style.use("default")

            c3, c4 = st.columns(2)
            with c3:
                if not trades.empty:
                    st.markdown("#### Long trades (per position)")
                    st.dataframe(
                        trades[
                            ["entry_date", "exit_date", "pnl_dollars", "pnl_pct", "duration_weeks", "reason_exit"]
                        ].sort_values("entry_date"),
                        use_container_width=True,
                    )
                else:
                    st.info("No long trades recorded.")
            with c4:
                if not sx.empty:
                    st.markdown("#### Short weekly legs (exits)")
                    sx_show = sx[["date", "underlying", "strike", "contracts", "pnl_leg", "reason"]].sort_values("date")
                    st.dataframe(sx_show, use_container_width=True)
                else:
                    st.info("No short-call legs (e.g. Long-only or ASL filtered them out).")

            st.caption(
                "Zoom into regimes, look at where short-call losses cluster relative to VIX spikes, "
                "and decide if the execution looks realistically tradeable."
            )

# ================================
# XLSX Export Buttons
# ================================

import io

st.markdown("### Export Trade Data")

if not trades.empty:
    long_buf = io.BytesIO()
    trades.to_excel(long_buf, index=False)
    st.download_button(
        "Download Long Trades (TradeExplorer_Long.xlsx)",
        data=long_buf.getvalue(),
        file_name="TradeExplorer_Long.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if not sx.empty:
    short_buf = io.BytesIO()
    sx_show.to_excel(short_buf, index=False)
    st.download_button(
        "Download Short Trades (TradeExplorer_Short.xlsx)",
        data=short_buf.getvalue(),
        file_name="TradeExplorer_Short.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Optional: combined export
if (not trades.empty) and (not sx.empty):
    both_buf = io.BytesIO()
    with pd.ExcelWriter(both_buf, engine="openpyxl") as writer:
        trades.to_excel(writer, sheet_name="Long", index=False)
        sx_show.to_excel(writer, sheet_name="Short", index=False)

    st.download_button(
        "Download Combined (Long_and_Short.xlsx)",
        data=both_buf.getvalue(),
        file_name="TradeExplorer_Long_and_Short.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
# =========================================================
# Page: Playbook / Help
# =========================================================

elif page == "Playbook / Help":
    st.subheader("Playbook / Help")
    st.markdown(
        """
### Strategy summary

- **Long-only**  
  Buy VIX calls when entry triggers (percentile / regime). Exit on TP / stop / expiry / percentile.

- **Diagonal**  
  Hold a longer-dated call (LEAP-style) and sell weekly OTM calls against it.

Short-call modes:

- **Roll (always)** – continuous weekly shorts (good in calm, dangerous in spikes).  
- **No Roll (exit & wait)** – only short in week 0 of each long.  
- **ASL (adaptive)** – short in calm regimes, stop selling above certain percentiles.

### Trade Explorer

Use **Trade Explorer** to:

- See long & short timing on the VIX chart.
- Identify patterns: do losers cluster in specific spike shapes?
- Decide whether the strategy requires knife-edge timing, or works on multi-week moves.
"""
    )