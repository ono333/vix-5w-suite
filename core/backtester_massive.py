#!/usr/bin/env python3
"""
Realistic VIX / UVXY 5-Weekly backtester using Massive historical option chains.

- Uses Massive for:
    * Real historical option prices (bid/ask/mid)
    * True expirations and DTE
    * IV surface (implicitly via the chain mid prices)

- Mirrors the same position logic as the synthetic engine:
    * mode == "diagonal": long LEAP call + short weekly call
    * mode == "long_only": long call only

Extra realism / safety:
    - alloc_pct is interpreted like the synthetic engine:
        * alloc_pct > 1.0  => treated as a percent (e.g. 50 -> 0.50)
        * alloc_pct <= 1.0 => used as-is (e.g. 0.01 = 1%)
    - For decaying vol ETPs (UVXY, VXX), long DTE is capped to 13 weeks (~3 months).
    - Equity is never allowed to go negative; once at 0 we stay flat.
    - Per-contract fees and a "realism" haircut multiply all cashflows.

Optional:
    - progress_cb(cur_step, total_steps): callback used by the UI
      (e.g. Streamlit progress bar) to show backtest progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .massive_client import get_option_chain, load_massive_config


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class OptionPosition:
    symbol: str
    quantity: int
    strike: float
    expiration: pd.Timestamp
    option_type: str  # "C" or "P"
    is_long: bool


# ---------------------------------------------------------------------
# Helper: select contracts from Massive chain
# ---------------------------------------------------------------------


def _select_long_call(
    chain: pd.DataFrame,
    underlying_px: float,
    trade_date: pd.Timestamp,
    target_dte_weeks: int,
    otm_pts: float,
) -> Optional[pd.Series]:
    """
    Pick a LEAP-like long call:
        - expiration ~ target_dte_weeks (within a window)
        - strike >= underlying + otm_pts (slightly OTM)
        - choose by closeness to desired strike / DTE
    """
    if chain.empty:
        return None

    chain = chain.loc[chain["option_type"] == "C"].copy()
    if chain.empty:
        return None

    target_dte_days = target_dte_weeks * 7
    dte = (chain["expiration"] - trade_date).dt.days
    chain = chain.assign(dte=dte)
    # ensure not near expiry
    chain = chain[chain["dte"] > 7]
    if chain.empty:
        return None

    # DTE window: +/- 4 weeks around target
    lb = target_dte_days - 28
    ub = target_dte_days + 28
    chain = chain[(chain["dte"] >= lb) & (chain["dte"] <= ub)]
    if chain.empty:
        return None

    desired_strike = underlying_px + otm_pts
    chain = chain.assign(
        strike_diff=(chain["strike"] - desired_strike).abs(),
    )

    # Prefer strikes >= desired, but fallback if none
    chain_ge = chain[chain["strike"] >= desired_strike]
    if not chain_ge.empty:
        chain = chain_ge

    # Sort: closest strike, then closest DTE, then mid
    chain = chain.sort_values(["strike_diff", "dte", "mid"])
    return chain.iloc[0]


def _select_short_call_weekly(
    chain: pd.DataFrame,
    underlying_px: float,
    trade_date: pd.Timestamp,
    otm_pts: float,
) -> Optional[pd.Series]:
    """
    Pick a short weekly call:
        - nearest expiration  ~1 week (5â€“12 days)
        - strike >= underlying + otm_pts
    """
    if chain.empty:
        return None

    calls = chain.loc[chain["option_type"] == "C"].copy()
    if calls.empty:
        return None

    dte = (calls["expiration"] - trade_date).dt.days
    calls = calls.assign(dte=dte)
    calls = calls[(calls["dte"] >= 5) & (calls["dte"] <= 12)]
    if calls.empty:
        return None

    desired_strike = underlying_px + otm_pts
    calls = calls.assign(
        strike_diff=(calls["strike"] - desired_strike).abs(),
    )

    calls_ge = calls[calls["strike"] >= desired_strike]
    if not calls_ge.empty:
        calls = calls_ge

    calls = calls.sort_values(["dte", "strike_diff", "mid"])
    return calls.iloc[0]


def _price_position(
    chain: pd.DataFrame,
    positions: List[OptionPosition],
    trade_date: pd.Timestamp,
    slippage_bps: float = 5.0,
) -> float:
    """
    Mark-to-market of all open positions using Massive mid prices, minus slippage.

    slippage_bps: 5 => 0.05% extra disadvantage on mid for each trade.
    """
    if not positions:
        return 0.0

    if chain.empty:
        # no fresh quotes -> treat as zero extra PnL for this week
        return 0.0

    total_val = 0.0
    for pos in positions:
        mask = (
            (chain["option_type"] == pos.option_type)
            & (chain["strike"] == pos.strike)
            & (chain["expiration"] == pos.expiration)
        )
        sub = chain.loc[mask]
        if sub.empty:
            continue

        row = sub.iloc[0]
        mid = float(row.get("mid", np.nan))
        if not np.isfinite(mid):
            bid = float(row.get("bid", np.nan))
            ask = float(row.get("ask", np.nan))
            if np.isfinite(bid) and np.isfinite(ask):
                mid = 0.5 * (bid + ask)
            elif np.isfinite(bid):
                mid = bid
            elif np.isfinite(ask):
                mid = ask

        if not np.isfinite(mid) or mid <= 0.0:
            continue

        # Apply slippage against us
        if pos.is_long:
            px = mid * (1.0 - slippage_bps / 10_000.0)
        else:
            px = mid * (1.0 + slippage_bps / 10_000.0)

        total_val += pos.quantity * px * 100.0

    return total_val


# ---------------------------------------------------------------------
# Main Massive-based backtest
# ---------------------------------------------------------------------


def run_backtest_massive(
    vix_weekly: pd.Series,
    params: Dict[str, Any],
    *,
    symbol: str = "^VIX",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Realistic backtest using Massive historical options.

    vix_weekly: weekly underlying closes (index = weekly dates)
    params: same schema as synthetic run_backtest (alloc_pct, etc.)
    symbol: underlying symbol Massive uses (e.g. "^VIX", "UVXY", "VXX").
    progress_cb: optional callback (cur_step, total_steps) for UI progress.

    Returns same dict keys as synthetic engine:
        equity, weekly_returns, realized_weekly, unrealized_weekly,
        trades, win_rate, avg_trade_dur, trade_log
    """
    config = load_massive_config()

    # ----------------- basic params & safety transforms -----------------
    initial_cap = float(params.get("initial_capital", 250_000))

    # alloc_pct handled like synthetic:
    #   1.0 => 100%   0.01 => 1%    50 => 50%
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

    raw_long_dte_weeks = int(params.get("long_dte_weeks", 26))

    # Decay-aware DTE cap for UVXY / VXX
    underlying = symbol.upper()
    if underlying in ("UVXY", "VXX"):
        long_dte_weeks = min(raw_long_dte_weeks, 13)  # ~3 months max
    else:
        long_dte_weeks = raw_long_dte_weeks

    fee = float(params.get("fee_per_contract", 0.65))
    realism = float(params.get("realism", 1.0))
    slippage_bps = float(params.get("slippage_bps", 5.0))

    # risk-free rate is baked into option prices already (we don't reprice here)
    # Liquidity / risk caps
    MAX_QTY = 10_000
    LIQUIDATION_FLOOR = 0.0

    prices = vix_weekly.values.astype(float)
    dates = vix_weekly.index.to_list()
    n = len(prices)
    if n < entry_lb + 2:
        return {
            "equity": np.asarray([initial_cap], float),
            "weekly_returns": np.asarray([], float),
            "realized_weekly": np.asarray([], float),
            "unrealized_weekly": np.asarray([], float),
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
        }

    # ----------------- percentile series -----------------
    pct = np.full(n, np.nan, float)
    lb = max(1, entry_lb)
    for i in range(lb, n):
        w = prices[i - lb : i]
        pct[i] = (w < prices[i]).mean()

    # ----------------- state -----------------
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]
    trade_log: List[Dict[str, Any]] = []
    win_flags: List[bool] = []
    durations: List[int] = []

    have_pos = False
    long_pos: Optional[OptionPosition] = None
    short_pos: Optional[OptionPosition] = None
    entry_notional = 0.0
    entry_week_idx: Optional[int] = None

    total_steps = max(n - 1, 1)

    # -----------------------------------------------------------------
    # main loop
    # -----------------------------------------------------------------
    for i in range(1, n):
        # progress callback for UI (e.g. Streamlit progress bar)
        if progress_cb is not None:
            try:
                progress_cb(i, total_steps)
            except Exception:
                # Never let a UI callback crash the backtest
                pass

        S = float(prices[i])
        trade_date = pd.to_datetime(dates[i])
        prev_eq = float(equity[-1])

        # If we've been effectively wiped out and flat, stay flat
        if prev_eq <= LIQUIDATION_FLOOR and not have_pos:
            equity.append(LIQUIDATION_FLOOR)
            realized_weekly.append(0.0)
            unrealized_weekly.append(0.0)
            weekly_returns.append(0.0)
            continue

        # ----------------------------------------------------------
        # ENTRY
        # ----------------------------------------------------------
        if (not have_pos) and np.isfinite(pct[i]) and pct[i] <= entry_pct:
            capital = prev_eq * alloc_pct
            if capital <= 0.0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            try:
                chain = get_option_chain(
                    symbol=symbol,
                    trade_date=trade_date,
                    config=config,
                    use_cache=True,
                )
            except Exception:
                # network / DNS / Massive error -> skip this week
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            if chain.empty:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            long_row = _select_long_call(
                chain,
                underlying_px=S,
                trade_date=trade_date,
                target_dte_weeks=long_dte_weeks,
                otm_pts=otm_pts,
            )
            if long_row is None or not np.isfinite(long_row.get("mid", np.nan)):
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            long_mid = float(long_row["mid"])
            if not np.isfinite(long_mid) or long_mid <= 0.0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            # contract price adjusted for slippage
            long_px = long_mid * (1.0 + slippage_bps / 10_000.0)
            denom = long_px * 100.0
            if denom <= 0.0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            qty_float = capital / denom
            if not np.isfinite(qty_float):
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            qty = int(min(max(qty_float, 0.0), MAX_QTY))
            if qty <= 0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            long_pos = OptionPosition(
                symbol=symbol,
                quantity=qty,
                strike=float(long_row["strike"]),
                expiration=pd.to_datetime(long_row["expiration"]),
                option_type="C",
                is_long=True,
            )

            cost_long = qty * long_px * 100.0 + fee * qty

            credit_short = 0.0
            if mode == "diagonal":
                short_row = _select_short_call_weekly(
                    chain,
                    underlying_px=S,
                    trade_date=trade_date,
                    otm_pts=otm_pts,
                )
                if short_row is not None and np.isfinite(short_row.get("mid", np.nan)):
                    short_mid = float(short_row["mid"])
                    short_px = short_mid * (1.0 - slippage_bps / 10_000.0)

                    short_pos = OptionPosition(
                        symbol=symbol,
                        quantity=-qty,
                        strike=float(short_row["strike"]),
                        expiration=pd.to_datetime(short_row["expiration"]),
                        option_type="C",
                        is_long=False,
                    )

                    credit_short = qty * short_px * 100.0 - fee * qty
                else:
                    short_pos = None
            else:
                short_pos = None

            net_cash = (-cost_long + credit_short) * realism
            eq_now = prev_eq + net_cash

            have_pos = True
            entry_week_idx = i
            entry_notional = long_px * qty * 100.0

            equity.append(max(eq_now, LIQUIDATION_FLOOR))
            realized_weekly.append(net_cash)
            unrealized_weekly.append(0.0)
            weekly_returns.append((eq_now - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
            continue

        # ----------------------------------------------------------
        # POSITION MANAGEMENT
        # ----------------------------------------------------------
        if have_pos and long_pos is not None:
            try:
                chain = get_option_chain(
                    symbol=symbol,
                    trade_date=trade_date,
                    config=config,
                    use_cache=True,
                )
            except Exception:
                # If we can't reprice today, assume no PnL change
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue

            positions: List[OptionPosition] = [long_pos]
            if short_pos is not None:
                positions.append(short_pos)

            pos_val = _price_position(
                chain,
                positions,
                trade_date,
                slippage_bps=slippage_bps,
            )

            eq_now = prev_eq + pos_val
            unreal_pnl = pos_val

            # exit rules
            exit_trigger = False
            long_exp_days = (long_pos.expiration - trade_date).days

            if long_exp_days <= 0:
                exit_trigger = True

            if entry_notional > 0:
                if pos_val >= entry_notional * target_mult:
                    exit_trigger = True
                if pos_val <= entry_notional * exit_mult:
                    exit_trigger = True

            if exit_trigger:
                payoff = pos_val
                eq2 = prev_eq + payoff

                dur = i - (entry_week_idx if entry_week_idx is not None else i)
                durations.append(dur)
                win_flags.append(eq2 > prev_eq)

                trade_log.append(
                    {
                        "entry_idx": entry_week_idx,
                        "exit_idx": i,
                        "entry_date": dates[entry_week_idx]
                        if entry_week_idx is not None
                        else None,
                        "exit_date": dates[i],
                        "entry_notional": entry_notional,
                        "exit_value": payoff,
                        "duration_weeks": dur,
                        "strike_long": long_pos.strike,
                        "strike_short": short_pos.strike if short_pos is not None else None,
                    }
                )

                have_pos = False
                long_pos = None
                short_pos = None

                eq2_clamped = max(eq2, LIQUIDATION_FLOOR)
                equity.append(eq2_clamped)
                realized_weekly.append(payoff)
                unrealized_weekly.append(0.0)
                weekly_returns.append((eq2_clamped - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
                continue

            # hold
            eq_hold = prev_eq + unreal_pnl
            eq_hold_clamped = max(eq_hold, LIQUIDATION_FLOOR)
            equity.append(eq_hold_clamped)
            realized_weekly.append(0.0)
            unrealized_weekly.append(unreal_pnl)
            weekly_returns.append((eq_hold_clamped - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
        else:
            # flat
            equity.append(prev_eq)
            realized_weekly.append(0.0)
            unrealized_weekly.append(0.0)
            weekly_returns.append(0.0)

    equity_arr = np.asarray(equity, float)
    weekly_arr = np.asarray(weekly_returns, float)
    realized_arr = np.asarray(realized_weekly, float)
    unreal_arr = np.asarray(unrealized_weekly, float)

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
    }