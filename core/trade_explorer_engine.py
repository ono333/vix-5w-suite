"""
Trade Explorer Engine for VIX 5-Weekly
-------------------------------------

This module builds on the unified backtest engine in `core.backtester`.

It does NOT call any legacy V2/V3 functions directly.

API
----
run_trade_explorer(vix_weekly: pd.Series, params: dict) -> dict

Returns a dict with:
    - equity           : np.ndarray
    - weekly_returns   : np.ndarray
    - realized_weekly  : np.ndarray
    - unrealized_weekly: np.ndarray
    - win_rate         : float
    - trades           : int
    - avg_trade_dur    : float
    - trade_log        : list[dict]  (raw log from engine)
    - trades_df        : pd.DataFrame (for table/xlsx export)
    - entry_markers    : dict with arrays for plotting entry points
    - exit_markers     : dict with arrays for plotting exit points
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .backtester import run_backtest


def _build_trades_df(
    trade_log: list[dict],
    vix_weekly: pd.Series,
) -> pd.DataFrame:
    """
    Convert the raw trade_log into a DataFrame, and enrich with dates
    and underlying VIX at entry/exit.
    """
    if not trade_log:
        return pd.DataFrame(
            columns=[
                "entry_idx",
                "exit_idx",
                "entry_date",
                "exit_date",
                "entry_vix",
                "exit_vix",
                "entry_price",
                "exit_value",
                "pnl",
                "duration_weeks",
                "strike_long",
                "strike_short",
            ]
        )

    records = []
    prices = vix_weekly.values
    index = vix_weekly.index

    for tr in trade_log:
        e_idx = tr.get("entry_idx")
        x_idx = tr.get("exit_idx")

        if e_idx is None or x_idx is None:
            continue

        entry_date = index[e_idx] if 0 <= e_idx < len(index) else None
        exit_date = index[x_idx] if 0 <= x_idx < len(index) else None

        entry_vix = float(prices[e_idx]) if 0 <= e_idx < len(prices) else float("nan")
        exit_vix = float(prices[x_idx]) if 0 <= x_idx < len(prices) else float("nan")

        entry_price = tr.get("entry_price", 0.0)
        exit_value = tr.get("exit_value", 0.0)
        pnl = exit_value  # since we didn't store per-trade initial cost separately here

        records.append(
            {
                "entry_idx": e_idx,
                "exit_idx": x_idx,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_vix": entry_vix,
                "exit_vix": exit_vix,
                "entry_price": entry_price,
                "exit_value": exit_value,
                "pnl": pnl,
                "duration_weeks": tr.get("duration_weeks", 0),
                "strike_long": tr.get("strike_long"),
                "strike_short": tr.get("strike_short"),
            }
        )

    df = pd.DataFrame.from_records(records)
    # Sort by entry_date to keep things chronologically neat
    df.sort_values(by="entry_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _build_markers(
    trades_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Build entry/exit markers suitable for plotting on VIX chart.
    Returns:
        {
            "entry_dates": np.ndarray[datetime64],
            "entry_vix": np.ndarray[float],
            "exit_dates": np.ndarray[datetime64],
            "exit_vix": np.ndarray[float],
        }
    """
    if trades_df.empty:
        return {
            "entry_dates": np.array([]),
            "entry_vix": np.array([]),
            "exit_dates": np.array([]),
            "exit_vix": np.array([]),
        }

    entry_dates = trades_df["entry_date"].to_numpy()
    entry_vix = trades_df["entry_vix"].to_numpy(dtype=float)
    exit_dates = trades_df["exit_date"].to_numpy()
    exit_vix = trades_df["exit_vix"].to_numpy(dtype=float)

    return {
        "entry_dates": entry_dates,
        "entry_vix": entry_vix,
        "exit_dates": exit_dates,
        "exit_vix": exit_vix,
    }


def run_trade_explorer(
    vix_weekly: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    High-level Trade Explorer entrypoint.

    Uses the same unified engine as the main backtester, but returns
    a richer structure for interactive exploration:
        - full trade table
        - entry/exit markers
        - plus the basic equity/return series
    """
    if vix_weekly is None or vix_weekly.empty:
        empty_arr = np.array([])
        empty_df = pd.DataFrame()
        empty_markers = {
            "entry_dates": empty_arr,
            "entry_vix": empty_arr,
            "exit_dates": empty_arr,
            "exit_vix": empty_arr,
        }
        return {
            "equity": empty_arr,
            "weekly_returns": empty_arr,
            "realized_weekly": empty_arr,
            "unrealized_weekly": empty_arr,
            "win_rate": 0.0,
            "trades": 0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
            "trades_df": empty_df,
            "entry_markers": empty_markers,
        }

    # Run unified engine
    bt = run_backtest(vix_weekly, params)
    trade_log = bt.get("trade_log", [])

    # Table of trades
    trades_df = _build_trades_df(trade_log, vix_weekly)

    # Markers for plotting (e.g., scatter on top of VIX line)
    markers = _build_markers(trades_df)

    # Return a merged structure: base bt results + explorer extras
    result: Dict[str, Any] = {
        "equity": bt["equity"],
        "weekly_returns": bt["weekly_returns"],
        "realized_weekly": bt["realized_weekly"],
        "unrealized_weekly": bt["unrealized_weekly"],
        "win_rate": bt["win_rate"],
        "trades": bt["trades"],
        "avg_trade_dur": bt["avg_trade_dur"],
        "trade_log": trade_log,
        "trades_df": trades_df,
        "entry_markers": markers,
    }

    return result