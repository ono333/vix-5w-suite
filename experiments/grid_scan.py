"""
Grid Scan for VIX 5-Weekly Strategy
-----------------------------------

Runs the unified backtest engine over a parameter grid and
returns a ranked DataFrame of results.

Also records the best parameter set per strategy (mode)
via core.param_history.record_best_from_grid().

Public API
----------
run_grid_scan(
    vix_weekly: pd.Series,
    base_params: dict,
    criteria: str = "balanced",
    entry_grid: list[float] | None = None,
    sigma_grid: list[float] | None = None,
    otm_grid: list[float] | None = None,
    dte_grid: list[int] | None = None,
) -> pd.DataFrame

Notes
-----
- If any of the *_grid arguments are None or empty, sensible defaults
  are used internally.
- This scan uses the LOCAL run_backtest engine (not Massive). That keeps
  it fast and deterministic, and still plays nicely with Massive-backed
  runs on the main backtester.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    from core.backtester import run_backtest
    from core.param_history import record_best_from_grid
except ImportError:
    # Fallback for different directory structures
    from backtester import run_backtest
    from param_history import record_best_from_grid


# ============================================================
# === Helper Metrics =========================================
# ============================================================

def _compute_cagr(equity: np.ndarray, weeks_per_year: float = 52.0) -> float:
    """Compute CAGR from weekly equity series."""
    n = len(equity)
    if n < 2 or equity[0] <= 0:
        return 0.0
    years = (n - 1) / weeks_per_year
    if years <= 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def _compute_max_dd(equity: np.ndarray) -> float:
    """Compute max drawdown as a negative fraction, e.g. -0.35 for -35%."""
    if len(equity) == 0:
        return 0.0
    e = np.asarray(equity, dtype=float)
    cummax = np.maximum.accumulate(e)
    dd = (e - cummax) / cummax
    return float(dd.min())


def _rank_series_high_good(x: pd.Series) -> pd.Series:
    """Convert a series into 0â€“1 ranks where higher values are better."""
    return x.rank(method="average", pct=True)


def _rank_series_low_good(x: pd.Series) -> pd.Series:
    """Convert a series into 0â€“1 ranks where lower values are better."""
    return (len(x) + 1 - x.rank(method="average")) / len(x)


def _apply_scoring(df: pd.DataFrame, criteria: str) -> pd.DataFrame:
    """
    Add a 'score' column based on the chosen criteria.

    criteria:
        - "balanced" : equal weight CAGRâ†‘ and MaxDDâ†“
        - "cagr"     : CAGRâ†‘ only
        - "maxdd"    : MaxDDâ†“ only
    """
    if df.empty:
        df["score"] = np.nan
        return df

    cagr = df["cagr"]
    maxdd = df["max_dd"]

    if criteria == "cagr":
        df["score"] = _rank_series_high_good(cagr)
    elif criteria == "maxdd":
        df["score"] = _rank_series_low_good(maxdd)
    else:
        # balanced
        r_cagr = _rank_series_high_good(cagr)
        r_dd = _rank_series_low_good(maxdd)
        df["score"] = 0.5 * r_cagr + 0.5 * r_dd

    df.sort_values(by="score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# === Main Scan ==============================================
# ============================================================

def run_grid_scan(
    vix_weekly: pd.Series,
    base_params: Dict[str, Any],
    criteria: str = "balanced",
    entry_grid: Optional[List[float]] = None,
    sigma_grid: Optional[List[float]] = None,
    otm_grid: Optional[List[float]] = None,
    dte_grid: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run a parameter grid scan over the unified VIX/UVXY strategy (LOCAL engine).

    Parameters
    ----------
    vix_weekly : pd.Series
        Weekly underlying closes (aligned with the backtester).
    base_params : dict
        Current parameter set from the sidebar (unified schema).
    criteria : str
        "balanced", "cagr", or "maxdd".
    entry_grid : list[float] | None
        Entry percentiles (0â€“1). If None/empty -> defaults used.
    sigma_grid : list[float] | None
        Sigma multipliers for long call. If None/empty -> defaults used.
    otm_grid : list[float] | None
        OTM distances in points. If None/empty -> defaults used.
    dte_grid : list[int] | None
        Long call DTE choices (weeks). If None/empty -> defaults used.

    Returns
    -------
    pd.DataFrame
        One row per parameter combo with columns such as:
            entry_pct, sigma_mult, otm_pts, long_dte_weeks,
            CAGR, MaxDD, Final_eq, Total_return, Trades, Win_rate,
            Avg_dur_weeks, Sharpe, Score, ...
    """
    if vix_weekly is None or vix_weekly.empty:
        return pd.DataFrame()

    # Identify "strategy" for history purposes â€“ for now we use position mode
    strategy_id = base_params.get("mode", "diagonal")

    # -----------------------------------------------
    # Fallback defaults if no grid passed in
    # -----------------------------------------------
    if not entry_grid:
        entry_grid = [0.10, 0.30, 0.50, 0.70, 0.90]

    if not sigma_grid:
        sigma_grid = [0.5, 0.8, 1.0]

    if not otm_grid:
        otm_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0]

    if not dte_grid:
        dte_grid = [3, 5, 15, 26]

    rows: List[Dict[str, Any]] = []

    for ep in entry_grid:
        for sig in sigma_grid:
            for otm in otm_grid:
                for dte_long in dte_grid:
                    # ------------------------------
                    # 1) Build params for this combo
                    # ------------------------------
                    params = dict(base_params)
                    params["entry_percentile"] = float(ep)
                    params["sigma_mult"] = float(sig)
                    params["otm_pts"] = float(otm)
                    params["long_dte_weeks"] = int(dte_long)

                    bt = run_backtest(vix_weekly, params)
                    eq = np.asarray(bt["equity"], dtype=float)

                    # ------------------------------
                    # 2) Core performance metrics
                    # ------------------------------
                    if len(eq) < 2 or eq[0] <= 0:
                        total_return = 0.0
                        cagr = 0.0
                        max_dd = 0.0
                        final_eq = float(eq[-1]) if len(eq) else 0.0
                    else:
                        final_eq = float(eq[-1])
                        total_return = float(final_eq / eq[0] - 1.0)
                        cagr = _compute_cagr(eq)
                        max_dd = _compute_max_dd(eq)

                    # Win-rate / trades / avg duration (if provided by backtester)
                    win_rate = float(bt.get("win_rate", np.nan))
                    trades = int(bt.get("trades", 0))
                    avg_dur = float(bt.get("avg_trade_dur", np.nan))

                    # Weekly Sharpe from weekly_returns (if available)
                    weekly = np.asarray(bt.get("weekly_returns", []), dtype=float)
                    if weekly.size > 1 and np.isfinite(weekly).all() and weekly.std() > 0:
                        sharpe = float(weekly.mean() / weekly.std() * np.sqrt(52.0))
                    else:
                        sharpe = float("nan")

                    # ------------------------------
                    # 3) Scoring function
                    # ------------------------------
                    if criteria == "cagr":
                        score = cagr
                    elif criteria == "maxdd":
                        score = -max_dd  # less drawdown is better
                    else:
                        # "balanced": reward CAGR, penalise drawdown
                        score = cagr - 0.5 * max_dd

                    # ------------------------------
                    # 4) Assemble row
                    # ------------------------------
                    row = {
                        # scanned parameters
                        "entry_pct": float(ep),
                        "entry_lookback_weeks": int(base_params.get("entry_lookback_weeks", 52)),
                        "sigma_mult": float(sig),
                        "otm_pts": float(otm),
                        "long_dte_weeks": int(dte_long),

                        # risk / trade-management settings (fixed during this scan)
                        "alloc_pct": float(base_params.get("alloc_pct", 0.01)),
                        "target_mult": float(base_params.get("target_mult", 1.20)),
                        "exit_mult": float(base_params.get("exit_mult", 0.50)),
                        "fee_per_contract": float(base_params.get("fee_per_contract", 0.65)),
                        "slippage_bps": float(base_params.get("slippage_bps", 5.0)),
                        "realism": float(base_params.get("realism", 1.0)),

                        # performance metrics
                        "Final_eq": final_eq,
                        "Total_return": total_return,
                        "CAGR": cagr,
                        "MaxDD": max_dd,
                        "Trades": trades,
                        "Win_rate": win_rate,
                        "Avg_dur_weeks": avg_dur,
                        "Sharpe": sharpe,
                        "Score": score,
                    }

                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort best-to-worst by the chosen score
    df.sort_values("Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df