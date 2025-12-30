#!/usr/bin/env python3
"""
Per-Regime Grid Scan for VIX 5% Weekly Suite

Runs a separate parameter grid scan restricted to the historical weeks that fell
into each regime (Ultra Low, Low, Medium, High, Extreme). The best row for each 
regime is saved to param_history.json under a regime-specific key.

This enables the adaptive backtester to dynamically use optimized parameters
based on the current market volatility regime.

Usage:
    from experiments.per_regime_grid_scan import run_per_regime_grid_scan
    
    grid_df, best_by_regime = run_per_regime_grid_scan(
        vix_weekly,
        base_params,
        criteria="balanced"
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable

from core.regime_adapter import RegimeAdapter, REGIME_CONFIGS, RegimeConfig
from core.backtester import run_backtest


# ============================================================
# Helper Metrics (same as grid_scan.py)
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
    """Compute max drawdown as a negative fraction."""
    if len(equity) == 0:
        return 0.0
    e = np.asarray(equity, dtype=float)
    cummax = np.maximum.accumulate(e)
    dd = (e - cummax) / cummax
    return float(dd.min())


def _compute_score(cagr: float, max_dd: float, criteria: str) -> float:
    """Compute score based on criteria."""
    if criteria == "cagr":
        return cagr
    elif criteria == "maxdd":
        return -max_dd  # less negative = better
    else:
        # balanced: reward CAGR, penalize drawdown
        return cagr - 0.5 * abs(max_dd)


# ============================================================
# Single Regime Grid Scan
# ============================================================

def run_single_regime_grid_scan(
    vix_weekly: pd.Series,
    base_params: Dict[str, Any],
    regime_config: RegimeConfig,
    criteria: str = "balanced",
    entry_grid: Optional[List[float]] = None,
    sigma_grid: Optional[List[float]] = None,
    otm_grid: Optional[List[float]] = None,
    dte_grid: Optional[List[int]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Run grid scan for a single regime using its historical data subset.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        VIX weekly data (only weeks in this regime)
    base_params : dict
        Base parameters from sidebar
    regime_config : RegimeConfig
        Configuration for this regime
    criteria : str
        Scoring criteria: "balanced", "cagr", or "maxdd"
    entry_grid : list[float]
        Entry percentiles to test
    sigma_grid : list[float]
        Sigma multipliers to test
    otm_grid : list[float]
        OTM distances to test
    dte_grid : list[int]
        DTE values to test
    progress_cb : callable
        Optional progress callback (current, total)
        
    Returns
    -------
    pd.DataFrame
        Grid results sorted by score
    """
    if vix_weekly is None or len(vix_weekly) < 26:
        return pd.DataFrame()
    
    # Use regime-specific defaults if no grid provided
    if not entry_grid:
        # Centered around regime's default entry percentile
        base_ep = regime_config.entry_percentile
        entry_grid = [
            max(0.01, base_ep - 0.10),
            max(0.01, base_ep - 0.05),
            base_ep,
            min(0.95, base_ep + 0.05),
            min(0.95, base_ep + 0.10),
        ]
    
    if not sigma_grid:
        base_sig = regime_config.sigma_mult
        sigma_grid = [
            max(0.3, base_sig - 0.3),
            base_sig,
            min(2.5, base_sig + 0.3),
        ]
    
    if not otm_grid:
        base_otm = regime_config.otm_pts
        otm_grid = [
            max(2.0, base_otm - 4.0),
            max(2.0, base_otm - 2.0),
            base_otm,
            base_otm + 2.0,
            base_otm + 4.0,
        ]
    
    if not dte_grid:
        base_dte = regime_config.long_dte_weeks
        dte_grid = [
            max(4, base_dte - 8),
            base_dte,
            min(52, base_dte + 8),
        ]
    
    rows: List[Dict[str, Any]] = []
    total_combos = len(entry_grid) * len(sigma_grid) * len(otm_grid) * len(dte_grid)
    current = 0
    
    for ep in entry_grid:
        for sig in sigma_grid:
            for otm in otm_grid:
                for dte in dte_grid:
                    current += 1
                    if progress_cb:
                        try:
                            progress_cb(current, total_combos)
                        except Exception:
                            pass
                    
                    # Build params for this combo
                    params = dict(base_params)
                    params.update({
                        "entry_percentile": float(ep),
                        "sigma_mult": float(sig),
                        "otm_pts": float(otm),
                        "long_dte_weeks": int(dte),
                        "mode": regime_config.mode,
                        "alloc_pct": float(base_params.get("alloc_pct", regime_config.alloc_pct)),
                        "target_mult": regime_config.target_mult,
                        "exit_mult": regime_config.exit_mult,
                    })
                    
                    # Run backtest
                    bt = run_backtest(vix_weekly, params)
                    eq = np.asarray(bt["equity"], dtype=float)
                    
                    # Compute metrics
                    if len(eq) < 2 or eq[0] <= 0:
                        cagr = 0.0
                        max_dd = 0.0
                        total_return = 0.0
                        final_eq = float(eq[-1]) if len(eq) else 0.0
                    else:
                        final_eq = float(eq[-1])
                        total_return = float(final_eq / eq[0] - 1.0)
                        cagr = _compute_cagr(eq)
                        max_dd = _compute_max_dd(eq)
                    
                    win_rate = float(bt.get("win_rate", 0.0))
                    trades = int(bt.get("trades", 0))
                    avg_dur = float(bt.get("avg_trade_dur", 0.0))
                    
                    # Sharpe ratio
                    weekly = np.asarray(bt.get("weekly_returns", []), dtype=float)
                    if weekly.size > 1 and np.isfinite(weekly).all() and weekly.std() > 0:
                        sharpe = float(weekly.mean() / weekly.std() * np.sqrt(52.0))
                    else:
                        sharpe = float("nan")
                    
                    score = _compute_score(cagr, max_dd, criteria)
                    
                    row = {
                        "entry_percentile": float(ep),
                        "sigma_mult": float(sig),
                        "otm_pts": float(otm),
                        "long_dte_weeks": int(dte),
                        "mode": regime_config.mode,
                        "alloc_pct": float(params["alloc_pct"]),
                        "target_mult": regime_config.target_mult,
                        "exit_mult": regime_config.exit_mult,
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
    df.sort_values("Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


# ============================================================
# Per-Regime Grid Scan (Main Entry Point)
# ============================================================

def run_per_regime_grid_scan(
    vix_weekly: pd.Series,
    base_params: Dict[str, Any],
    criteria: str = "balanced",
    entry_grid: Optional[List[float]] = None,
    sigma_grid: Optional[List[float]] = None,
    otm_grid: Optional[List[float]] = None,
    dte_grid: Optional[List[int]] = None,
    lookback_weeks: int = 52,
    min_weeks_per_regime: int = 52,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Execute grid scan separately for each regime's historical period.
    
    This is the main entry point for per-regime optimization.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Full VIX weekly history
    base_params : dict
        Base parameters from sidebar
    criteria : str
        Scoring criteria: "balanced", "cagr", or "maxdd"
    entry_grid : list[float] | None
        Entry percentiles to test (regime-specific defaults if None)
    sigma_grid : list[float] | None
        Sigma multipliers to test
    otm_grid : list[float] | None
        OTM distances to test
    dte_grid : list[int] | None
        DTE values to test
    lookback_weeks : int
        Lookback for percentile calculation
    min_weeks_per_regime : int
        Minimum weeks required to run scan for a regime
    progress_cb : callable
        Progress callback (regime_name, current, total)
        
    Returns
    -------
    combined_df : pd.DataFrame
        All grid rows with "Regime" column
    best_by_regime : dict
        {regime_name: best_record_dict}
    """
    # Initialize regime adapter
    adapter = RegimeAdapter(vix_weekly, lookback_weeks=lookback_weeks)
    adapter.compute_regime_history()
    
    # Convert history to DataFrame for masking
    history_df = pd.DataFrame(adapter.regime_history)
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.set_index("date")
    
    combined_rows: List[pd.DataFrame] = []
    best_by_regime: Dict[str, Dict] = {}
    
    for regime_name, config in REGIME_CONFIGS.items():
        # Get weeks in this regime
        mask = history_df["regime"] == regime_name
        regime_dates = history_df[mask].index
        
        if len(regime_dates) < min_weeks_per_regime:
            print(f"[Per-Regime Scan] Skipping {regime_name}: only {len(regime_dates)} weeks")
            continue
        
        print(f"[Per-Regime Scan] Running grid for {regime_name} ({len(regime_dates)} weeks)")
        
        # Create progress callback wrapper
        def regime_progress(current: int, total: int):
            if progress_cb:
                progress_cb(regime_name, current, total)
        
        # Slice VIX series to this regime's dates
        regime_vix = vix_weekly.loc[vix_weekly.index.isin(regime_dates)]
        
        if len(regime_vix) < min_weeks_per_regime:
            print(f"[Per-Regime Scan] Insufficient data for {regime_name}")
            continue
        
        # Run grid scan for this regime
        grid_df = run_single_regime_grid_scan(
            vix_weekly=regime_vix,
            base_params=base_params,
            regime_config=config,
            criteria=criteria,
            entry_grid=entry_grid,
            sigma_grid=sigma_grid,
            otm_grid=otm_grid,
            dte_grid=dte_grid,
            progress_cb=regime_progress,
        )
        
        if grid_df.empty:
            continue
        
        # Add regime column
        grid_df = grid_df.copy()
        grid_df.insert(0, "Regime", config.name)
        combined_rows.append(grid_df)
        
        # Extract best row
        best_row = grid_df.iloc[0].to_dict()
        
        # Record to param_history
        try:
            from core.param_history import record_best_from_grid
            
            strategy_id = f"{base_params.get('mode', 'diagonal')}__{regime_name}"
            record_best_from_grid(
                strategy_id=strategy_id,
                df=grid_df,
                base_params=base_params,
                criteria=criteria,
            )
        except Exception as e:
            print(f"[Per-Regime Scan] Warning: could not save to history: {e}")
        
        # Store for immediate use
        best_by_regime[regime_name] = {
            "criteria": criteria,
            "params": base_params.copy(),
            "row": best_row,
            "weeks_tested": len(regime_vix),
        }
    
    # Combine all regime results
    if not combined_rows:
        return pd.DataFrame(), {}
    
    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_df.sort_values("Score", ascending=False, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df, best_by_regime


# ============================================================
# Summary and Visualization Helpers
# ============================================================

def create_regime_comparison_df(best_by_regime: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame comparing best params across regimes.
    """
    rows = []
    
    for regime_name, rec in best_by_regime.items():
        row = rec.get("row", {})
        config = REGIME_CONFIGS.get(regime_name)
        
        rows.append({
            "Regime": config.name if config else regime_name,
            "Percentile Range": f"{config.percentile_range[0]:.0%}–{config.percentile_range[1]:.0%}" if config else "N/A",
            "Entry %ile": row.get("entry_percentile", "N/A"),
            "OTM pts": row.get("otm_pts", "N/A"),
            "Sigma ×": row.get("sigma_mult", "N/A"),
            "DTE weeks": row.get("long_dte_weeks", "N/A"),
            "Mode": row.get("mode", "diagonal"),
            "CAGR": f"{row.get('CAGR', 0):.1%}",
            "Max DD": f"{row.get('MaxDD', 0):.1%}",
            "Score": f"{row.get('Score', 0):.3f}",
            "Weeks Tested": rec.get("weeks_tested", "N/A"),
        })
    
    return pd.DataFrame(rows)


def get_optimized_params_for_regime(
    mode: str,
    regime_name: str,
    fallback_to_static: bool = True,
) -> Dict[str, Any]:
    """
    Get optimized parameters for a regime from history.
    
    Falls back to static REGIME_CONFIGS if no optimization found.
    """
    try:
        from core.param_history import get_best_for_strategy
        
        strategy_id = f"{mode}__{regime_name}"
        rec = get_best_for_strategy(strategy_id)
        
        if rec and "row" in rec:
            row = rec["row"]
            return {
                "entry_percentile": row.get("entry_percentile"),
                "sigma_mult": row.get("sigma_mult"),
                "otm_pts": row.get("otm_pts"),
                "long_dte_weeks": int(row.get("long_dte_weeks", 26)),
                "alloc_pct": row.get("alloc_pct"),
                "target_mult": row.get("target_mult"),
                "exit_mult": row.get("exit_mult"),
                "mode": row.get("mode"),
            }
    except Exception:
        pass
    
    # Fallback to static config
    if fallback_to_static and regime_name in REGIME_CONFIGS:
        config = REGIME_CONFIGS[regime_name]
        return config.to_dict()
    
    return {}
