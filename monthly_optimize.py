#!/usr/bin/env python3
"""
Monthly Dynamic Profile Re-Optimization for VIX 5% Weekly Suite

This script runs grid scans on regime-filtered data and updates
the param_history.json with freshly optimized parameters.

Run this monthly (e.g., 1st of each month) via cron:
    0 5 1 * * /usr/bin/python3 /path/to/monthly_optimize.py >> /path/to/opt_log.txt 2>&1

Or manually:
    python monthly_optimize.py
    python monthly_optimize.py --regime ULTRA_LOW --lookback 104
    python monthly_optimize.py --all-regimes --lookback 52

Features:
- Detects current VIX regime
- Filters historical data to only weeks in that regime
- Runs grid scan on filtered data
- Saves optimized params per regime
- Sends optional email notification
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data_loader import load_vix_weekly
from core.regime_adapter import RegimeAdapter, REGIME_CONFIGS
from core.backtester import run_backtest
from core.param_history import record_best_from_grid, get_best_for_regime


# Default parameters for optimization
DEFAULT_PARAMS = {
    "initial_capital": 250_000.0,
    "alloc_pct": 0.01,
    "mode": "diagonal",
    "risk_free": 0.03,
    "fee_per_contract": 0.65,
    "realism": 1.0,
    "entry_lookback_weeks": 52,
}

# Grid search ranges
ENTRY_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SIGMA_GRID = [0.5, 0.8, 1.0]
OTM_GRID = [2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0]
DTE_GRID = [5, 8, 13, 26]


def compute_vix_percentile(prices: np.ndarray, lookback: int = 52) -> np.ndarray:
    """Compute rolling percentile for price series."""
    n = len(prices)
    pct = np.full(n, np.nan, dtype=float)
    
    for i in range(lookback, n):
        window = prices[i - lookback:i]
        pct[i] = (window < prices[i]).mean()
    
    return pct


def get_regime_from_percentile(pct: float) -> str:
    """Map percentile to regime name."""
    if pct <= 0.10:
        return "ULTRA_LOW"
    elif pct <= 0.25:
        return "LOW"
    elif pct <= 0.50:
        return "MEDIUM"
    elif pct <= 0.75:
        return "HIGH"
    else:
        return "EXTREME"


def filter_data_by_regime(
    vix_weekly: pd.Series,
    target_regime: str,
    lookback_weeks: int = 52,
) -> pd.Series:
    """
    Filter VIX data to only include weeks where regime == target_regime.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Full VIX weekly series
    target_regime : str
        Target regime to filter for
    lookback_weeks : int
        Lookback period for percentile calculation
        
    Returns
    -------
    pd.Series
        Filtered series with only weeks in target regime
    """
    prices = vix_weekly.values.astype(float)
    pct = compute_vix_percentile(prices, lookback_weeks)
    
    # Get regime for each week
    regimes = [get_regime_from_percentile(p) if np.isfinite(p) else None for p in pct]
    
    # Filter to target regime
    mask = np.array([r == target_regime for r in regimes])
    
    filtered = vix_weekly[mask]
    
    print(f"  Regime {target_regime}: {mask.sum()} weeks out of {len(vix_weekly)} total")
    
    return filtered


def compute_cagr(equity: np.ndarray, weeks_per_year: float = 52.0) -> float:
    """Compute CAGR from equity curve."""
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / weeks_per_year
    if years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0


def compute_max_dd(equity: np.ndarray) -> float:
    """Compute maximum drawdown."""
    if len(equity) == 0:
        return 0.0
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    return float(dd.min())


def run_regime_grid_scan(
    vix_weekly: pd.Series,
    regime_name: str,
    base_params: Dict[str, Any],
    criteria: str = "balanced",
) -> pd.DataFrame:
    """
    Run grid scan on regime-filtered data.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Full VIX weekly data
    regime_name : str
        Target regime to optimize for
    base_params : dict
        Base strategy parameters
    criteria : str
        Optimization criteria ("balanced", "cagr", "maxdd")
        
    Returns
    -------
    pd.DataFrame
        Grid scan results sorted by score
    """
    # Filter data to target regime
    filtered = filter_data_by_regime(vix_weekly, regime_name)
    
    if len(filtered) < 20:
        print(f"  Warning: Only {len(filtered)} weeks in {regime_name} regime - insufficient for optimization")
        return pd.DataFrame()
    
    rows = []
    total_combos = len(ENTRY_GRID) * len(SIGMA_GRID) * len(OTM_GRID) * len(DTE_GRID)
    combo_count = 0
    
    for entry_pct in ENTRY_GRID:
        for sigma in SIGMA_GRID:
            for otm in OTM_GRID:
                for dte in DTE_GRID:
                    combo_count += 1
                    
                    # Build params for this combo
                    params = dict(base_params)
                    params["entry_percentile"] = entry_pct
                    params["sigma_mult"] = sigma
                    params["otm_pts"] = otm
                    params["long_dte_weeks"] = dte
                    
                    # Run backtest on filtered data
                    bt = run_backtest(filtered, params)
                    eq = np.asarray(bt["equity"], dtype=float)
                    
                    if len(eq) < 2 or eq[0] <= 0:
                        continue
                    
                    # Compute metrics
                    final_eq = float(eq[-1])
                    total_return = final_eq / eq[0] - 1.0
                    cagr = compute_cagr(eq)
                    max_dd = compute_max_dd(eq)
                    trades = int(bt.get("trades", 0))
                    win_rate = float(bt.get("win_rate", 0))
                    
                    # Compute score based on criteria
                    if criteria == "cagr":
                        score = cagr
                    elif criteria == "maxdd":
                        score = -max_dd  # Less drawdown is better
                    else:  # balanced
                        score = cagr - 0.5 * max_dd
                    
                    rows.append({
                        "entry_pct": entry_pct,
                        "sigma_mult": sigma,
                        "otm_pts": otm,
                        "long_dte_weeks": dte,
                        "regime": regime_name,
                        "Final_eq": final_eq,
                        "Total_return": total_return,
                        "CAGR": cagr,
                        "MaxDD": max_dd,
                        "Trades": trades,
                        "Win_rate": win_rate,
                        "Score": score,
                    })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df.sort_values("Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def optimize_regime(
    vix_weekly: pd.Series,
    regime_name: str,
    base_params: Dict[str, Any],
    criteria: str = "balanced",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run optimization for a single regime and save results.
    
    Returns the best parameter set found, or None if optimization failed.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing for regime: {regime_name}")
        print(f"{'='*60}")
    
    # Run grid scan
    df = run_regime_grid_scan(vix_weekly, regime_name, base_params, criteria)
    
    if df.empty:
        if verbose:
            print(f"  No valid results for {regime_name}")
        return None
    
    # Get best result
    best = df.iloc[0].to_dict()
    
    if verbose:
        print(f"\n  Best parameters for {regime_name}:")
        print(f"    Entry Percentile: {best['entry_pct']}")
        print(f"    Sigma Mult: {best['sigma_mult']}")
        print(f"    OTM Points: {best['otm_pts']}")
        print(f"    Long DTE (weeks): {best['long_dte_weeks']}")
        print(f"    CAGR: {best['CAGR']*100:.2f}%")
        print(f"    Max DD: {best['MaxDD']*100:.2f}%")
        print(f"    Trades: {best['Trades']}")
        print(f"    Score: {best['Score']:.4f}")
    
    # Save to param history
    strategy_id = f"{base_params.get('mode', 'diagonal')}__{regime_name}"
    record_best_from_grid(strategy_id, df, base_params, criteria)
    
    if verbose:
        print(f"\n  ✓ Saved to param_history.json as '{strategy_id}'")
    
    return best


def get_current_regime(lookback_weeks: int = 52) -> str:
    """
    Get the current VIX regime based on latest data.
    
    Returns
    -------
    str
        Current regime name
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(weeks=lookback_weeks + 10)
    
    vix = load_vix_weekly(start_date, end_date)
    
    if vix is None or len(vix) < lookback_weeks:
        return "MEDIUM"  # Default fallback
    
    prices = vix.values.astype(float)
    pct = compute_vix_percentile(prices, lookback_weeks)
    
    current_pct = pct[-1] if np.isfinite(pct[-1]) else 0.5
    
    return get_regime_from_percentile(current_pct)


def main():
    parser = argparse.ArgumentParser(
        description="Monthly VIX regime optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python monthly_optimize.py                     # Optimize current regime
    python monthly_optimize.py --all-regimes      # Optimize all regimes
    python monthly_optimize.py --regime ULTRA_LOW  # Optimize specific regime
    python monthly_optimize.py --lookback 104      # Use 2-year lookback
        """
    )
    
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Specific regime to optimize (ULTRA_LOW, LOW, MEDIUM, HIGH, EXTREME)"
    )
    
    parser.add_argument(
        "--all-regimes",
        action="store_true",
        help="Optimize all regimes"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=104,
        help="Lookback period in weeks for data filtering (default: 104 = 2 years)"
    )
    
    parser.add_argument(
        "--criteria",
        type=str,
        default="balanced",
        choices=["balanced", "cagr", "maxdd"],
        help="Optimization criteria (default: balanced)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="diagonal",
        choices=["diagonal", "long_only"],
        help="Strategy mode (default: diagonal)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("\n" + "="*60)
        print("VIX 5% Weekly Suite - Monthly Regime Optimization")
        print(f"Run time: {dt.datetime.now().isoformat()}")
        print("="*60)
    
    # Load full historical data
    end_date = dt.date.today()
    start_date = dt.date(2004, 1, 1)  # Full history
    
    if verbose:
        print(f"\nLoading VIX data from {start_date} to {end_date}...")
    
    vix_weekly = load_vix_weekly(start_date, end_date)
    
    if vix_weekly is None or vix_weekly.empty:
        print("ERROR: Could not load VIX data")
        sys.exit(1)
    
    if verbose:
        print(f"  Loaded {len(vix_weekly)} weeks of data")
    
    # Build base params
    base_params = dict(DEFAULT_PARAMS)
    base_params["mode"] = args.mode
    base_params["entry_lookback_weeks"] = args.lookback
    
    # Determine which regimes to optimize
    if args.all_regimes:
        regimes_to_optimize = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"]
    elif args.regime:
        regimes_to_optimize = [args.regime.upper()]
    else:
        # Default: optimize current regime only
        current = get_current_regime(args.lookback)
        if verbose:
            print(f"\nCurrent VIX regime: {current}")
        regimes_to_optimize = [current]
    
    # Run optimization for each regime
    results = {}
    for regime in regimes_to_optimize:
        best = optimize_regime(
            vix_weekly,
            regime,
            base_params,
            criteria=args.criteria,
            verbose=verbose,
        )
        if best:
            results[regime] = best
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        for regime, params in results.items():
            print(f"\n{regime}:")
            print(f"  Entry: {params['entry_pct']}, Sigma: {params['sigma_mult']}, OTM: {params['otm_pts']}, DTE: {params['long_dte_weeks']}")
            print(f"  CAGR: {params['CAGR']*100:.1f}%, MaxDD: {params['MaxDD']*100:.1f}%, Trades: {params['Trades']}")
        
        print(f"\n✓ Optimization complete. Results saved to param_history.json")
        print(f"  Run time: {dt.datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    main()
