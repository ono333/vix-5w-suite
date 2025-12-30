#!/usr/bin/env python3
"""
Parameter history helper for the VIX 5% Weekly suite.

We store, per strategy_id (currently the `mode` string), the best
grid-scan rows together with a small snapshot of the base params
used when the scan was run.

Extended to support per-regime parameter storage with keys like:
    "diagonal__ULTRA_LOW"
    "diagonal__LOW"
    etc.

Public helpers:
    record_best_from_grid(strategy_id, df, base_params, criteria)
    get_best_for_strategy(strategy_id)
    get_best_for_regime(mode, regime_name)  # NEW
    apply_best_if_requested(params)
    apply_regime_params(params, regime_name)  # NEW
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Where we store the history JSON
_HISTORY_PATH = Path(__file__).resolve().parent / "param_history.json"


# ============================================================
# JSON helpers
# ============================================================

def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert `obj` into something JSON-serializable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    if isinstance(obj, (pd.Series, pd.Index, np.ndarray)):
        return [_to_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    return str(obj)


def _load_history() -> Dict[str, Any]:
    if not _HISTORY_PATH.exists():
        return {"strategies": {}}
    try:
        with _HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "strategies" not in data or not isinstance(data["strategies"], dict):
            data = {"strategies": {}}
        return data
    except Exception:
        return {"strategies": {}}


def _save_history(hist: Dict[str, Any]) -> None:
    safe = _to_jsonable(hist)
    with _HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, sort_keys=True)


# ============================================================
# Recording best rows from grid scan
# ============================================================

def record_best_from_grid(
    strategy_id: str,
    df: pd.DataFrame,
    base_params: Dict[str, Any],
    criteria: str,
) -> None:
    """
    Take the top row of `df`, plus a small snapshot of `base_params`,
    and append it to the history for `strategy_id`.
    
    strategy_id can be:
        - Simple mode: "diagonal", "long_only"
        - Per-regime: "diagonal__ULTRA_LOW", "diagonal__LOW", etc.
    """
    if df is None or df.empty:
        return

    best_row = df.iloc[0].to_dict()

    hist = _load_history()
    strategies = hist.setdefault("strategies", {})
    strat_hist = strategies.setdefault(strategy_id, [])

    param_snapshot = {
        "mode": base_params.get("mode"),
        "initial_capital": base_params.get("initial_capital"),
        "alloc_pct": base_params.get("alloc_pct"),
        "entry_percentile": base_params.get("entry_percentile"),
        "sigma_mult": base_params.get("sigma_mult"),
        "otm_pts": base_params.get("otm_pts"),
        "long_dte_weeks": base_params.get("long_dte_weeks"),
        "risk_free": base_params.get("risk_free"),
        "fee_per_contract": base_params.get("fee_per_contract"),
        "target_mult": base_params.get("target_mult"),
        "exit_mult": base_params.get("exit_mult"),
    }

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "criteria": criteria,
        "row": best_row,
        "params": param_snapshot,
    }

    strat_hist.append(entry)
    _save_history(hist)


# ============================================================
# Reading / applying best params
# ============================================================

def get_best_for_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the *last* recorded best entry for a given strategy_id, or None.
    """
    hist = _load_history()
    strategies = hist.get("strategies", {})
    entries = strategies.get(strategy_id) or []
    if not entries:
        return None
    return entries[-1]


def get_best_for_regime(mode: str, regime_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the best recorded params for a specific regime.
    
    Parameters
    ----------
    mode : str
        Strategy mode ("diagonal" or "long_only")
    regime_name : str
        Regime name ("ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME")
        
    Returns
    -------
    dict or None
        Best parameter record for this regime, or None if not found
    """
    strategy_id = f"{mode}__{regime_name}"
    return get_best_for_strategy(strategy_id)


def get_all_regime_params(mode: str) -> Dict[str, Dict[str, Any]]:
    """
    Get best params for all regimes for a given mode.
    
    Returns dict mapping regime_name -> best_record
    """
    regime_names = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"]
    results = {}
    
    for regime in regime_names:
        best = get_best_for_regime(mode, regime)
        if best:
            results[regime] = best
    
    return results


def apply_best_if_requested(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally override the current params with the best ones from history
    for the selected strategy.

    Expected flags in `params` (any of these will trigger it if True):
      - "use_best_from_history"
      - "use_best_params"

    Only these tuning keys are overridden:
      - entry_percentile
      - sigma_mult
      - otm_pts
      - long_dte_weeks
    """
    use_best = bool(
        params.get("use_best_from_history")
        or params.get("use_best_params")
    )
    if not use_best:
        return params

    strategy_id = params.get("mode", "diagonal")
    rec = get_best_for_strategy(strategy_id)
    if not rec:
        return params

    row = rec.get("row") or {}
    new_params = dict(params)

    for key in ("entry_percentile", "sigma_mult", "otm_pts", "long_dte_weeks",
                "target_mult", "exit_mult", "alloc_pct"):
        if key in row:
            try:
                new_params[key] = float(row[key])
            except Exception:
                pass

    return new_params


def apply_regime_params(
    params: Dict[str, Any],
    regime_name: str,
    fallback_to_static: bool = True,
) -> Dict[str, Any]:
    """
    Apply optimized parameters for a specific regime.
    
    Parameters
    ----------
    params : dict
        Base parameters to modify
    regime_name : str
        Regime to apply params for
    fallback_to_static : bool
        If True and no optimized params found, use static REGIME_CONFIGS
        
    Returns
    -------
    dict
        Modified parameters with regime-specific values
    """
    mode = params.get("mode", "diagonal")
    rec = get_best_for_regime(mode, regime_name)
    
    new_params = dict(params)
    
    if rec and "row" in rec:
        row = rec["row"]
        for key in ("entry_percentile", "sigma_mult", "otm_pts", "long_dte_weeks",
                    "target_mult", "exit_mult", "alloc_pct", "mode"):
            if key in row and row[key] is not None:
                try:
                    if key == "long_dte_weeks":
                        new_params[key] = int(row[key])
                    elif key == "mode":
                        new_params[key] = str(row[key])
                    else:
                        new_params[key] = float(row[key])
                except Exception:
                    pass
        return new_params
    
    # Fallback to static config
    if fallback_to_static:
        try:
            from core.regime_adapter import REGIME_CONFIGS
            if regime_name in REGIME_CONFIGS:
                config = REGIME_CONFIGS[regime_name]
                new_params.update({
                    "entry_percentile": config.entry_percentile,
                    "sigma_mult": config.sigma_mult,
                    "otm_pts": config.otm_pts,
                    "long_dte_weeks": config.long_dte_weeks,
                    "target_mult": config.target_mult,
                    "exit_mult": config.exit_mult,
                    "alloc_pct": config.alloc_pct,
                    "mode": config.mode,
                })
        except ImportError:
            pass
    
    return new_params


# ============================================================
# Utility functions
# ============================================================

def list_all_strategies() -> list[str]:
    """List all strategy IDs that have stored history."""
    hist = _load_history()
    strategies = hist.get("strategies", {})
    return list(strategies.keys())


def clear_strategy_history(strategy_id: str) -> bool:
    """Clear history for a specific strategy. Returns True if found and cleared."""
    hist = _load_history()
    strategies = hist.get("strategies", {})
    
    if strategy_id in strategies:
        del strategies[strategy_id]
        _save_history(hist)
        return True
    return False


def get_history_summary() -> pd.DataFrame:
    """
    Get a summary of all stored parameter history.
    
    Returns DataFrame with columns:
        - strategy_id
        - num_records
        - latest_timestamp
        - latest_criteria
        - latest_score (if available)
    """
    hist = _load_history()
    strategies = hist.get("strategies", {})
    
    rows = []
    for strategy_id, entries in strategies.items():
        if entries:
            latest = entries[-1]
            rows.append({
                "strategy_id": strategy_id,
                "num_records": len(entries),
                "latest_timestamp": latest.get("timestamp"),
                "latest_criteria": latest.get("criteria"),
                "latest_score": latest.get("row", {}).get("Score"),
            })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()
