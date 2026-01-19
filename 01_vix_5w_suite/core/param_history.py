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


# ============================================================
# LBR-Grade Profile Management
# ============================================================

REGIME_NAMES = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"]

# Default regime configurations
DEFAULT_PROFILES = {
    "ULTRA_LOW": {
        "entry_percentile": 0.08,
        "sigma_mult": 1.0,
        "otm_pts": 8,
        "long_dte_weeks": 26,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "mode": "diagonal",
        "description": "Aggressive entry, max premium harvest",
    },
    "LOW": {
        "entry_percentile": 0.15,
        "sigma_mult": 1.0,
        "otm_pts": 10,
        "long_dte_weeks": 26,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "mode": "diagonal",
        "description": "Standard entry, balanced approach",
    },
    "MEDIUM": {
        "entry_percentile": 0.30,
        "sigma_mult": 1.0,
        "otm_pts": 12,
        "long_dte_weeks": 20,
        "target_mult": 1.3,
        "exit_mult": 0.4,
        "mode": "diagonal",
        "description": "Cautious, selective entries",
    },
    "HIGH": {
        "entry_percentile": 0.60,
        "sigma_mult": 0.8,
        "otm_pts": 15,
        "long_dte_weeks": 13,
        "target_mult": 1.5,
        "exit_mult": 0.3,
        "mode": "long_only",
        "description": "Defensive, reduced exposure",
    },
    "EXTREME": {
        "entry_percentile": 0.90,
        "sigma_mult": 0.5,
        "otm_pts": 20,
        "long_dte_weeks": 8,
        "target_mult": 2.0,
        "exit_mult": 0.2,
        "mode": "long_only",
        "description": "No new positions, wait for calm",
    },
}


def get_profile(regime_name: str, mode: str = "diagonal") -> Dict[str, Any]:
    """
    Get profile for a specific regime with metadata.
    
    Returns dict with:
        - params: dict of parameters
        - last_optimized: timestamp or None
        - is_edited: bool
        - sample_count: int (weeks tested)
        - score: float or None
    """
    strategy_id = f"{mode}__{regime_name}"
    rec = get_best_for_strategy(strategy_id)
    
    if rec and "row" in rec:
        row = rec["row"]
        return {
            "params": {
                "entry_percentile": row.get("entry_percentile"),
                "sigma_mult": row.get("sigma_mult"),
                "otm_pts": row.get("otm_pts"),
                "long_dte_weeks": row.get("long_dte_weeks"),
                "target_mult": row.get("target_mult"),
                "exit_mult": row.get("exit_mult"),
                "mode": row.get("mode", mode),
            },
            "last_optimized": rec.get("timestamp"),
            "is_edited": rec.get("is_edited", False),
            "sample_count": row.get("weeks_tested", rec.get("weeks_tested", 0)),
            "score": row.get("Score"),
            "cagr": row.get("CAGR"),
            "max_dd": row.get("MaxDD"),
            "criteria": rec.get("criteria"),
        }
    
    # Return defaults
    defaults = DEFAULT_PROFILES.get(regime_name, {})
    return {
        "params": defaults.copy(),
        "last_optimized": None,
        "is_edited": False,
        "sample_count": 0,
        "score": None,
        "cagr": None,
        "max_dd": None,
        "criteria": None,
    }


def get_all_profiles(mode: str = "diagonal") -> Dict[str, Dict[str, Any]]:
    """
    Get profiles for all regimes.
    
    Returns dict mapping regime_name -> profile dict
    """
    return {regime: get_profile(regime, mode) for regime in REGIME_NAMES}


def save_profile(
    regime_name: str,
    params: Dict[str, Any],
    mode: str = "diagonal",
    is_edited: bool = True,
    sample_count: int = 0,
    score: float = None,
    cagr: float = None,
    max_dd: float = None,
    criteria: str = "manual",
) -> None:
    """
    Save or update a profile for a specific regime.
    
    This is for manual edits - use record_best_from_grid for optimization results.
    """
    strategy_id = f"{mode}__{regime_name}"
    
    hist = _load_history()
    strategies = hist.setdefault("strategies", {})
    strat_hist = strategies.setdefault(strategy_id, [])
    
    row = {
        "entry_percentile": params.get("entry_percentile"),
        "sigma_mult": params.get("sigma_mult"),
        "otm_pts": params.get("otm_pts"),
        "long_dte_weeks": params.get("long_dte_weeks"),
        "target_mult": params.get("target_mult"),
        "exit_mult": params.get("exit_mult"),
        "mode": params.get("mode", mode),
        "Score": score,
        "CAGR": cagr,
        "MaxDD": max_dd,
        "weeks_tested": sample_count,
    }
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "criteria": criteria,
        "row": row,
        "params": params,
        "is_edited": is_edited,
        "weeks_tested": sample_count,
    }
    
    strat_hist.append(entry)
    _save_history(hist)


def reset_profile_to_default(regime_name: str, mode: str = "diagonal") -> None:
    """Reset a profile to default values."""
    if regime_name in DEFAULT_PROFILES:
        save_profile(
            regime_name=regime_name,
            params=DEFAULT_PROFILES[regime_name].copy(),
            mode=mode,
            is_edited=False,
            criteria="default_reset",
        )


def get_profile_summary_df(mode: str = "diagonal") -> pd.DataFrame:
    """
    Get a summary DataFrame of all profiles for display.
    
    Returns DataFrame with columns:
        - Profile (regime name)
        - Sample Count
        - Last Optimized
        - Is Edited
        - Entry %
        - OTM
        - DTE
        - Score
    """
    profiles = get_all_profiles(mode)
    
    rows = []
    for regime_name, profile in profiles.items():
        params = profile.get("params", {})
        last_opt = profile.get("last_optimized")
        if last_opt:
            try:
                # Format timestamp nicely
                dt_obj = datetime.fromisoformat(last_opt.replace('Z', '+00:00'))
                last_opt_str = dt_obj.strftime("%Y-%m-%d %H:%M")
            except:
                last_opt_str = last_opt[:16] if last_opt else "Never"
        else:
            last_opt_str = "Never"
        
        rows.append({
            "Profile": regime_name,
            "Sample Count": profile.get("sample_count", 0),
            "Last Optimized": last_opt_str,
            "Is Edited": "✏️ Yes" if profile.get("is_edited") else "No",
            "Entry %": f"{params.get('entry_percentile', 0):.0%}",
            "OTM": params.get("otm_pts", 0),
            "DTE (wks)": params.get("long_dte_weeks", 0),
            "Mode": params.get("mode", "diagonal"),
            "Score": f"{profile.get('score', 0):.3f}" if profile.get('score') else "—",
        })
    
    return pd.DataFrame(rows)


def export_all_profiles(mode: str = "diagonal") -> str:
    """Export all profiles as JSON string."""
    profiles = get_all_profiles(mode)
    return json.dumps(_to_jsonable(profiles), indent=2)


def import_profiles(json_str: str, mode: str = "diagonal") -> bool:
    """Import profiles from JSON string. Returns True on success."""
    try:
        profiles = json.loads(json_str)
        for regime_name, profile in profiles.items():
            if regime_name in REGIME_NAMES:
                params = profile.get("params", profile)
                save_profile(
                    regime_name=regime_name,
                    params=params,
                    mode=mode,
                    is_edited=True,
                    sample_count=profile.get("sample_count", 0),
                    criteria="imported",
                )
        return True
    except Exception:
        return False
