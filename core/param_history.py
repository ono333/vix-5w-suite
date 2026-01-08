# core/param_history.py

"""
Parameter history helper for the VIX 5% Weekly suite.

We store, per strategy_id (currently the `mode` string), the best
grid-scan rows together with a small snapshot of the base params
used when the scan was run.

Public helpers:
    record_best_from_grid(strategy_id, df, base_params, criteria)
    get_best_for_strategy(strategy_id)
    apply_best_if_requested(params)  # used in app.py
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Where we store the history JSON (in the core/ folder)
_HISTORY_PATH = Path(__file__).resolve().parent / "param_history.json"


# ============================================================
# JSON helpers
# ============================================================

def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert `obj` into something JSON-serializable.
    This also guarantees we build a *new* tree with no references,
    so circular refs cannot survive.
    """
    # Simple scalars
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Dataclasses
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    # Pandas Timestamps / datetimes
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # Pandas Series / Index / ndarray â†’ list
    if isinstance(obj, (pd.Series, pd.Index, np.ndarray)):
        return [_to_jsonable(x) for x in obj.tolist()]

    # Dict â†’ dict
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # List / tuple / set â†’ list
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    # Fallback: string representation
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
        # If file is corrupt, start fresh
        return {"strategies": {}}


def _save_history(hist: Dict[str, Any]) -> None:
    # Important: convert the whole tree to a NEW, JSON-safe structure
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
    """
    if df is None or df.empty:
        return

    best_row = df.iloc[0].to_dict()

    hist = _load_history()
    strategies = hist.setdefault("strategies", {})
    strat_hist = strategies.setdefault(strategy_id, [])

    # Only keep a *subset* of base_params that is useful for replay:
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


def apply_best_if_requested(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally override the current params with the best ones from history
    for the selected strategy.

    Expected flags in `params` (any of these will trigger it if True):
      - "use_best_from_history"
      - "use_best_params"

    If no history is found, or the flag is False, this returns `params`
    unchanged.

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
        # no history yet
        return params

    row = rec.get("row") or {}
    new_params = dict(params)

    for key in ("entry_percentile", "sigma_mult", "otm_pts", "long_dte_weeks"):
        if key in row:
            try:
                new_params[key] = float(row[key])
            except Exception:
                # if casting fails, just ignore that field
                pass

    return new_params