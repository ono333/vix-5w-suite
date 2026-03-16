"""
Assignment Store
----------------
Persistent store for share-assignment events (e.g. short UVXY shares
from a V4 Spike Trade, or any direct equity position alongside the
diagonal spreads).

Separate files for REAL vs PAPER so they never mix:
  ~/.vix_suite/assignments_real.json
  ~/.vix_suite/assignments_paper.json

Schema matches the existing assignment_engine output:
  assignment_id, date, shares, entry_price, status, strategy_context

Extended with: mode, exit_date, exit_price, realized_pnl, notes, created_at
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

STORE_DIR = Path.home() / ".vix_suite"
_FILES = {
    "real":  STORE_DIR / "assignments_real.json",
    "paper": STORE_DIR / "assignments_paper.json",
}

VALID_MODES = ("real", "paper")


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class AssignmentEvent:
    # Core fields — matches existing assignment_engine schema exactly
    assignment_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    date:             str   = ""        # ISO date of entry  e.g. "2026-03-13"
    shares:           int   = 0         # negative = short, positive = long
    entry_price:      float = 0.0
    status:           str   = "open"    # "open" | "closed"
    strategy_context: str   = ""        # e.g. "V4 Spike Trade"

    # Extended fields
    mode:             str   = "real"    # "real" | "paper"
    exit_date:        str   = ""
    exit_price:       float = 0.0
    realized_pnl:     float = 0.0
    notes:            str   = ""
    created_at:       str   = field(default_factory=lambda: datetime.now().isoformat())

    # ── computed ──────────────────────────────────────────────────────────────

    @property
    def direction(self) -> str:
        return "SHORT" if self.shares < 0 else "LONG"

    @property
    def abs_shares(self) -> int:
        return abs(self.shares)

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.entry_price

    @property
    def days_held(self) -> int:
        if not self.date:
            return 0
        try:
            d = date.fromisoformat(self.date)
            if self.status == "closed" and self.exit_date:
                ex = date.fromisoformat(self.exit_date)
                return (ex - d).days
            return (date.today() - d).days
        except ValueError:
            return 0

    def unrealized_pnl(self, current_price: float) -> float:
        """P&L at a given current price. Negative shares = short."""
        return self.shares * (current_price - self.entry_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.notional * 100


# ── Internal helpers ──────────────────────────────────────────────────────────

def _store_file(mode: str) -> Path:
    assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
    return _FILES[mode]


def _load_raw(mode: str) -> List[dict]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    f = _store_file(mode)
    if not f.exists():
        return []
    with open(f) as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


def _save_raw(mode: str, records: List[dict]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_store_file(mode), "w") as fh:
        json.dump(records, fh, indent=2)


def _to_event(d: dict) -> AssignmentEvent:
    known = set(AssignmentEvent.__dataclass_fields__)
    return AssignmentEvent(**{k: v for k, v in d.items() if k in known})


# ── Public CRUD API ───────────────────────────────────────────────────────────

def load_all(mode: str) -> List[AssignmentEvent]:
    """Load all events (open + closed) for the given mode."""
    return [_to_event(r) for r in _load_raw(mode)]


def load_open(mode: str) -> List[AssignmentEvent]:
    return [e for e in load_all(mode) if e.status == "open"]


def load_closed(mode: str) -> List[AssignmentEvent]:
    return [e for e in load_all(mode) if e.status == "closed"]


def save_event(event: AssignmentEvent) -> None:
    """Insert or update (upsert by assignment_id)."""
    mode    = event.mode
    records = _load_raw(mode)
    data    = asdict(event)
    for i, r in enumerate(records):
        if r.get("assignment_id") == event.assignment_id:
            records[i] = data
            _save_raw(mode, records)
            return
    records.append(data)
    _save_raw(mode, records)


def update_event(mode: str, assignment_id: str, **fields) -> Optional[AssignmentEvent]:
    """Patch specific fields on an existing event."""
    records = _load_raw(mode)
    for i, r in enumerate(records):
        if r.get("assignment_id") == assignment_id:
            r.update(fields)
            records[i] = r
            _save_raw(mode, records)
            return _to_event(r)
    return None


def delete_event(mode: str, assignment_id: str) -> bool:
    """Hard-delete a record. Returns True if found and removed."""
    records = _load_raw(mode)
    new_records = [r for r in records if r.get("assignment_id") != assignment_id]
    if len(new_records) == len(records):
        return False
    _save_raw(mode, new_records)
    return True


def close_event(
    mode:         str,
    assignment_id:str,
    exit_price:   float,
    exit_date:    str = "",
) -> Optional[AssignmentEvent]:
    """Mark an event as closed and compute realized P&L."""
    records = _load_raw(mode)
    for i, r in enumerate(records):
        if r.get("assignment_id") == assignment_id:
            ev = _to_event(r)
            ev.status     = "closed"
            ev.exit_price = exit_price
            ev.exit_date  = exit_date or date.today().isoformat()
            # P&L: short = (entry - exit) × abs_shares
            ev.realized_pnl = ev.shares * (exit_price - ev.entry_price)
            records[i] = asdict(ev)
            _save_raw(mode, records)
            return ev
    return None


# ── Migration helper ──────────────────────────────────────────────────────────

def migrate_from_assignment_engine(mode: str = "real") -> int:
    """
    Try to import records from the existing assignment_engine store.
    Safe to call repeatedly — skips IDs already present.
    Returns number of records imported.
    """
    try:
        import importlib
        ae = importlib.import_module("assignment_engine")
        store = ae.get_assignment_store()
        events = store.events
    except Exception:
        return 0

    existing_ids = {r.get("assignment_id") for r in _load_raw(mode)}
    imported = 0

    for ev in events:
        aid = getattr(ev, "assignment_id", None)
        if not aid or aid in existing_ids:
            continue
        new_ev = AssignmentEvent(
            assignment_id    = aid,
            date             = str(getattr(ev, "date", "")),
            shares           = int(getattr(ev, "shares", 0)),
            entry_price      = float(getattr(ev, "entry_price", 0.0)),
            status           = str(getattr(ev, "status", "open")),
            strategy_context = str(getattr(ev, "strategy_context", "")),
            mode             = mode,
        )
        save_event(new_ev)
        imported += 1

    return imported


# ── Summary ───────────────────────────────────────────────────────────────────

def portfolio_summary(mode: str, current_price: float = 0.0) -> dict:
    open_evs = load_open(mode)
    closed_evs = load_closed(mode)

    total_notional  = sum(e.notional for e in open_evs)
    total_unrealized = sum(e.unrealized_pnl(current_price) for e in open_evs) if current_price else 0.0
    total_realized  = sum(e.realized_pnl for e in closed_evs)
    net_shares      = sum(e.shares for e in open_evs)

    return {
        "open_count":        len(open_evs),
        "closed_count":      len(closed_evs),
        "net_shares":        net_shares,
        "total_notional":    total_notional,
        "total_unrealized":  total_unrealized,
        "total_realized":    total_realized,
        "net_pnl":           total_unrealized + total_realized,
    }
