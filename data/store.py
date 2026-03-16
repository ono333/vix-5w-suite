"""
Position Store
--------------
Persistent JSON storage for real trade positions.
Supports multiple concurrent positions per strategy.

Storage path: ~/.vix_suite/real_positions.json
Each position has a unique UUID and full trade metadata.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional


STORE_DIR  = Path.home() / ".vix_suite"
STORE_FILE = STORE_DIR / "real_positions.json"


# ── Position dataclass ────────────────────────────────────────────────────────

@dataclass
class Position:
    # Identity
    pos_id:      str       = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy:    str       = ""    # V1-V5
    status:      str       = "OPEN"  # OPEN | CLOSED | ROLLED

    # Entry metadata
    entry_date:  str       = ""    # ISO date string
    entry_regime:str       = ""
    vix_at_entry:float     = 0.0
    uvxy_at_entry:float    = 0.0
    contracts:   int       = 1

    # Long leg (LEAP)
    long_strike: float     = 0.0
    long_expiry: str       = ""   # ISO date
    long_entry_ask:float   = 0.0  # price paid

    # Short leg (weekly)
    short_strike:float     = 0.0
    short_expiry:str       = ""   # ISO date
    short_entry_bid:float  = 0.0  # credit received

    # Net position cost
    net_debit:   float     = 0.0  # long_ask - short_bid (what you paid)

    # Management targets
    profit_target_net: float = 0.0   # close when current_net >= this
    stop_loss_net:     float = 0.0   # close when current_net <= this
    roll_at_dte:       int   = 4

    # Current state (updated at each mark-to-market)
    current_net:     float  = 0.0
    last_marked:     str    = ""

    # Exit metadata (filled when closed)
    exit_date:   str        = ""
    exit_net:    float      = 0.0
    realized_pnl:float      = 0.0
    exit_reason: str        = ""   # TP | SL | ROLL | EXPIRY | MANUAL

    # Roll tracking (if this position was rolled from another)
    parent_id:   str        = ""   # pos_id of the position that was rolled
    roll_count:  int        = 0    # how many times this short leg has been rolled

    # Notes
    notes:       str        = ""

    # ── computed properties ───────────────────────────────────────────────

    @property
    def long_expiry_date(self) -> date:
        return date.fromisoformat(self.long_expiry) if self.long_expiry else date.today()

    @property
    def short_expiry_date(self) -> date:
        return date.fromisoformat(self.short_expiry) if self.short_expiry else date.today()

    @property
    def short_dte(self) -> int:
        return max((self.short_expiry_date - date.today()).days, 0)

    @property
    def long_dte(self) -> int:
        return max((self.long_expiry_date - date.today()).days, 0)

    @property
    def unrealized_pnl(self) -> float:
        if self.current_net == 0.0:
            return 0.0
        return (self.current_net - self.net_debit) * self.contracts * 100

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.net_debit == 0:
            return 0.0
        return (self.current_net - self.net_debit) / abs(self.net_debit) * 100

    @property
    def days_held(self) -> int:
        if not self.entry_date:
            return 0
        entry = date.fromisoformat(self.entry_date)
        return (date.today() - entry).days

    @property
    def health(self) -> str:
        """HEALTHY | WATCH | CRITICAL | ROLL_NOW"""
        if self.short_dte <= self.roll_at_dte:
            return "ROLL_NOW"
        if self.current_net > 0 and self.current_net >= self.profit_target_net:
            return "TAKE_PROFIT"
        if self.current_net > 0 and self.current_net <= self.stop_loss_net:
            return "STOP_LOSS"
        if self.unrealized_pnl_pct < -20:
            return "CRITICAL"
        if self.unrealized_pnl_pct < -10 or self.short_dte <= self.roll_at_dte + 3:
            return "WATCH"
        return "HEALTHY"

    @property
    def health_color(self) -> str:
        c = {
            "HEALTHY":     "#22c55e",
            "WATCH":       "#f59e0b",
            "CRITICAL":    "#ef4444",
            "ROLL_NOW":    "#8b5cf6",
            "TAKE_PROFIT": "#06b6d4",
            "STOP_LOSS":   "#ef4444",
        }
        return c.get(self.health, "#94a3b8")


# ── Store helpers ─────────────────────────────────────────────────────────────

def _load_raw() -> List[dict]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_FILE.exists():
        return []
    with open(STORE_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_raw(records: List[dict]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORE_FILE, "w") as f:
        json.dump(records, f, indent=2)


def _to_position(d: dict) -> Position:
    known = {k for k in Position.__dataclass_fields__}
    return Position(**{k: v for k, v in d.items() if k in known})


# ── Public API ────────────────────────────────────────────────────────────────

def load_all() -> List[Position]:
    return [_to_position(r) for r in _load_raw()]


def load_open() -> List[Position]:
    return [p for p in load_all() if p.status == "OPEN"]


def load_closed() -> List[Position]:
    return [p for p in load_all() if p.status == "CLOSED"]


def save_position(pos: Position) -> None:
    records = _load_raw()
    data    = asdict(pos)
    for i, r in enumerate(records):
        if r.get("pos_id") == pos.pos_id:
            records[i] = data
            _save_raw(records)
            return
    records.append(data)
    _save_raw(records)


def delete_position(pos_id: str) -> None:
    records = [r for r in _load_raw() if r.get("pos_id") != pos_id]
    _save_raw(records)


def close_position(
    pos_id:      str,
    exit_net:    float,
    exit_reason: str = "MANUAL",
) -> Optional[Position]:
    records = _load_raw()
    for i, r in enumerate(records):
        if r.get("pos_id") == pos_id:
            pos = _to_position(r)
            pos.status       = "CLOSED"
            pos.exit_date    = date.today().isoformat()
            pos.exit_net     = exit_net
            pos.realized_pnl = (exit_net - pos.net_debit) * pos.contracts * 100
            pos.exit_reason  = exit_reason
            records[i]       = asdict(pos)
            _save_raw(records)
            return pos
    return None


def mark_position(pos_id: str, current_net: float) -> Optional[Position]:
    records = _load_raw()
    for i, r in enumerate(records):
        if r.get("pos_id") == pos_id:
            pos = _to_position(r)
            pos.current_net = current_net
            pos.last_marked = datetime.now().isoformat()
            records[i]      = asdict(pos)
            _save_raw(records)
            return pos
    return None


def portfolio_summary(positions: List[Position] = None) -> dict:
    if positions is None:
        positions = load_open()
    open_pos  = [p for p in positions if p.status == "OPEN"]
    total_pnl = sum(p.unrealized_pnl for p in open_pos)
    at_risk   = sum(p.net_debit * p.contracts * 100 for p in open_pos)
    return {
        "open_count":  len(open_pos),
        "total_unrealized_pnl": total_pnl,
        "total_at_risk": at_risk,
        "roll_due":    sum(1 for p in open_pos if p.health == "ROLL_NOW"),
        "take_profit": sum(1 for p in open_pos if p.health == "TAKE_PROFIT"),
        "stop_loss":   sum(1 for p in open_pos if p.health == "STOP_LOSS"),
        "healthy":     sum(1 for p in open_pos if p.health == "HEALTHY"),
    }
