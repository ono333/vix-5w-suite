"""
real_trade_log.py
─────────────────
Real money trade log — separate from paper trade log.
Stores to ~/.vix_suite/real_trade_log.json
Same data structure as trade_log.py but with:
  - broker field (Fidelity, IB, etc.)
  - account_id field
  - actual_fill_price (vs estimated)
  - commission tracking
  - slippage tracking (fill vs mid-price at order time)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict

# ── Storage location (separate from paper trades)
REAL_LOG_PATH = Path.home() / ".vix_suite" / "real_trade_log.json"

# ── Supported brokers
BROKERS = ["Fidelity", "IB", "Schwab", "TD", "Other"]


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES  (mirroring trade_log.py + real-trade fields)
# ═══════════════════════════════════════════════════════════

@dataclass
class RealShortLeg:
    leg_id:           str
    position_id:      str
    entry_date:       str
    strike:           float
    expiration_date:  str
    entry_credit:     float       # estimated mid at order time
    fill_price:       float       # actual fill (can differ)
    contracts:        int
    broker:           str         = "Fidelity"
    account_id:       str         = ""
    commission:       float       = 0.65  # per contract (Fidelity: $0.65)
    slippage:         float       = 0.0   # fill_price - entry_credit (negative = better fill)
    status:           str         = "open"   # open / rolled / expired / closed
    current_price:    float       = 0.0
    exit_date:        Optional[str]   = None
    exit_price:       Optional[float] = None
    exit_fill_price:  Optional[float] = None  # actual exit fill
    exit_commission:  float       = 0.65
    exit_reason:      Optional[str]   = None
    pnl:              float       = 0.0
    notes:            str         = ""
    created_at:       str         = field(default_factory=lambda: datetime.now().isoformat())

    def days_to_expiry(self) -> int:
        if not self.expiration_date:
            return 0
        try:
            exp = date.fromisoformat(self.expiration_date)
            return max(0, (exp - date.today()).days)
        except:
            return 0

    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def total_commission(self) -> float:
        return (self.commission + self.exit_commission) * self.contracts

    @property
    def net_credit(self) -> float:
        """Actual net credit after commissions."""
        gross = self.fill_price * self.contracts * 100
        costs = self.total_commission
        return gross - costs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealRollRecord:
    roll_id:          str
    position_id:      str
    roll_date:        str
    old_strike:       float
    old_expiration:   str
    old_exit_price:   float
    old_fill_price:   float        # actual fill on buy-back
    new_strike:       float
    new_expiration:   str
    new_credit:       float        # mid at order time
    new_fill_price:   float        # actual fill on new short
    roll_credit:      float        # new_fill_price - old_fill_price
    underlying_price: float
    contracts:        int
    roll_type:        str          = "short"  # short / long
    regime:           str          = ""
    broker:           str          = "Fidelity"
    account_id:       str          = ""
    commission:       float        = 0.0      # total for both legs
    roll_reason:      str          = ""       # "delta_trigger" / "order_roll" / "manual"
    notes:            str          = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealDiagonalPosition:
    position_id:       str
    variant_id:        str
    variant_name:      str
    entry_date:        str
    entry_regime:      str
    entry_vix_level:   float
    entry_percentile:  float
    contracts:         int
    broker:            str         = "Fidelity"
    account_id:        str         = ""

    # Long leg
    long_strike:       float       = 0.0
    long_expiration:   str         = ""
    long_entry_price:  float       = 0.0     # estimated mid
    long_fill_price:   float       = 0.0     # actual fill
    long_commission:   float       = 0.65    # per contract
    long_current_price: float      = 0.0
    long_status:       str         = "open"

    # Short legs
    short_legs:        List[RealShortLeg]  = field(default_factory=list)
    roll_history:      List[RealRollRecord] = field(default_factory=list)

    # Position metadata
    status:            str         = "open"   # open / closed
    close_date:        Optional[str]  = None
    close_reason:      Optional[str]  = None
    notes:             str         = ""
    created_at:        str         = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:        str         = field(default_factory=lambda: datetime.now().isoformat())

    # ── Computed properties ─────────────────────────────────

    @property
    def current_short_leg(self) -> Optional[RealShortLeg]:
        for leg in reversed(self.short_legs):
            if leg.status == "open":
                return leg
        return None

    @property
    def long_cost(self) -> float:
        """Actual long debit paid including commission."""
        return (self.long_fill_price * self.contracts * 100
                + self.long_commission * self.contracts)

    @property
    def gross_short_credits(self) -> float:
        return sum(l.fill_price * l.contracts * 100 for l in self.short_legs)

    @property
    def total_buybacks(self) -> float:
        return sum((l.exit_fill_price or 0) * l.contracts * 100
                   for l in self.short_legs
                   if l.status in ("rolled", "closed"))

    @property
    def total_commissions(self) -> float:
        lc = self.long_commission * self.contracts
        sc = sum(l.total_commission for l in self.short_legs)
        return lc + sc

    @property
    def net_short_credits(self) -> float:
        return self.gross_short_credits - self.total_buybacks - self.total_commissions

    @property
    def long_pnl(self) -> float:
        if self.long_current_price <= 0:
            return 0.0
        return ((self.long_current_price - self.long_fill_price)
                * self.contracts * 100)

    @property
    def total_pnl(self) -> float:
        return self.long_pnl + self.net_short_credits

    @property
    def short_coverage_pct(self) -> float:
        if self.long_cost <= 0:
            return 0.0
        return min(100.0, self.net_short_credits / self.long_cost * 100)

    @property
    def total_slippage(self) -> float:
        """Total slippage vs mid-price estimates."""
        entry_slip = (self.long_fill_price - self.long_entry_price) * self.contracts * 100
        short_slip = sum(l.slippage * l.contracts * 100 for l in self.short_legs)
        return entry_slip + short_slip

    def days_to_expiry(self) -> int:
        short = self.current_short_leg
        return short.days_to_expiry() if short else -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["short_legs"]    = [l.to_dict() for l in self.short_legs]
        d["roll_history"]  = [r.to_dict() for r in self.roll_history]
        return d


# ═══════════════════════════════════════════════════════════
# TRADE LOG CLASS
# ═══════════════════════════════════════════════════════════

class RealTradeLog:
    def __init__(self, path: Path = REAL_LOG_PATH):
        self.path = path
        self.diagonal_positions: Dict[str, RealDiagonalPosition] = {}
        self.history: list = []
        self.updated_at: str = datetime.now().isoformat()
        self._load()

    # ── Persistence ─────────────────────────────────────────

    def _load(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save()
            return
        try:
            data = json.loads(self.path.read_text())
            for pid, pd in data.get("diagonal_positions", {}).items():
                short_legs = [RealShortLeg(**l) for l in pd.pop("short_legs", [])]
                roll_history = [RealRollRecord(**r) for r in pd.pop("roll_history", [])]
                self.diagonal_positions[pid] = RealDiagonalPosition(
                    **pd, short_legs=short_legs, roll_history=roll_history)
            self.history    = data.get("history", [])
            self.updated_at = data.get("updated_at", self.updated_at)
        except Exception as e:
            print(f"Warning: could not load real trade log: {e}")

    def _save(self):
        self.updated_at = datetime.now().isoformat()
        data = {
            "diagonal_positions": {
                pid: pos.to_dict()
                for pid, pos in self.diagonal_positions.items()
            },
            "history":    self.history,
            "updated_at": self.updated_at,
        }
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def save(self):
        self._save()

    # ── Position management ──────────────────────────────────

    def open_diagonal(
        self,
        variant_id:        str,
        variant_name:      str,
        regime:            str,
        vix_level:         float,
        vix_percentile:    float,
        contracts:         int,
        long_strike:       float,
        long_expiration:   str,
        long_entry_price:  float,    # mid at order time
        long_fill_price:   float,    # actual fill
        short_strike:      float,
        short_expiration:  str,
        short_credit:      float,    # mid at order time
        short_fill_price:  float,    # actual fill
        broker:            str       = "Fidelity",
        account_id:        str       = "",
        long_commission:   float     = 0.65,
        short_commission:  float     = 0.65,
        notes:             str       = "",
    ) -> RealDiagonalPosition:

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        position_id = f"{variant_id[:2].upper()}-REAL-{ts}"
        leg_id      = f"{position_id}-S1"
        entry_date  = date.today().isoformat()

        slippage = short_fill_price - short_credit

        short_leg = RealShortLeg(
            leg_id          = leg_id,
            position_id     = position_id,
            entry_date      = entry_date,
            strike          = short_strike,
            expiration_date = short_expiration,
            entry_credit    = short_credit,
            fill_price      = short_fill_price,
            contracts       = contracts,
            broker          = broker,
            account_id      = account_id,
            commission      = short_commission,
            slippage        = slippage,
        )

        pos = RealDiagonalPosition(
            position_id      = position_id,
            variant_id       = variant_id,
            variant_name     = variant_name,
            entry_date       = entry_date,
            entry_regime     = regime,
            entry_vix_level  = vix_level,
            entry_percentile = vix_percentile,
            contracts        = contracts,
            broker           = broker,
            account_id       = account_id,
            long_strike      = long_strike,
            long_expiration  = long_expiration,
            long_entry_price = long_entry_price,
            long_fill_price  = long_fill_price,
            long_commission  = long_commission,
            short_legs       = [short_leg],
            notes            = notes,
        )

        self.diagonal_positions[position_id] = pos
        self._save()
        return pos

    def roll_short(
        self,
        position_id:      str,
        old_exit_price:   float,    # mid at roll time
        old_fill_price:   float,    # actual fill on buy-back
        new_strike:       float,
        new_expiration:   str,
        new_credit:       float,    # mid at order time
        new_fill_price:   float,    # actual fill on new short
        underlying_price: float,
        regime:           str       = "",
        broker:           str       = "Fidelity",
        account_id:       str       = "",
        commission:       float     = 1.30,  # both legs combined
        roll_reason:      str       = "order_roll",
        notes:            str       = "",
    ) -> Optional[RealRollRecord]:

        pos = self.diagonal_positions.get(position_id)
        if not pos:
            return None

        old_short = pos.current_short_leg
        if not old_short:
            return None

        # Close old short
        old_short.status         = "rolled"
        old_short.exit_date      = date.today().isoformat()
        old_short.exit_price     = old_exit_price
        old_short.exit_fill_price = old_fill_price
        old_short.exit_reason    = "rolled"
        old_short.pnl            = ((old_short.fill_price - old_fill_price)
                                     * old_short.contracts * 100)

        # Open new short
        n_legs   = len(pos.short_legs) + 1
        leg_id   = f"{position_id}-S{n_legs}"
        slippage = new_fill_price - new_credit

        new_leg = RealShortLeg(
            leg_id          = leg_id,
            position_id     = position_id,
            entry_date      = date.today().isoformat(),
            strike          = new_strike,
            expiration_date = new_expiration,
            entry_credit    = new_credit,
            fill_price      = new_fill_price,
            contracts       = old_short.contracts,
            broker          = broker,
            account_id      = account_id,
            commission      = commission / 2,
            exit_commission = commission / 2,
            slippage        = slippage,
        )
        pos.short_legs.append(new_leg)

        # Roll record
        roll_id = f"{position_id}-R{len(pos.roll_history)+1}"
        roll = RealRollRecord(
            roll_id          = roll_id,
            position_id      = position_id,
            roll_date        = date.today().isoformat(),
            old_strike       = old_short.strike,
            old_expiration   = old_short.expiration_date,
            old_exit_price   = old_exit_price,
            old_fill_price   = old_fill_price,
            new_strike       = new_strike,
            new_expiration   = new_expiration,
            new_credit       = new_credit,
            new_fill_price   = new_fill_price,
            roll_credit      = new_fill_price - old_fill_price,
            underlying_price = underlying_price,
            contracts        = old_short.contracts,
            regime           = regime,
            broker           = broker,
            account_id       = account_id,
            commission       = commission,
            roll_reason      = roll_reason,
            notes            = notes,
        )
        pos.roll_history.append(roll)
        pos.updated_at = datetime.now().isoformat()
        self._save()
        return roll

    def close_position(
        self,
        position_id:   str,
        reason:        str = "manual",
        notes:         str = "",
    ):
        pos = self.diagonal_positions.get(position_id)
        if pos:
            pos.status       = "closed"
            pos.close_date   = date.today().isoformat()
            pos.close_reason = reason
            pos.notes        = notes
            pos.updated_at   = datetime.now().isoformat()
            self._save()

    # ── Queries ──────────────────────────────────────────────

    def open_positions(self) -> Dict[str, RealDiagonalPosition]:
        return {pid: p for pid, p in self.diagonal_positions.items()
                if p.status == "open"}

    def positions_for_variant(self, variant_id: str) -> list:
        return [p for p in self.diagonal_positions.values()
                if p.variant_id.upper() == variant_id.upper()
                and p.status == "open"]

    def summary(self) -> dict:
        open_pos = self.open_positions()
        total_pnl        = sum(p.total_pnl        for p in open_pos.values())
        total_commissions = sum(p.total_commissions for p in open_pos.values())
        total_slippage   = sum(p.total_slippage    for p in open_pos.values())
        return dict(
            open_count       = len(open_pos),
            total_pnl        = total_pnl,
            total_commissions = total_commissions,
            total_slippage   = total_slippage,
            net_after_costs  = total_pnl - total_commissions,
        )


# ── Singleton ────────────────────────────────────────────────
_real_log_instance: Optional[RealTradeLog] = None

def get_real_trade_log() -> RealTradeLog:
    global _real_log_instance
    if _real_log_instance is None:
        _real_log_instance = RealTradeLog()
    return _real_log_instance

def reset_real_trade_log_cache():
    global _real_log_instance
    _real_log_instance = None
