"""
Assignment Detection & Covered Option Engine
Handles UVXY option assignment → stock position → covered call harvest cycle.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import json

STORAGE_DIR  = Path.home() / ".vix_suite"
ASSIGNMENT_FILE = STORAGE_DIR / "assignments.json"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AssignmentEvent:
    assignment_id:       str
    date:                str
    symbol:              str
    shares:              int          # negative = short
    entry_price:         float
    total_exposure:      float
    strategy_context:    str          = ""
    position_state:      str          = "SHORT"   # SHORT / LONG
    position_origin:     str          = "option_assignment"
    assignment_flag:     bool         = True
    covered_option_active: bool       = False
    decision_score:      float        = 0.0
    decision:            str          = "HOLD"    # CLOSE / HOLD / HARVEST
    decision_confidence: str          = "LOW"
    status:              str          = "open"    # open / closed
    notes:               str          = ""
    created_at:          str          = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class CoveredCallSuggestion:
    symbol:         str
    shares:         int
    current_price:  float
    suggested_strike: float
    expiration_dte: int
    expiration_date: str
    target_delta:   float
    contracts:      int
    est_premium_lo: float
    est_premium_hi: float
    rationale:      str


# ── Storage ───────────────────────────────────────────────────────────────────

class AssignmentStore:
    def __init__(self):
        self.events: list[AssignmentEvent] = []
        self._load()

    def _load(self):
        if ASSIGNMENT_FILE.exists():
            try:
                raw = json.loads(ASSIGNMENT_FILE.read_text())
                self.events = [AssignmentEvent.from_dict(r) for r in raw]
            except Exception:
                self.events = []

    def save(self):
        ASSIGNMENT_FILE.write_text(
            json.dumps([e.to_dict() for e in self.events], indent=2))

    def open_assignments(self) -> list[AssignmentEvent]:
        return [e for e in self.events if e.status == "open"]

    def add(self, event: AssignmentEvent):
        self.events.append(event)
        self.save()

    def update(self, assignment_id: str, **kwargs):
        for e in self.events:
            if e.assignment_id == assignment_id:
                for k, v in kwargs.items():
                    if hasattr(e, k):
                        setattr(e, k, v)
        self.save()


def get_assignment_store() -> AssignmentStore:
    return AssignmentStore()


# ── Decision Score ────────────────────────────────────────────────────────────

def compute_decision_score(snap) -> tuple[float, str, str]:
    """
    Assignment Decision Score (0-100).
    Higher = more favorable to harvest premium (sell covered calls).

    Formula:
    0.35 × spike_exhaustion
    0.25 × VIX_percentile
    0.25 × UVXY_percentile
    0.15 × term_structure_normalization
    """
    spike_norm = snap.spike_score                      # already 0-100
    vix_norm   = snap.vix_pct  * 100
    uvxy_norm  = snap.uvxy_pct * 100
    # Term structure: contango (iv_ratio < 1) = bullish for harvest
    # Strong contango = 100, flat = 50, backwardation = 0
    if snap.iv_ratio < 0.90:   ts_norm = 100.0
    elif snap.iv_ratio < 0.95: ts_norm = 75.0
    elif snap.iv_ratio < 1.00: ts_norm = 50.0
    elif snap.iv_ratio < 1.05: ts_norm = 25.0
    else:                      ts_norm = 0.0

    score = (0.35 * spike_norm +
             0.25 * vix_norm   +
             0.25 * uvxy_norm  +
             0.15 * ts_norm)
    score = round(min(100, max(0, score)), 1)

    if score >= 70:
        decision, confidence = "HARVEST PREMIUM", "HIGH"
    elif score >= 40:
        decision, confidence = "HOLD / MONITOR",  "MEDIUM"
    else:
        decision, confidence = "CLOSE POSITION",  "HIGH"

    # Override: crisis conditions → always close
    if snap.vvix > 130 and snap.vix_pct > 0.95:
        decision, confidence = "CLOSE POSITION", "HIGH"

    return score, decision, confidence


# ── Covered Call Suggestion ───────────────────────────────────────────────────

def suggest_covered_call(
    shares: int,
    current_price: float,
    snap,
    dte_target: int = 10,
) -> CoveredCallSuggestion:
    """
    Suggest covered call strike based on volatility regime.
    Wider OTM in high vol (more cushion), tighter in calm.
    """
    from datetime import timedelta

    abs_shares = abs(shares)
    contracts  = abs_shares // 100

    # Strike OTM% based on regime
    if snap.vix_pct >= 0.90:   otm_pct = 0.12   # 12% OTM — panic, wide cushion
    elif snap.vix_pct >= 0.80: otm_pct = 0.10
    elif snap.vix_pct >= 0.65: otm_pct = 0.08
    else:                      otm_pct = 0.06

    strike = round(current_price * (1 + otm_pct), 0)

    # Premium estimate: higher vol → higher premium
    vol_mult  = 1.5 if snap.vix_pct >= 0.85 else 1.0
    base_prem = current_price * 0.04 * vol_mult   # ~4% of price per week
    dte_factor = dte_target / 7
    prem_per_contract = round(base_prem * dte_factor * 100, 0)
    est_lo = round(prem_per_contract * contracts * 0.75, 0)
    est_hi = round(prem_per_contract * contracts * 1.25, 0)

    exp_date = (date.today() + timedelta(days=dte_target)).strftime("%Y-%m-%d")

    rationale = (
        f"{otm_pct:.0%} OTM (VIX at {snap.vix_pct:.0%} percentile) | "
        f"Harvest spike IV | "
        f"{'Wide cushion — crisis regime' if snap.vix_pct >= 0.85 else 'Standard harvest'}"
    )

    return CoveredCallSuggestion(
        symbol          = "UVXY",
        shares          = abs_shares,
        current_price   = current_price,
        suggested_strike= strike,
        expiration_dte  = dte_target,
        expiration_date = exp_date,
        target_delta    = 0.20,
        contracts       = contracts,
        est_premium_lo  = est_lo,
        est_premium_hi  = est_hi,
        rationale       = rationale,
    )


# ── Log helpers ───────────────────────────────────────────────────────────────

def income_projection(sugg, entry_price: float, total_exposure: float) -> dict:
    """
    Weekly/monthly income projection and yield vs exposure.
    """
    weekly_income  = round((sugg.est_premium_lo + sugg.est_premium_hi) / 2, 0)
    monthly_income = round(weekly_income * 4.33, 0)
    yield_pct      = round(weekly_income / max(total_exposure, 1) * 100, 2) if total_exposure else 0
    monthly_yield  = round(yield_pct * 4.33, 2)
    return {
        "weekly_income":   weekly_income,
        "monthly_income":  monthly_income,
        "weekly_yield_pct":  yield_pct,
        "monthly_yield_pct": monthly_yield,
    }


def log_assignment(
    symbol: str,
    shares: int,
    entry_price: float,
    strategy_context: str = "",
    snap=None,
) -> AssignmentEvent:
    import uuid
    store = get_assignment_store()
    score, decision, confidence = (
        compute_decision_score(snap) if snap else (50.0, "HOLD / MONITOR", "LOW"))

    event = AssignmentEvent(
        assignment_id    = str(uuid.uuid4())[:8],
        date             = date.today().isoformat(),
        symbol           = symbol,
        shares           = shares,
        entry_price      = entry_price,
        total_exposure   = round(abs(shares) * entry_price, 2),
        strategy_context = strategy_context,
        position_state   = "SHORT" if shares < 0 else "LONG",
        covered_option_active = decision == "HARVEST PREMIUM",
        decision_score   = score,
        decision         = decision,
        decision_confidence = confidence,
    )
    store.add(event)
    return event


# ── Extended helpers ──────────────────────────────────────────────────────────

def days_since_assignment(event: AssignmentEvent) -> int:
    try:
        return (date.today() - date.fromisoformat(event.date)).days
    except Exception:
        return 0


def unrealized_pnl(event: AssignmentEvent, current_price: float) -> float:
    """
    Short position: profit when price falls below entry.
    Long position:  profit when price rises above entry.
    """
    if event.position_state == "SHORT":
        return round((event.entry_price - current_price) * abs(event.shares), 2)
    else:
        return round((current_price - event.entry_price) * abs(event.shares), 2)


def risk_level(current_price: float, entry_price: float,
               position_state: str) -> tuple[str, float, float]:
    """
    Returns risk_label, warning_threshold, crisis_threshold.
    Short UVXY: risk = price rising above entry.
    """
    if position_state == "SHORT":
        warning  = round(entry_price * 1.20, 2)
        crisis   = round(entry_price * 1.50, 2)
        if current_price >= crisis:   label = "🔴 CRISIS"
        elif current_price >= warning: label = "🟡 WARNING"
        else:                          label = "🟢 SAFE"
    else:
        warning  = round(entry_price * 0.80, 2)
        crisis   = round(entry_price * 0.50, 2)
        if current_price <= crisis:   label = "🔴 CRISIS"
        elif current_price <= warning: label = "🟡 WARNING"
        else:                          label = "🟢 SAFE"
    return label, warning, crisis


def suggest_short_put(
    shares: int,
    current_price: float,
    snap,
    dte_target: int = 10,
) -> CoveredCallSuggestion:
    """
    For long UVXY assignment — sell puts below to harvest premium.
    """
    from datetime import timedelta
    abs_shares = abs(shares)
    contracts  = abs_shares // 100

    otm_pct = (0.12 if snap.vix_pct >= 0.90 else
               0.10 if snap.vix_pct >= 0.80 else 0.08)
    strike   = round(current_price * (1 - otm_pct), 0)

    vol_mult  = 1.5 if snap.vix_pct >= 0.85 else 1.0
    base_prem = current_price * 0.04 * vol_mult
    dte_factor = dte_target / 7
    prem_per   = round(base_prem * dte_factor * 100, 0)
    est_lo     = round(prem_per * contracts * 0.75, 0)
    est_hi     = round(prem_per * contracts * 1.25, 0)
    exp_date   = (date.today() + timedelta(days=dte_target)).strftime("%Y-%m-%d")

    return CoveredCallSuggestion(
        symbol="UVXY", shares=abs_shares, current_price=current_price,
        suggested_strike=strike, expiration_dte=dte_target,
        expiration_date=exp_date, target_delta=0.20, contracts=contracts,
        est_premium_lo=est_lo, est_premium_hi=est_hi,
        rationale=f"{otm_pct:.0%} OTM put — income against long UVXY assignment",
    )


def portfolio_structure(trade_log=None, assignments=None) -> list[dict]:
    """
    Returns portfolio layers for display:
    Decay Engine / Tail Hedge / Income Layer / Assignment Layer
    """
    layers = []

    if assignments:
        for evt in assignments:
            layers.append({
                "layer": "Decay Engine" if evt.position_state == "SHORT" else "Long Exposure",
                "type":  f"{'Short' if evt.position_state=='SHORT' else 'Long'} {evt.symbol} Shares",
                "detail": f"{abs(evt.shares):,} shares @ ${evt.entry_price:.2f}",
                "status": evt.decision,
            })

    if trade_log:
        try:
            positions = trade_log.open_positions()
            if isinstance(positions, dict):
                positions = list(positions.values())
            for pos in positions:
                # Long leg = tail hedge
                if hasattr(pos, "long_strike"):
                    layers.append({
                        "layer": "Tail Hedge",
                        "type":  f"Long {getattr(pos,'symbol','UVXY')} Calls",
                        "detail": f"Strike ${pos.long_strike} exp {getattr(pos,'long_expiration','')}",
                        "status": "ACTIVE",
                    })
                # Short legs = income layer
                for leg in getattr(pos, "short_legs", []):
                    is_open = getattr(leg, "status", "") == "open"
                    exp_ok  = True
                    try:
                        from datetime import date as _date
                        import pandas as pd
                        exp_ok = pd.Timestamp(leg.expiration_date) >= pd.Timestamp.now()
                    except Exception:
                        pass
                    if is_open and exp_ok:
                        layers.append({
                            "layer": "Income Layer",
                            "type":  f"Short Weekly Call — {getattr(pos,'variant_name','')}",
                            "detail": f"Strike ${leg.strike} exp {leg.expiration_date}",
                            "status": "ACTIVE",
                        })
        except Exception:
            pass

    return layers


# ── Position Sizing Risk Check ────────────────────────────────────────────────

def position_risk_check(
    shares: int,
    entry_price: float,
    current_price: float,
    account_size: float,
    snap,
) -> dict:
    """
    Comprehensive position sizing and margin stress analysis.
    Returns risk metrics and trim recommendations.
    """
    abs_shares   = abs(shares)
    is_short     = shares < 0

    # Current exposure and P&L
    exposure     = round(abs_shares * current_price, 2)
    entry_value  = round(abs_shares * entry_price, 2)
    unrealized   = round((entry_price - current_price) * abs_shares, 2) if is_short                    else round((current_price - entry_price) * abs_shares, 2)
    exposure_pct = round(exposure / account_size * 100, 1)

    # Stress scenarios
    scenarios = {}
    for label, mult in [("UVXY +20%", 1.20), ("UVXY +40%", 1.40),
                        ("UVXY +60%", 1.60), ("UVXY +100%", 2.00)]:
        stressed_price = current_price * mult
        if is_short:
            loss = round((stressed_price - entry_price) * abs_shares, 2)
        else:
            loss = round((entry_price - stressed_price) * abs_shares, 2)
        loss_pct = round(abs(loss) / account_size * 100, 1)
        scenarios[label] = {
            "uvxy_price": round(stressed_price, 2),
            "pnl":        -abs(loss) if loss < 0 else loss,
            "pnl_pct":    loss_pct,
            "status":     ("🔴 CRITICAL" if loss_pct > 20 else
                          "🟡 WARNING"  if loss_pct > 10 else "🟢 SAFE"),
        }

    # Hard stop price (20% account drawdown threshold)
    max_loss     = account_size * 0.20
    if is_short:
        stop_price = round(entry_price + max_loss / abs_shares, 2)
    else:
        stop_price = round(entry_price - max_loss / abs_shares, 2)

    # Trim recommendation based on regime
    if snap.vix_pct >= 0.90 and snap.aftershock_risk in ("HIGH","MEDIUM"):
        trim_pct   = 50
        trim_reason = "PANIC phase + aftershock risk — cut exposure in half"
    elif snap.vix_pct >= 0.80:
        trim_pct   = 25
        trim_reason = "Elevated regime — reduce by 25%"
    elif snap.collapse_flag:
        trim_pct   = 0
        trim_reason = "Collapse detected — hold, decay working in your favor"
    else:
        trim_pct   = 0
        trim_reason = "Regime favorable — hold position"

    trim_shares   = round(abs_shares * trim_pct / 100)
    trim_value    = round(trim_shares * current_price, 2)
    trim_pnl      = round((entry_price - current_price) * trim_shares, 2) if is_short                     else round((current_price - entry_price) * trim_shares, 2)

    # Overall risk rating
    if exposure_pct > 30 or scenarios["UVXY +40%"]["pnl_pct"] > 15:
        overall_risk = "🔴 HIGH"
    elif exposure_pct > 15 or scenarios["UVXY +40%"]["pnl_pct"] > 8:
        overall_risk = "🟡 MEDIUM"
    else:
        overall_risk = "🟢 LOW"

    return {
        "exposure":        exposure,
        "exposure_pct":    exposure_pct,
        "entry_value":     entry_value,
        "unrealized_pnl":  unrealized,
        "overall_risk":    overall_risk,
        "stop_price":      stop_price,
        "trim_pct":        trim_pct,
        "trim_shares":     trim_shares,
        "trim_value":      trim_value,
        "trim_pnl":        trim_pnl,
        "trim_reason":     trim_reason,
        "scenarios":       scenarios,
        "account_pct":     exposure_pct,
    }
