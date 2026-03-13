"""
Event Calendar — Macro event tracker with VIX impact ratings.
Auto-calculates recurring events + manual entry for surprises.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
import json
import calendar

STORAGE_DIR   = Path.home() / ".vix_suite"
CALENDAR_FILE = STORAGE_DIR / "event_calendar.json"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MarketEvent:
    event_id:     str
    date:         str          # YYYY-MM-DD
    time_et:      str          # e.g. "08:30", "14:00", "TBD"
    event_type:   str          # FOMC / CPI / NFP / OPEX / EARNINGS / CUSTOM
    title:        str
    description:  str          = ""
    vix_impact:   str          = "MEDIUM"   # LOW / MEDIUM / HIGH / EXTREME
    direction:    str          = "SPIKE"    # SPIKE / SUPPRESS / NEUTRAL / UNKNOWN
    is_manual:    bool         = False
    source:       str          = "auto"
    created_at:   str          = field(
                                 default_factory=lambda: datetime.now().isoformat())

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @property
    def days_away(self) -> int:
        try:
            return (date.fromisoformat(self.date) - date.today()).days
        except Exception:
            return 999

    @property
    def is_today(self) -> bool:
        return self.days_away == 0

    @property
    def is_past(self) -> bool:
        return self.days_away < 0


# ── Auto-calculation helpers ──────────────────────────────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth weekday (0=Mon) of given month."""
    first = date(year, month, 1)
    days_ahead = weekday - first.weekday()
    if days_ahead < 0:
        days_ahead += 7
    first_occurrence = first + timedelta(days=days_ahead)
    return first_occurrence + timedelta(weeks=n - 1)


def _third_friday(year: int, month: int) -> date:
    """Options expiration — third Friday of month."""
    return _nth_weekday(year, month, 4, 3)  # 4=Friday


def _fomc_dates_2026() -> list[date]:
    """
    Known FOMC meeting dates for 2026.
    Decision announced ~14:00 ET on last day.
    """
    return [
        date(2026, 1, 29),
        date(2026, 3, 19),
        date(2026, 5, 7),
        date(2026, 6, 18),
        date(2026, 7, 30),
        date(2026, 9, 17),
        date(2026, 10, 29),
        date(2026, 12, 10),
    ]


def _cpi_dates_2026() -> list[date]:
    """
    CPI typically released 2nd or 3rd Tuesday/Wednesday of month at 08:30 ET.
    Approximate schedule — update manually if BLS changes.
    """
    return [
        date(2026, 1, 15), date(2026, 2, 12), date(2026, 3, 12),
        date(2026, 4, 10), date(2026, 5, 13), date(2026, 6, 11),
        date(2026, 7, 14), date(2026, 8, 13), date(2026, 9, 11),
        date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 11),
    ]


def _nfp_dates_2026() -> list[date]:
    """
    NFP (Jobs Report) — first Friday of month at 08:30 ET.
    """
    results = []
    for month in range(1, 13):
        results.append(_nth_weekday(2026, month, 4, 1))  # first Friday
    return results


def generate_auto_events(
    months_ahead: int = 3,
) -> list[MarketEvent]:
    """Generate recurring macro events for the next N months."""
    import uuid
    today = date.today()
    cutoff = today + timedelta(days=months_ahead * 31)
    events = []

    # FOMC
    for d in _fomc_dates_2026():
        if today - timedelta(days=1) <= d <= cutoff:
            events.append(MarketEvent(
                event_id    = f"fomc_{d.isoformat()}",
                date        = d.isoformat(),
                time_et     = "14:00",
                event_type  = "FOMC",
                title       = "FOMC Rate Decision",
                description = "Federal Reserve interest rate decision + press conference. "
                              "Surprise cuts/hikes cause extreme VIX moves.",
                vix_impact  = "EXTREME",
                direction   = "UNKNOWN",
                source      = "auto",
            ))

    # CPI
    for d in _cpi_dates_2026():
        if today - timedelta(days=1) <= d <= cutoff:
            events.append(MarketEvent(
                event_id    = f"cpi_{d.isoformat()}",
                date        = d.isoformat(),
                time_et     = "08:30",
                event_type  = "CPI",
                title       = "CPI Inflation Report",
                description = "Consumer Price Index. Hot CPI = rate fear = VIX spike. "
                              "Cool CPI = rally = VIX suppression.",
                vix_impact  = "HIGH",
                direction   = "UNKNOWN",
                source      = "auto",
            ))

    # NFP
    for d in _nfp_dates_2026():
        if today - timedelta(days=1) <= d <= cutoff:
            events.append(MarketEvent(
                event_id    = f"nfp_{d.isoformat()}",
                date        = d.isoformat(),
                time_et     = "08:30",
                event_type  = "NFP",
                title       = "Non-Farm Payrolls",
                description = "Monthly jobs report. Weak jobs = recession fear = VIX spike. "
                              "Strong jobs = Fed hawkish risk.",
                vix_impact  = "HIGH",
                direction   = "UNKNOWN",
                source      = "auto",
            ))

    # Monthly OpEx (3rd Friday)
    for month in range(today.month, today.month + months_ahead + 1):
        yr = 2026 + (month - 1) // 12
        mo = ((month - 1) % 12) + 1
        d  = _third_friday(yr, mo)
        if today - timedelta(days=1) <= d <= cutoff:
            is_quarterly = mo in (3, 6, 9, 12)
            events.append(MarketEvent(
                event_id    = f"opex_{d.isoformat()}",
                date        = d.isoformat(),
                time_et     = "16:00",
                event_type  = "OPEX",
                title       = f"{'Quarterly' if is_quarterly else 'Monthly'} OpEx",
                description = ("Quarterly expiration (Triple Witching) — "
                               "large gamma unwind, elevated volatility." if is_quarterly
                               else "Monthly options expiration — moderate vol impact."),
                vix_impact  = "HIGH" if is_quarterly else "MEDIUM",
                direction   = "SPIKE",
                source      = "auto",
            ))

    return sorted(events, key=lambda e: e.date)


# ── Store ─────────────────────────────────────────────────────────────────────

class EventCalendarStore:
    def __init__(self):
        self.manual_events: list[MarketEvent] = []
        self._load()

    def _load(self):
        if CALENDAR_FILE.exists():
            try:
                raw = json.loads(CALENDAR_FILE.read_text())
                self.manual_events = [MarketEvent.from_dict(r) for r in raw]
            except Exception:
                self.manual_events = []

    def save(self):
        CALENDAR_FILE.write_text(
            json.dumps([e.to_dict() for e in self.manual_events], indent=2))

    def add(self, event: MarketEvent):
        self.manual_events.append(event)
        self.save()

    def remove(self, event_id: str):
        self.manual_events = [e for e in self.manual_events
                              if e.event_id != event_id]
        self.save()

    def all_events(self, include_past: bool = False) -> list[MarketEvent]:
        auto   = generate_auto_events()
        manual = self.manual_events
        # Merge — manual overrides auto if same date+type
        auto_ids = {e.event_id for e in auto}
        combined = auto + [e for e in manual if e.event_id not in auto_ids]
        combined = sorted(combined, key=lambda e: e.date)
        if not include_past:
            combined = [e for e in combined if not e.is_past]
        return combined

    def upcoming(self, days: int = 21) -> list[MarketEvent]:
        return [e for e in self.all_events()
                if 0 <= e.days_away <= days]


def get_event_store() -> EventCalendarStore:
    return EventCalendarStore()


# ── VIX impact assessment ─────────────────────────────────────────────────────

def event_risk_summary(events: list[MarketEvent]) -> dict:
    """Summarize event risk for next 7 days."""
    next7 = [e for e in events if 0 <= e.days_away <= 7]
    if not next7:
        return {"level": "LOW", "color": "#2e7d32",
                "label": "No major events in 7 days — calm window"}

    has_extreme = any(e.vix_impact == "EXTREME" for e in next7)
    has_high    = any(e.vix_impact == "HIGH"    for e in next7)
    today_events = [e for e in next7 if e.is_today]

    if has_extreme or len(next7) >= 3:
        level, color = "EXTREME", "#c62828"
    elif has_high or len(next7) >= 2:
        level, color = "HIGH",    "#e65100"
    else:
        level, color = "MEDIUM",  "#f57f17"

    next_event = next7[0] if next7 else None
    label = (f"{len(next7)} events in 7 days — "
             f"next: {next_event.title} in {next_event.days_away}d"
             if next_event else "")

    return {
        "level":        level,
        "color":        color,
        "label":        label,
        "today_events": today_events,
        "count":        len(next7),
    }
