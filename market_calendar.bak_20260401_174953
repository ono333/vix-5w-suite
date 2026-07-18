#!/usr/bin/env python3
"""
Market Calendar - Holidays and Important Dates
Helps the signal generator avoid trading on closed days
and warn about high-impact events.
"""

from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarketEvent:
    """Represents a market event or holiday."""
    date: date
    name: str
    event_type: str  # "holiday", "fomc", "vix_expiry", "jobs", "cpi"
    market_closed: bool = False
    high_impact: bool = False


# 2026 US Market Holidays (NYSE/NASDAQ closed)
US_HOLIDAYS_2026 = [
    (date(2026, 1, 1), "New Year's Day"),
    (date(2026, 1, 19), "MLK Day"),
    (date(2026, 2, 16), "Presidents Day"),
    (date(2026, 4, 3), "Good Friday"),
    (date(2026, 5, 25), "Memorial Day"),
    (date(2026, 6, 19), "Juneteenth"),
    (date(2026, 7, 3), "Independence Day (observed)"),
    (date(2026, 9, 7), "Labor Day"),
    (date(2026, 11, 26), "Thanksgiving"),
    (date(2026, 12, 25), "Christmas"),
]

# 2026 FOMC Meeting Dates (announcement days - high volatility)
FOMC_DATES_2026 = [
    date(2026, 1, 29),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
]

# VIX Monthly Expiration (typically Wednesday, 30 days before SPX expiry)
# These are approximate - VIX settles on Wednesday AM
VIX_EXPIRY_2026 = [
    date(2026, 1, 21),
    date(2026, 2, 18),
    date(2026, 3, 18),
    date(2026, 4, 15),
    date(2026, 5, 20),
    date(2026, 6, 17),
    date(2026, 7, 22),
    date(2026, 8, 19),
    date(2026, 9, 16),
    date(2026, 10, 21),
    date(2026, 11, 18),
    date(2026, 12, 16),
]


def get_market_events(start_date: date, end_date: date) -> List[MarketEvent]:
    """Get all market events in a date range."""
    events = []
    
    # Add holidays
    for holiday_date, name in US_HOLIDAYS_2026:
        if start_date <= holiday_date <= end_date:
            events.append(MarketEvent(
                date=holiday_date,
                name=name,
                event_type="holiday",
                market_closed=True,
                high_impact=False,
            ))
    
    # Add FOMC dates
    for fomc_date in FOMC_DATES_2026:
        if start_date <= fomc_date <= end_date:
            events.append(MarketEvent(
                date=fomc_date,
                name="FOMC Announcement",
                event_type="fomc",
                market_closed=False,
                high_impact=True,
            ))
    
    # Add VIX expiry
    for vix_date in VIX_EXPIRY_2026:
        if start_date <= vix_date <= end_date:
            events.append(MarketEvent(
                date=vix_date,
                name="VIX Monthly Expiration",
                event_type="vix_expiry",
                market_closed=False,
                high_impact=True,
            ))
    
    return sorted(events, key=lambda e: e.date)


def is_market_open(check_date: date) -> bool:
    """Check if market is open on a given date."""
    # Weekend check
    if check_date.weekday() >= 5:
        return False
    
    # Holiday check
    for holiday_date, _ in US_HOLIDAYS_2026:
        if check_date == holiday_date:
            return False
    
    return True


def get_next_trading_day(from_date: date) -> date:
    """Get the next day when market is open."""
    next_day = from_date + timedelta(days=1)
    while not is_market_open(next_day):
        next_day += timedelta(days=1)
    return next_day


def get_upcoming_events(days_ahead: int = 7) -> List[MarketEvent]:
    """Get market events in the next N days."""
    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    return get_market_events(today, end_date)


def format_calendar_warning() -> Optional[str]:
    """Generate a warning string for upcoming high-impact events."""
    events = get_upcoming_events(days_ahead=5)
    
    if not events:
        return None
    
    warnings = []
    today = date.today()
    
    for event in events:
        days_until = (event.date - today).days
        
        if event.market_closed:
            if days_until == 0:
                warnings.append(f"⛔ TODAY: Market CLOSED ({event.name})")
            elif days_until == 1:
                warnings.append(f"⛔ TOMORROW: Market CLOSED ({event.name})")
            else:
                warnings.append(f"📅 {event.date.strftime('%a %b %d')}: Market closed ({event.name})")
        
        elif event.high_impact:
            if days_until == 0:
                warnings.append(f"⚠️ TODAY: {event.name} - Expect high volatility!")
            elif days_until == 1:
                warnings.append(f"⚠️ TOMORROW: {event.name} - Plan accordingly")
            else:
                warnings.append(f"📌 {event.date.strftime('%a %b %d')}: {event.name}")
    
    return "\n".join(warnings) if warnings else None


if __name__ == "__main__":
    print("📅 Market Calendar Check")
    print("=" * 50)
    
    today = date.today()
    print(f"\nToday: {today.strftime('%A, %B %d, %Y')}")
    print(f"Market open today: {is_market_open(today)}")
    
    if not is_market_open(today):
        next_open = get_next_trading_day(today)
        print(f"Next trading day: {next_open.strftime('%A, %B %d')}")
    
    print("\n📌 Upcoming Events (next 14 days):")
    events = get_upcoming_events(days_ahead=14)
    for event in events:
        days = (event.date - today).days
        icon = "⛔" if event.market_closed else "⚠️" if event.high_impact else "📅"
        print(f"  {icon} {event.date.strftime('%a %b %d')} (+{days}d): {event.name}")
    
    print("\n" + "=" * 50)
    warning = format_calendar_warning()
    if warning:
        print("⚡ WARNINGS:")
        print(warning)
    else:
        print("✅ No critical events in the next 5 days")
