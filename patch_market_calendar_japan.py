"""
Patch: market_calendar.py — add FOMC, VIX expiry, and Memorial Day
for Apr 30 – Jun 10 2026 Japan trip period.

Adds:
  - FOMC May 5-6 (decision day May 6, 2pm ET) — HIGH IMPACT
  - VIX May expiry May 19 (Tuesday, shifted from Juneteenth Friday)
  - Memorial Day May 25 — market closed
  - FOMC Jun 16-17 (just after return, good to have)

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_market_calendar_japan.py
  python market_calendar.py   # verify output
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "market_calendar.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Check what's already in the file ─────────────────────
already_has_memorial = "Memorial Day" in src
already_has_fomc_may = "May 5" in src or "May 6" in src or "(2026, 5, 6)" in src
already_has_vix_may  = "(2026, 5, 19)" in src

print(f"Memorial Day already present: {already_has_memorial}")
print(f"FOMC May 5-6 already present: {already_has_fomc_may}")
print(f"VIX May 19 expiry already present: {already_has_vix_may}")

# ── Fix 1: Add Memorial Day to US_HOLIDAYS_2026 ──────────
# Memorial Day is a market close — find the holidays list and insert
OLD_GOOD_FRIDAY = '    (date(2026, 4, 3), "Good Friday"),'
NEW_GOOD_FRIDAY = '''\
    (date(2026, 4, 3), "Good Friday"),
    (date(2026, 5, 25), "Memorial Day"),'''

if not already_has_memorial and OLD_GOOD_FRIDAY in src:
    src = src.replace(OLD_GOOD_FRIDAY, NEW_GOOD_FRIDAY)
    print("✅ Fix 1: Memorial Day May 25 added to holidays")
elif already_has_memorial:
    print("ℹ️  Fix 1: Memorial Day already in file")
else:
    print("⚠️  Fix 1: Could not find Good Friday anchor — check holidays list format")

# ── Fix 2: Add FOMC and VIX expiry as high-impact events ─
# Find where FOMC events are added (or add after holidays block)
# Look for existing FOMC entries to find the pattern
fomc_pattern_found = "fomc" in src.lower() or "FOMC" in src

# Find a good anchor — look for where events are built
# Try to find the events list construction
FOMC_BLOCK = '''
# ── Japan trip high-impact dates (Apr 30 – Jun 10 2026) ──
JAPAN_TRIP_EVENTS = [
    # FOMC May 5-6: decision announced May 6 at 2pm ET (3am Japan May 7)
    # Week 1 short expires May 7 — highest risk overlap
    MarketEvent(
        date=date(2026, 5, 6),
        name="FOMC Decision — May (⚠️ Japan: 3am May 7 JST)",
        event_type="fomc",
        market_closed=False,
        high_impact=True,
    ),
    MarketEvent(
        date=date(2026, 5, 5),
        name="FOMC Day 1 — May",
        event_type="fomc",
        market_closed=False,
        high_impact=True,
    ),
    # VIX May expiry — moved to Tuesday May 19 due to Juneteenth
    MarketEvent(
        date=date(2026, 5, 19),
        name="VIX May Expiry (shifted — Juneteenth)",
        event_type="vix_expiry",
        market_closed=False,
        high_impact=True,
    ),
    # FOMC June 16-17 (just after return Jun 10)
    MarketEvent(
        date=date(2026, 6, 16),
        name="FOMC Day 1 — June",
        event_type="fomc",
        market_closed=False,
        high_impact=True,
    ),
    MarketEvent(
        date=date(2026, 6, 17),
        name="FOMC Decision — June",
        event_type="fomc",
        market_closed=False,
        high_impact=True,
    ),
]
'''

# Find where to inject — look for a function definition after the constants
# or existing event list. Try several anchors.
anchors = [
    "def get_market_events(",
    "def format_calendar_warning(",
    "def is_market_open(",
]

injected = False
for anchor in anchors:
    if anchor in src and "JAPAN_TRIP_EVENTS" not in src:
        src = src.replace(anchor, FOMC_BLOCK + "\n\n" + anchor, 1)
        print(f"✅ Fix 2: FOMC/VIX events block injected before '{anchor}'")
        injected = True
        break

if not injected and "JAPAN_TRIP_EVENTS" in src:
    print("ℹ️  Fix 2: JAPAN_TRIP_EVENTS already in file")
elif not injected:
    print("⚠️  Fix 2: Could not find anchor — appending to end of file")
    src = src + "\n" + FOMC_BLOCK

# ── Fix 3: Wire JAPAN_TRIP_EVENTS into get_market_events() ──
# Find where events are assembled and add JAPAN_TRIP_EVENTS
OLD_RETURN = "    return events"
NEW_RETURN = """\
    # Add Japan trip high-impact events if in range
    if "JAPAN_TRIP_EVENTS" in dir():
        for evt in JAPAN_TRIP_EVENTS:
            if start_date <= evt.date <= end_date:
                events.append(evt)
    return events"""

# Only patch if JAPAN_TRIP_EVENTS was added and return not already patched
if "JAPAN_TRIP_EVENTS" in src and "Add Japan trip" not in src:
    # Replace only the last occurrence of return events (in get_market_events)
    last_idx = src.rfind(OLD_RETURN)
    if last_idx != -1:
        src = src[:last_idx] + NEW_RETURN + src[last_idx + len(OLD_RETURN):]
        print("✅ Fix 3: JAPAN_TRIP_EVENTS wired into get_market_events()")
    else:
        print("⚠️  Fix 3: Could not find 'return events' — check manually")
else:
    print("ℹ️  Fix 3: Already wired or not needed")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("\nVerify:")
print("  python market_calendar.py")
