#!/usr/bin/env python3
"""
apply_holiday_shift.py — shift a short entry to the trading day BEFORE a holiday.

Rule (Shin, Jul 2026): if the scheduled Mon/Fri entry day is a market holiday,
enter on the last trading day before it instead of skipping the cycle.
Concrete case: Fri Jul 3 (holiday) -> enter Thu Jul 2.

SAFETY: does NOT risk double entry — the orchestrator already syncs live Tradier
state and only enters a variant when short_pos is None. The shift merely changes
WHICH day is treated as an entry day; the existing 'short already exists?' check
still prevents doubling. This patch only touches the is_short_day computation.

Line-anchored. Depends on market_calendar.is_market_open (already used in main()).
RUN ON SERVER: python3 apply_holiday_shift.py
"""
import shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_holshift_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# ── 1. add the helper function just before `def run(` ──
if "_is_short_entry_day" not in src:
    anchor = "def run(sandbox: bool = True, preview: bool = False, check_only: bool = False):"
    if anchor not in src:
        print("\u274c run() signature not found. File unchanged."); shutil.copy(bak, TARGET); sys.exit(1)
    helper = '''def _is_short_entry_day(today) -> bool:
    """True if today should be treated as a short-entry day.

    Normal: today is Mon or Fri and the market is open.
    Holiday shift: if a scheduled Mon/Fri entry day is CLOSED (holiday) and today
    is the last open trading day before it, enter today instead of skipping.
    Example: Fri Jul 3 is a holiday -> Thu Jul 2 becomes the entry day.

    Safe against double entry: the caller separately syncs live Tradier state and
    only enters a variant whose short is None, so shifting the day cannot double up.
    """
    from datetime import timedelta
    try:
        from market_calendar import is_market_open
    except Exception:
        # If the calendar is unavailable, fall back to the plain Mon/Fri rule.
        return today.weekday() in (0, 4)

    # Must be an open trading day to place anything at all.
    if not is_market_open(today):
        return False

    # Normal scheduled entry day.
    if today.weekday() in (0, 4):
        return True

    # Holiday-shift: walk forward over consecutive CLOSED days. If any of those
    # closed days is a scheduled entry day (Mon/Fri), then today (the last open
    # day before them) inherits that entry. Stop at the next open day.
    nxt = today + timedelta(days=1)
    guard = 0
    while not is_market_open(nxt) and guard < 10:
        if nxt.weekday() in (0, 4):
            return True
        nxt += timedelta(days=1)
        guard += 1
    return False


'''
    src = src.replace(anchor, helper + anchor, 1)
    print("\u2713 _is_short_entry_day helper inserted before run()")
else:
    print("\u2022 helper already present")

# ── 2. replace the naive is_short_day line ──
old_line = "    is_short_day  = weekday in (0, 4)  # Monday or Friday"
new_line = "    is_short_day  = _is_short_entry_day(today)  # Mon/Fri, or the trading day before a holiday Mon/Fri"
if old_line not in src:
    print("\u274c is_short_day line not found as expected. File unchanged."); shutil.copy(bak, TARGET); sys.exit(1)
src = src.replace(old_line, new_line, 1)
print("\u2713 is_short_day now uses holiday-shift helper")

open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --preview")
print(f"   Revert: cp {bak} {TARGET}")
