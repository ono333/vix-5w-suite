#!/usr/bin/env python3
"""
apply_loop_v3.py — replaces the reprice loop's `for attempt in range(MAX_REPRICE)`
with a time-governed `while _market_open()` loop + a 400-minute elapsed backstop.

WHY: MAX_REPRICE=20 (~27 min) quit before the descent's fill deadline (CALM=360min),
undercutting the guaranteed-fill design. Iteration count is the wrong terminator;
time is the right one. The loop now runs while the market is open and the descent
reaches the bid and fills well before close. The 400-min backstop only fires if the
clock check malfunctions (invisible in normal operation).

RUN ON SERVER:  python3 apply_loop_v3.py
"""
import re, shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_loopv3_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# The v2 loop currently reads:  for attempt in range(MAX_REPRICE):
#                                   elapsed_min = (_et_now() - placed_at).total_seconds() / 60.0
# Replace the `for` line + initialise an attempt counter manually so log [{attempt+1}] still works.
old_loop = "    for attempt in range(MAX_REPRICE):\n        elapsed_min = (_et_now() - placed_at).total_seconds() / 60.0"
new_loop = ("    attempt = 0\n"
            "    while _market_open():\n"
            "        elapsed_min = (_et_now() - placed_at).total_seconds() / 60.0\n"
            "        if elapsed_min > 400:  # backstop: longer than any trading day -> clock bug, bail\n"
            "            LOG.log(f\"   \\u26a0\\ufe0f Elapsed {elapsed_min:.0f}m > 400m backstop \\u2014 stopping\")\n"
            "            try: client.cancel_order(oid)\n"
            "            except Exception: pass\n"
            "            return {\"status\": \"working\", \"order_id\": oid, \"last_price\": price}\n"
            "        attempt += 1")

if old_loop not in src:
    print("\u274c v2 loop header not found (is reprice v2 applied?). File unchanged.")
    print("   Expected exactly:")
    print("     for attempt in range(MAX_REPRICE):")
    print("         elapsed_min = (_et_now() - placed_at).total_seconds() / 60.0")
    shutil.copy(bak, TARGET)
    sys.exit(1)

src = src.replace(old_loop, new_loop, 1)
open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. MAX_REPRICE no longer governs the loop; market-close + 400m backstop do.")
print("   Test: python3 tradier_orchestrator.py --preview")
print(f"   Revert: cp {bak} {TARGET}")
