#!/usr/bin/env python3
"""
apply_takebid.py — adds the take-the-bid rule to place_with_reprice.

Rule (all tiers, evaluated BEFORE the chase/descent target each cycle):
  - previous-mid trigger (always, after first cycle): bid >= prev_mid - TAKE_BID_TOL -> take bid
  - opening-mid trigger (only after TAKE_BID_OPEN_DELAY_MIN): bid >= open_mid - TAKE_BID_TOL -> take bid
Rationale (Shin): holding an unfilled order costs more than sacrificing <=5c to fill
when a buyer steps up near a reference price. Opening-mid rescues no-fill days and
avoids a worse late-day forced fill. Time-gate prevents cycle-1 collapse on tight spreads.

Depends on reprice v2 + loop v3 being installed.
RUN ON SERVER:  python3 apply_takebid.py
"""
import re, shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_takebid_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# 1. constants
if "TAKE_BID_TOL" not in src:
    anchor = "MICRO_STEEPEN_PCT = 0.03   # UVXY +3% vs open -> one tier more guarded"
    if anchor not in src:
        print("\u274c anchor (MICRO_STEEPEN_PCT) not found \u2014 is reprice v2 installed? File unchanged.")
        shutil.copy(bak, TARGET); sys.exit(1)
    src = src.replace(anchor, anchor +
        "\nTAKE_BID_TOL = 0.05              # take the bid if within this of a reference mid"
        "\nTAKE_BID_OPEN_DELAY_MIN = 15    # opening-mid rescue trigger activates after this many min", 1)
    print("\u2713 take-bid constants inserted")

# 2. init open_mid + prev_mid right after mid0/price are set at placement
init_anchor = "    mid0 = round((bid + ask) / 2, 2)\n    price = mid0\n    placed_at = _et_now()"
if init_anchor not in src:
    print("\u274c placement init block not found. File unchanged."); shutil.copy(bak, TARGET); sys.exit(1)
src = src.replace(init_anchor,
    "    mid0 = round((bid + ask) / 2, 2)\n    price = mid0\n    placed_at = _et_now()\n"
    "    open_mid = mid0          # opening mid reference for the take-bid rescue\n"
    "    prev_mid = mid0          # previous-cycle mid for the take-bid pop trigger", 1)
print("\u2713 open_mid/prev_mid initialised at placement")

# 3. insert the take-bid decision just before the buy/sell target decision.
#    The v2 loop computes live_mid then does `if is_buy: ... else: _reprice_target(...)`.
#    We insert BEFORE that block, for sells only.
target_anchor = "        if is_buy:\n            # buys: simple ratchet toward ask (kept minimal; buys are the protective long)\n            new_price = min(round(price + 0.05, 2), ask)\n        else:\n            new_price, tier = _reprice_target(phase_label, mid0, live_mid, bid, ask,\n                                              elapsed_min, uvxy_open, uvxy_now)"
if target_anchor not in src:
    print("\u274c v2 target-decision block not found. File unchanged."); shutil.copy(bak, TARGET); sys.exit(1)

takebid_block = (
    "        # ── take-the-bid rule (sells): fill into buying rather than hold/chase ──\n"
    "        take_bid = False\n"
    "        if not is_buy and bid > 0:\n"
    "            if bid >= prev_mid - TAKE_BID_TOL:\n"
    "                take_bid = True\n"
    "                LOG.log(f\"   \\U0001f3af Take-bid: bid {bid:.2f} \\u2248 prev mid {prev_mid:.2f} \\u2014 filling\")\n"
    "            elif elapsed_min >= TAKE_BID_OPEN_DELAY_MIN and bid >= open_mid - TAKE_BID_TOL:\n"
    "                take_bid = True\n"
    "                LOG.log(f\"   \\U0001f3af Take-bid (rescue): bid {bid:.2f} \\u2248 open mid {open_mid:.2f} \\u2014 filling\")\n"
    "        prev_mid = live_mid   # update for next cycle (after the comparison)\n"
    "\n"
    "        if take_bid:\n"
    "            new_price = bid\n"
    "        elif is_buy:\n"
    "            new_price = min(round(price + 0.05, 2), ask)\n"
    "        else:\n"
    "            new_price, tier = _reprice_target(phase_label, mid0, live_mid, bid, ask,\n"
    "                                              elapsed_min, uvxy_open, uvxy_now)"
)
src = src.replace(target_anchor, takebid_block, 1)
print("\u2713 take-bid decision inserted before tier logic")

open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --preview")
print(f"   Revert: cp {bak} {TARGET}")
