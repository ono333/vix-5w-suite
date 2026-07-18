#!/usr/bin/env python3
"""
apply_oi_floor.py — adds an open-interest floor to the liquidity check so the
orchestrator won't sell genuinely-dead strikes. Keyed on OPEN INTEREST (persists
overnight), NOT daily volume (which is 0 each morning for liquid strikes too).

RUN ON SERVER:  python3 apply_oi_floor.py
Then test:      python3 tradier_orchestrator.py --preview

Touches (all reversible — timestamped backup first):
  1. tradier_liquidity.py: add MIN_OPEN_INTEREST + optional open_interest arg
  2. tradier_orchestrator.py: carry open_interest/volume into candidate dicts
  3. tradier_orchestrator.py: pass open_interest at the check_liquidity call sites
"""
import re, shutil, datetime, py_compile, os, sys

ORCH = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
LIQ  = os.path.expanduser("~/vix_suite/tradier_liquidity.py")

for f in (ORCH, LIQ):
    bak = f + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(f, bak)
    print(f"✓ backup: {bak}")

# ── 1. tradier_liquidity.py: add OI floor ──
liq = open(LIQ).read()
if "MIN_OPEN_INTEREST" not in liq:
    liq = liq.replace(
        "MAX_SPREAD_ABS: float = 2.00",
        "MAX_SPREAD_ABS: float = 2.00\nMIN_OPEN_INTEREST: int = 10  # reject strikes with fewer contracts outstanding (dead strikes)",
        1,
    )
    # extend the function signature + add the check before the final return True
    liq = liq.replace(
        "def check_liquidity(bid: float, ask: float) -> tuple[bool, str | None]:",
        "def check_liquidity(bid: float, ask: float, open_interest: int | None = None) -> tuple[bool, str | None]:",
        1,
    )
    liq = liq.replace(
        "    if spread_pct > MAX_SPREAD_PCT and spread_abs > MAX_SPREAD_ABS:\n"
        "        return False, f\"spread {spread_pct:.0%} / ${spread_abs:.2f} exceeds both limits\"\n"
        "    return True, None",
        "    if spread_pct > MAX_SPREAD_PCT and spread_abs > MAX_SPREAD_ABS:\n"
        "        return False, f\"spread {spread_pct:.0%} / ${spread_abs:.2f} exceeds both limits\"\n"
        "    if open_interest is not None and open_interest < MIN_OPEN_INTEREST:\n"
        "        return False, f\"open interest {open_interest} < {MIN_OPEN_INTEREST} floor\"\n"
        "    return True, None",
        1,
    )
    open(LIQ, "w").write(liq)
    print("✓ tradier_liquidity.py: MIN_OPEN_INTEREST + OI check added")
else:
    print("• tradier_liquidity.py already patched — skipping")

# ── 2. orchestrator: carry open_interest into candidate dicts ──
orch = open(ORCH).read()
# Both find_long_strike and find_short_strike build dicts ending with the mid line.
# Add OI/volume right after the mid line. There are 2 such dict literals.
old_dict_tail = '''"mid": round((bid+ask)/2, 2),
            })'''
new_dict_tail = '''"mid": round((bid+ask)/2, 2),
                "open_interest": int(c.get("open_interest", 0) or 0),
                "volume": int(c.get("volume", 0) or 0),
            })'''
n_dicts = orch.count(old_dict_tail)
orch = orch.replace(old_dict_tail, new_dict_tail)
print(f"✓ orchestrator: open_interest carried into {n_dicts} candidate dict(s) (expected 2)")

# ── 3. orchestrator: pass open_interest at check_liquidity call sites ──
# Patterns from the grep: check_liquidity(_best["bid"], _best["ask"]) and best_new[...]
repls = [
    ('check_liquidity(_best["bid"], _best["ask"])',
     'check_liquidity(_best["bid"], _best["ask"], _best.get("open_interest"))'),
    ('check_liquidity(\n                                    best_new["bid"], best_new["ask"])',
     'check_liquidity(\n                                    best_new["bid"], best_new["ask"], best_new.get("open_interest"))'),
]
n_calls = 0
for old, new in repls:
    c = orch.count(old)
    if c:
        orch = orch.replace(old, new); n_calls += c
print(f"✓ orchestrator: open_interest passed at {n_calls} call site(s)")

open(ORCH, "w").write(orch)

# ── verify both compile ──
ok = True
for f in (LIQ, ORCH):
    try:
        py_compile.compile(f, doraise=True)
        print(f"✓ py_compile OK: {os.path.basename(f)}")
    except py_compile.PyCompileError as e:
        print(f"❌ py_compile FAILED: {os.path.basename(f)}\n{e}")
        ok = False
if not ok:
    print("❌ One or more files failed to compile — RESTORE from the .bak files above.")
    sys.exit(1)
print("\n✅ DONE. Test: python3 tradier_orchestrator.py --preview")
