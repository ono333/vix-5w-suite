#!/usr/bin/env python3
"""
apply_stc_logfill.py — adds the missing log_fill for a successful STC (close of
old long) in the roll path of tradier_orchestrator.py, so the closing trade is
recorded in the exec log (reconciliation accuracy).

Line-anchored (NOT pattern replacement) on the unique STC block. Only ADDS a
logging call inside the existing `if r_buy["status"]=="filled":` block — changes
no existing logic, cannot alter the roll's behavior, only records it.

RUN ON SERVER:  python3 apply_stc_logfill.py
"""
import shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()

bak = TARGET + ".bak_stclog_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# Anchor: the exact STC call block (unique in the file).
anchor = '''                                        r_sell = place_with_reprice(
                                            client, "UVXY", long_pos["symbol"],
                                            "sell_to_close", long_pos["quantity"],
                                            old_bid, old_ask, phase_label=phase)'''

if anchor not in src:
    print("\u274c STC anchor block not found (whitespace mismatch?). File unchanged.")
    print("   Showing what to look for vs what's there would help — paste lines 770-774.")
    sys.exit(1)

# Already patched?
if "\"STC\"" in src and "log_fill(key, long_pos[\"strike\"]" in src:
    print("\u2022 STC log_fill already present \u2014 skipping.")
    sys.exit(0)

addition = anchor + '''
                                        if r_sell.get("status") == "filled":
                                            log_fill(key, long_pos["strike"], str(long_exp), "STC",
                                                     round((old_bid + old_ask) / 2, 2),
                                                     float(r_sell.get("fill_price", 0) or 0),
                                                     long_pos["quantity"],
                                                     str(r_sell.get("order_id", "")))'''

src = src.replace(anchor, addition, 1)
open(TARGET, "w").write(src)

try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK \u2014 file imports cleanly")
except py_compile.PyCompileError as e:
    print(f"\u274c py_compile FAILED \u2014 restoring backup.\n{e}")
    shutil.copy(bak, TARGET)
    sys.exit(1)

print("\n\u2705 DONE. STC close now logged when it fills.")
print("   Verify:  grep -n 'STC' ~/vix_suite/tradier_orchestrator.py")
print("   Test:    python3 tradier_orchestrator.py --preview")
