#!/usr/bin/env python3
"""
apply_marketdata_integration.py — wire market_data.py freshness guard + SQLite
logging into the orchestrator's quote sites.

Sites (from the real code):
  top of place_with_reprice : validate the INCOMING bid/ask before first placement
                              (the phantom-$0.53 guard) + log
  494 option re-quote       : validate; accept new bid/ask only if OK; log; hold-last-good
  413 UVXY placement        : log (context=placement_uvxy)
  453 UVXY tripwire         : log (context=tripwire)
  503 UVXY micro-regime     : log (context=micro)
  852 roll STC quote        : log (context=roll)

Venue-aware: in sandbox the timestamp check is skipped (bid_date frozen), so paper
runs won't be blocked on staleness — guard mainly catches $0-bid + logs wide spread.
All logging fail-safe (market_data wraps its own DB errors).

Line-anchored. RUN ON SERVER: python3 apply_marketdata_integration.py
"""
import shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_mdintegration_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

edits = 0

# ── 1. import market_data (after the tradier_exec_log import) ──
if "import market_data" not in src:
    anchor = "from tradier_exec_log import log_fill"
    if anchor not in src:
        # fallback: after tradier_liquidity import
        anchor = "from tradier_liquidity import check_liquidity, log_skip"
    if anchor not in src:
        print("\u274c import anchor not found. File unchanged."); shutil.copy(bak, TARGET); sys.exit(1)
    src = src.replace(anchor, anchor + "\nimport market_data", 1)
    print("\u2713 import market_data added"); edits += 1

# ── 2. TOP-OF-FUNCTION incoming bid/ask guard (phantom-$0.53) ──
# Anchor: the existing low-bid skip block at the top of place_with_reprice.
old_top = '''    is_buy = "buy" in side.lower()

    if not is_buy and bid <= 0.05:
        LOG.log(f"   \\u26a0\\ufe0f Bid ${bid:.2f} too low \\u2014 skipping")
        return {"status": "skipped_low_bid"}'''
new_top = '''    is_buy = "buy" in side.lower()

    if not is_buy and bid <= 0.05:
        LOG.log(f"   \\u26a0\\ufe0f Bid ${bid:.2f} too low \\u2014 skipping")
        return {"status": "skipped_low_bid"}

    # ── freshness guard on the INCOMING quote (phantom-price protection) ──
    _incoming_q = {"bid": bid, "ask": ask}
    _ok, _reason = market_data.check_and_log(_incoming_q, option_symbol, "placement", client.sandbox)
    if not _ok:
        LOG.log(f"   \\U0001f6d1 Placement quote rejected ({_reason}) \\u2014 not placing")
        return {"status": "rejected_quote", "reason": _reason}'''
if old_top in src:
    src = src.replace(old_top, new_top, 1)
    print("\u2713 incoming-quote guard added at top of place_with_reprice"); edits += 1
else:
    print("\u26a0\ufe0f  top-of-function anchor not found (skipping site)")

# ── 3. SITE 494 — the critical option re-quote ──
old_494 = '''        # \u2500\u2500 re-quote the live option \u2500\u2500
        try:
            q = client.get_quote(option_symbol)
            nb = float(q.get("bid", bid) or bid); na = float(q.get("ask", ask) or ask)
            if nb > 0 and na > 0: bid, ask = nb, na
        except Exception:
            pass
        live_mid = round((bid + ask) / 2, 2)'''
new_494 = '''        # \u2500\u2500 re-quote the live option (freshness-guarded) \u2500\u2500
        try:
            q = client.get_quote(option_symbol)
            _ok, _reason = market_data.check_and_log(q, option_symbol, "reprice", client.sandbox)
            if _ok:
                nb = float(q.get("bid", bid) or bid); na = float(q.get("ask", ask) or ask)
                if nb > 0 and na > 0: bid, ask = nb, na
            else:
                LOG.log(f"   \u26a0\ufe0f Re-quote rejected ({_reason}) \u2014 holding last good ${bid:.2f}/{ask:.2f}")
        except Exception:
            pass
        live_mid = round((bid + ask) / 2, 2)'''
if old_494 in src:
    src = src.replace(old_494, new_494, 1)
    print("\u2713 site 494 option re-quote freshness-guarded"); edits += 1
else:
    print("\u274c site 494 anchor not found \u2014 restoring."); shutil.copy(bak, TARGET); sys.exit(1)

# ── 4. SITE 413 — UVXY at placement (log only) ──
old_413 = '''    try:
        uq = client.get_quote("UVXY")
        uvxy_open = float(uq.get("open") or uq.get("prevclose") or 0)
        uvxy_base = float(uq.get("last") or uq.get("close") or 0)
    except Exception:
        pass'''
new_413 = '''    try:
        uq = client.get_quote("UVXY")
        market_data.log_quote(uq, "UVXY", "placement_uvxy", "info", client.sandbox)
        uvxy_open = float(uq.get("open") or uq.get("prevclose") or 0)
        uvxy_base = float(uq.get("last") or uq.get("close") or 0)
    except Exception:
        pass'''
if old_413 in src:
    src = src.replace(old_413, new_413, 1)
    print("\u2713 site 413 UVXY placement logged"); edits += 1
else:
    print("\u26a0\ufe0f  site 413 anchor not found (skipping)")

# ── 5. SITE 453 — UVXY tripwire (log only) ──
old_453 = '''            try:
                uq = client.get_quote("UVXY")
                un = float(uq.get("last") or uq.get("close") or 0)
                if un > 0 and (un - uvxy_base)/uvxy_base >= UVXY_SPIKE_ABORT_PCT:'''
new_453 = '''            try:
                uq = client.get_quote("UVXY")
                market_data.log_quote(uq, "UVXY", "tripwire", "info", client.sandbox)
                un = float(uq.get("last") or uq.get("close") or 0)
                if un > 0 and (un - uvxy_base)/uvxy_base >= UVXY_SPIKE_ABORT_PCT:'''
if old_453 in src:
    src = src.replace(old_453, new_453, 1)
    print("\u2713 site 453 UVXY tripwire logged"); edits += 1
else:
    print("\u26a0\ufe0f  site 453 anchor not found (skipping)")

# ── 6. SITE 503 — UVXY micro-regime (log only) ──
old_503 = '''        # \u2500\u2500 re-fetch UVXY for micro-regime \u2500\u2500
        try:
            uq = client.get_quote("UVXY")
            uvxy_now = float(uq.get("last") or uq.get("close") or uvxy_base)
        except Exception:
            uvxy_now = uvxy_base'''
new_503 = '''        # \u2500\u2500 re-fetch UVXY for micro-regime \u2500\u2500
        try:
            uq = client.get_quote("UVXY")
            market_data.log_quote(uq, "UVXY", "micro", "info", client.sandbox)
            uvxy_now = float(uq.get("last") or uq.get("close") or uvxy_base)
        except Exception:
            uvxy_now = uvxy_base'''
if old_503 in src:
    src = src.replace(old_503, new_503, 1)
    print("\u2713 site 503 UVXY micro-regime logged"); edits += 1
else:
    print("\u26a0\ufe0f  site 503 anchor not found (skipping)")

# ── 7. SITE 852 — roll STC option quote (validate + log) ──
old_852 = '''                                        old_quote = client.get_quote(long_pos["symbol"])
                                        old_bid   = float(old_quote.get("bid", 0.10))
                                        old_ask   = float(old_quote.get("ask", 0.20))'''
new_852 = '''                                        old_quote = client.get_quote(long_pos["symbol"])
                                        market_data.check_and_log(old_quote, long_pos["symbol"], "roll", client.sandbox)
                                        old_bid   = float(old_quote.get("bid", 0.10))
                                        old_ask   = float(old_quote.get("ask", 0.20))'''
if old_852 in src:
    src = src.replace(old_852, new_852, 1)
    print("\u2713 site 852 roll quote logged"); edits += 1
else:
    print("\u26a0\ufe0f  site 852 anchor not found (skipping)")

open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print(f"\u2713 py_compile OK  ({edits} edits applied)")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --check")
print(f"   Revert: cp {bak} {TARGET}")
