#!/usr/bin/env python3
"""
apply_regime_reprice.py — installs the regime-aware three-phase reprice with
+8% UVXY spike tripwire into tradier_orchestrator.py.

RUN ON THE SERVER:   python3 apply_regime_reprice.py
Then test:           python3 tradier_orchestrator.py --preview

What it does (all reversible — timestamped backup made first):
  1. backup tradier_orchestrator.py
  2. insert profile table + tripwire constants after STO_FLOOR_DROP
  3. replace the place_with_reprice function (regime param, 3 phases,
     live re-quote, Phase 2/3 +8% UVXY tripwire → abort)
  4. update the 6 call sites to pass regime=phase  (phase is in scope @ line 551)
  5. py_compile to verify it still imports
"""
import re, shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
orig = src

# ── 0. backup ──
bak = TARGET + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"✓ backup: {bak}")

# ── 1. constants (idempotent) ──
if "REPRICE_PROFILES" not in src:
    anchor = "STO_FLOOR_DROP = 0.20  # max drop from mid before giving up"
    if anchor not in src:
        print("❌ anchor constant not found — aborting, file unchanged.")
        sys.exit(1)
    consts = anchor + '''

# ── Regime-aware reprice (Shin spec, Jun 2026) ──────────────────────────────
# Patience keyed to the orchestrator's existing snapshot regime/spike_label.
# PATIENT regimes: time favors the call seller (theta + contango).
# GUARDED regimes: conditions deteriorating — fill faster.
# EXTREME/backwardation never reach here (engine/no-strike logic stands down).
REPRICE_PATIENT = {"CALM", "RISING", "Spike Exhaustion", "Calm", "Spike Fading"}
REPRICE_GUARDED = {"EXPANSION", "Possible Peak", "STRESSED", "EXTREME"}

def _reprice_profile(phase_label: str) -> dict:
    """Map snapshot phase/regime → patience profile. Unknown → guarded (safe)."""
    if phase_label in REPRICE_PATIENT:
        return dict(fast_tries=5, fast_sec=20, slow_sec=180, descent_hhmm=(14, 30), step=0.03)
    # default + explicit guarded set → less patient
    return dict(fast_tries=3, fast_sec=20, slow_sec=90, descent_hhmm=(12, 0), step=0.03)

UVXY_SPIKE_ABORT_PCT = 0.08   # +8% live UVXY move vs placement baseline → abort (Phase 2/3 only)
'''
    src = src.replace(anchor, consts, 1)
    print("✓ constants + profile helper inserted")
else:
    print("• constants already present — skipping")

# ── 2. replace place_with_reprice ──
# Find the function from its def to the next top-level def.
fn_start = src.find("def place_with_reprice(")
if fn_start == -1:
    print("❌ place_with_reprice not found — restoring backup.")
    shutil.copy(bak, TARGET); sys.exit(1)
# next top-level def after it
nxt = re.search(r"\ndef [A-Za-z_]", src[fn_start+10:])
fn_end = fn_start + 10 + nxt.start() if nxt else len(src)

new_fn = '''def place_with_reprice(client, underlying: str, option_symbol: str, side: str,
                       quantity: int, bid: float, ask: float,
                       phase_label: str = "") -> dict:
    """Regime-aware three-phase order management with +8% UVXY spike abort.

    phase_label comes from the orchestrator's snapshot (regime/spike_label).
      P1 fast mid discovery · P2 patient hold (time favors seller) · P3 near-close
      descent to live bid. Tripwire (P2/P3 only): if live UVXY rises
      >= UVXY_SPIKE_ABORT_PCT vs the price at placement, CANCEL and abort —
      never complete a short into a spike.
    Sandbox: cancel+re-place. Live: modify.
    """
    if not _market_open():
        et = _et_now()
        LOG.log(f"   \\U0001f6ab Market closed ({et.strftime('%H:%M ET')}) \\u2014 order blocked")
        return {"status": "market_closed"}

    is_buy = "buy" in side.lower()
    prof   = _reprice_profile(phase_label)

    if not is_buy and bid <= 0.05:
        LOG.log(f"   \\u26a0\\ufe0f Bid ${bid:.2f} too low \\u2014 skipping")
        return {"status": "skipped_low_bid"}

    # UVXY baseline for the spike tripwire (live quote, not the stale snapshot).
    uvxy_base = 0.0
    try:
        qd = client.get_quote("UVXY")
        uvxy_base = float(qd.get("last") or qd.get("close")
                          or ((float(qd.get("bid",0) or 0)+float(qd.get("ask",0) or 0))/2) or 0)
    except Exception:
        uvxy_base = 0.0

    mid   = round((bid + ask) / 2, 2)
    price = mid
    LOG.log(f"   {'BTO' if is_buy else 'STO'} {option_symbol} \\u00d7{quantity} @ ${price:.2f} "
            f"(bid=${bid:.2f} ask=${ask:.2f}) regime='{phase_label}' "
            f"[{prof['fast_tries']} fast, descend@{prof['descent_hhmm'][0]:02d}:{prof['descent_hhmm'][1]:02d}, "
            f"uvxy_base=${uvxy_base:.2f}]")

    try:
        result = client.place_order(underlying, option_symbol, side, quantity, price)
        oid = result.get("order", {}).get("id")
        if not oid:
            LOG.log(f"   \\u274c Place failed: {result}")
            return {"status": "failed"}
        LOG.log(f"   Order {oid} placed")
    except Exception as e:
        LOG.log(f"   \\u274c Place error: {e}")
        return {"status": "failed"}

    for attempt in range(MAX_REPRICE):
        et = _et_now()
        now_hhmm = (et.hour, et.minute)
        if now_hhmm >= prof["descent_hhmm"]:
            phase, interval = 3, 60
        elif attempt < prof["fast_tries"]:
            phase, interval = 1, prof["fast_sec"]
        else:
            phase, interval = 2, prof["slow_sec"]
        if phase == 3 and attempt == 0:
            LOG.log(f"   \\U0001f534 Descent window ('{phase_label}') \\u2014 stepping toward bid")
        time.sleep(interval)

        # ── +8% UVXY spike tripwire (Phase 2/3 only) ──
        if phase in (2, 3) and not is_buy and uvxy_base > 0:
            try:
                qd = client.get_quote("UVXY")
                uvxy_now = float(qd.get("last") or qd.get("close")
                                 or ((float(qd.get("bid",0) or 0)+float(qd.get("ask",0) or 0))/2) or 0)
                if uvxy_now > 0 and (uvxy_now - uvxy_base) / uvxy_base >= UVXY_SPIKE_ABORT_PCT:
                    LOG.log(f"   \\U0001f6d1 UVXY spike +{(uvxy_now-uvxy_base)/uvxy_base*100:.1f}% "
                            f"(${uvxy_base:.2f}\\u2192${uvxy_now:.2f}) \\u2014 ABORTING short")
                    try: client.cancel_order(oid)
                    except Exception: pass
                    return {"status": "aborted_uvxy_spike", "order_id": oid,
                            "uvxy_base": uvxy_base, "uvxy_now": uvxy_now}
            except Exception:
                pass  # quote failure: do not abort on missing data

        try:
            status = client.get_order(oid)
            state = status.get("status", "")
        except Exception:
            state = "unknown"
        LOG.log(f"   [{attempt+1}] phase={phase} state={state} @ ${price:.2f}")

        if not _market_open():
            LOG.log(f"   \\U0001f6ab Market closed \\u2014 stopping reprice loop")
            try: client.cancel_order(oid)
            except Exception: pass
            return {"status": "market_closed", "order_id": oid}

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            LOG.log(f"   \\u2705 Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid, "fill_price": fill_px, "quantity": quantity}

        if state == "partially_filled":
            filled_qty = int(status.get("exec_quantity", 0) or 0)
            fill_px = float(status.get("avg_fill_price", price))
            remaining = quantity - filled_qty
            LOG.log(f"   \\u26a1 Partial: {filled_qty}/{quantity} @ ${fill_px:.2f} \\u2014 {remaining} left")
            if phase == 3:
                try: client.cancel_order(oid)
                except Exception: pass
                if filled_qty > 0:
                    return {"status": "filled", "order_id": oid, "fill_price": fill_px,
                            "quantity": filled_qty, "partial": True}
                return {"status": "canceled", "order_id": oid}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # ── re-fetch live option quote (graceful fallback) ──
        try:
            q = client.get_quote(option_symbol)
            nb = float(q.get("bid", bid) or bid)
            na = float(q.get("ask", ask) or ask)
            if nb > 0 and na > 0:
                bid, ask = nb, na
        except Exception:
            pass
        fresh_mid = round((bid + ask) / 2, 2)

        if phase in (1, 2):
            new_price = fresh_mid
        else:
            new_price = max(round(price - prof["step"], 2), bid)

        if new_price == price:
            LOG.log(f"   Holding ${price:.2f} (mid {fresh_mid}, bid {bid})")
            continue
        LOG.log(f"   Repricing ${price:.2f} \\u2192 ${new_price:.2f} (phase {phase}, mid {fresh_mid}, bid {bid})")

        if client.sandbox:
            try: client.cancel_order(oid)
            except Exception: pass
            try:
                result = client.place_order(underlying, option_symbol, side, quantity, new_price)
                new_oid = result.get("order", {}).get("id")
                if new_oid:
                    oid, price = new_oid, new_price
            except Exception as e:
                LOG.log(f"   \\u26a0\\ufe0f Re-place failed: {e}")
        else:
            try:
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as e:
                LOG.log(f"   \\u26a0\\ufe0f Modify failed: {e}")

    LOG.log(f"   \\u26a0\\ufe0f Max attempts \\u2014 order {oid} working @ ${price:.2f}")
    return {"status": "working", "order_id": oid, "last_price": price}


'''
src = src[:fn_start] + new_fn + src[fn_end:]
print("✓ place_with_reprice replaced")

# ── 3. update the 6 call sites: add phase_label=phase ──
# Each call passes positional bid/ask then closes. We append phase_label=phase.
# Match the closing of each call's argument list that ends with best/old/new bid+ask.
patterns = [
    ('best["bid"], best["ask"])', 'best["bid"], best["ask"], phase_label=phase)'),
    ('best_new["bid"], best_new["ask"])', 'best_new["bid"], best_new["ask"], phase_label=phase)'),
    ('old_bid, old_ask)', 'old_bid, old_ask, phase_label=phase)'),
]
count = 0
for old, new in patterns:
    n = src.count(old)
    if n:
        src = src.replace(old, new)
        count += n
print(f"✓ updated {count} call-site argument lists (expected 6)")

# ── 4. write + verify ──
open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("✓ py_compile OK — file imports cleanly")
except py_compile.PyCompileError as e:
    print(f"❌ py_compile FAILED — restoring backup.\\n{e}")
    shutil.copy(bak, TARGET)
    sys.exit(1)

print("\\n✅ DONE. Now test (no orders placed in preview):")
print("   python3 tradier_orchestrator.py --preview")
print(f"\\nTo revert:  cp {bak} {TARGET}")
