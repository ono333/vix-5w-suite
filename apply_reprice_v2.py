#!/usr/bin/env python3
"""
apply_reprice_v2.py — replaces place_with_reprice with the regime-gated
chase-up + guaranteed-descent design (verified logic from reprice_v2_logic.py).

Fixes the live bug found Jun 26: orders parked at a static mid and never filled.
New behaviour:
  - 3 tiers (CALM/NORMAL/GUARDED) from snapshot regime
  - micro-regime: UVXY up >=3% vs OPEN bumps one tier more guarded
  - descent floor ratchets mid->bid; HARD deadline forces bid (guarantees fill)
  - chase-up allowed in CALM/NORMAL (capture credit), tapers to 0 by deadline
  - GUARDED: no chase, fast descent
  - +8% UVXY spike vs placement baseline still ABORTS (phase 2/3 equiv)

RUN ON SERVER:  python3 apply_reprice_v2.py
Anchored replacement of the whole place_with_reprice function.
"""
import re, shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_v2reprice_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# Insert tier constants after the existing reprice constants if absent
if "REPRICE_TIERS" not in src:
    anchor_c = "UVXY_SPIKE_ABORT_PCT = 0.08"
    if anchor_c not in src:
        # fall back to inserting after MAX_REPRICE
        anchor_c = "MAX_REPRICE    = 20"
    tiers = anchor_c + '''

# ── Reprice v2: regime-gated chase + guaranteed descent (Jun 26 fix) ──
REPRICE_TIERS = {
    "CALM":    dict(target_min=300, hard_min=360, chase=True,  chase_cap0=1.00),
    "NORMAL":  dict(target_min=120, hard_min=180, chase=True,  chase_cap0=0.50),
    "GUARDED": dict(target_min=45,  hard_min=60,  chase=False, chase_cap0=0.00),
}
CALM_REGIMES    = {"CALM", "RISING", "Spike Exhaustion", "Calm"}
GUARDED_REGIMES = {"EXPANSION", "Possible Peak", "STRESSED", "EXTREME"}
MICRO_STEEPEN_PCT = 0.03   # UVXY +3% vs open -> one tier more guarded

def _base_tier(regime: str) -> str:
    if regime in CALM_REGIMES: return "CALM"
    if regime in GUARDED_REGIMES: return "GUARDED"
    return "NORMAL"

def _apply_micro(tier: str, uvxy_open: float, uvxy_now: float) -> str:
    if uvxy_open and uvxy_now and (uvxy_now - uvxy_open)/uvxy_open >= MICRO_STEEPEN_PCT:
        return {"CALM":"NORMAL","NORMAL":"GUARDED","GUARDED":"GUARDED"}[tier]
    return tier

def _descent_floor(mid0, bid, elapsed_min, target_min, hard_min):
    if elapsed_min >= hard_min: return bid
    frac = min(max(elapsed_min/target_min,0.0),1.0) if target_min>0 else 1.0
    return round(mid0 - (mid0-bid)*frac, 2)

def _chase_cap_now(cap0, elapsed_min, hard_min):
    if hard_min<=0: return 0.0
    return cap0 * max(0.0, 1.0 - elapsed_min/hard_min)

def _reprice_target(regime, mid0, live_mid, live_bid, live_ask, elapsed_min, uo, un):
    tier = _apply_micro(_base_tier(regime), uo, un)
    t = REPRICE_TIERS[tier]
    floor = _descent_floor(mid0, live_bid, elapsed_min, t["target_min"], t["hard_min"])
    price = max(floor, live_bid)
    if t["chase"] and live_mid > price:
        cap = _chase_cap_now(t["chase_cap0"], elapsed_min, t["hard_min"])
        ceiling = price + (live_ask - price)*cap
        price = min(max(price, live_mid), ceiling)
    if elapsed_min >= t["hard_min"]:
        price = live_bid
    return round(max(price, 0.01), 2), tier
'''
    src = src.replace(anchor_c, tiers, 1)
    print("\u2713 tier constants + helpers inserted")
else:
    print("\u2022 tier constants already present")

# Replace the place_with_reprice function body
fn_start = src.find("def place_with_reprice(")
if fn_start == -1:
    print("\u274c place_with_reprice not found"); shutil.copy(bak, TARGET); sys.exit(1)
nxt = re.search(r"\ndef [A-Za-z_]", src[fn_start+10:])
fn_end = fn_start + 10 + nxt.start() if nxt else len(src)

new_fn = '''def place_with_reprice(client, underlying: str, option_symbol: str, side: str,
                       quantity: int, bid: float, ask: float,
                       phase_label: str = "") -> dict:
    """Regime-gated chase + guaranteed descent. phase_label = snapshot regime.
    Fixes the Jun 26 park-at-mid-never-fill bug. See REPRICE_TIERS."""
    if not _market_open():
        LOG.log(f"   \\U0001f6ab Market closed \\u2014 order blocked")
        return {"status": "market_closed"}

    is_buy = "buy" in side.lower()

    if not is_buy and bid <= 0.05:
        LOG.log(f"   \\u26a0\\ufe0f Bid ${bid:.2f} too low \\u2014 skipping")
        return {"status": "skipped_low_bid"}

    # UVXY open + baseline for micro-regime and spike tripwire (live quote).
    uvxy_open = uvxy_base = 0.0
    try:
        uq = client.get_quote("UVXY")
        uvxy_open = float(uq.get("open") or uq.get("prevclose") or 0)
        uvxy_base = float(uq.get("last") or uq.get("close") or 0)
    except Exception:
        pass

    mid0 = round((bid + ask) / 2, 2)
    price = mid0
    placed_at = _et_now()
    LOG.log(f"   {'BTO' if is_buy else 'STO'} {option_symbol} \\u00d7{quantity} @ ${price:.2f} "
            f"(bid=${bid:.2f} ask=${ask:.2f}) regime='{phase_label}' "
            f"tier={_apply_micro(_base_tier(phase_label), uvxy_open, uvxy_base)} "
            f"uvxy_open=${uvxy_open:.2f}")

    try:
        result = client.place_order(underlying, option_symbol, side, quantity, price)
        oid = result.get("order", {}).get("id")
        if not oid:
            LOG.log(f"   \\u274c Place failed: {result}"); return {"status": "failed"}
        LOG.log(f"   Order {oid} placed")
    except Exception as e:
        LOG.log(f"   \\u274c Place error: {e}"); return {"status": "failed"}

    for attempt in range(MAX_REPRICE):
        elapsed_min = (_et_now() - placed_at).total_seconds() / 60.0
        # cadence: faster early, also bounded so we re-check responsively
        time.sleep(30 if elapsed_min < 5 else 90)

        # ── spike tripwire: UVXY +8% vs placement baseline -> abort short ──
        if not is_buy and uvxy_base > 0:
            try:
                uq = client.get_quote("UVXY")
                un = float(uq.get("last") or uq.get("close") or 0)
                if un > 0 and (un - uvxy_base)/uvxy_base >= UVXY_SPIKE_ABORT_PCT:
                    LOG.log(f"   \\U0001f6d1 UVXY spike +{(un-uvxy_base)/uvxy_base*100:.1f}% \\u2014 ABORT")
                    try: client.cancel_order(oid)
                    except Exception: pass
                    return {"status": "aborted_uvxy_spike", "order_id": oid}
            except Exception:
                pass

        try:
            status = client.get_order(oid); state = status.get("status", "")
        except Exception:
            state = "unknown"
        LOG.log(f"   [{attempt+1}] t={elapsed_min:.0f}m state={state} @ ${price:.2f}")

        if not _market_open():
            LOG.log(f"   \\U0001f6ab Market closed \\u2014 stopping")
            try: client.cancel_order(oid)
            except Exception: pass
            return {"status": "market_closed", "order_id": oid}

        if state == "filled":
            fp = float(status.get("avg_fill_price", price))
            LOG.log(f"   \\u2705 Filled @ ${fp:.2f}")
            return {"status": "filled", "order_id": oid, "fill_price": fp, "quantity": quantity}
        if state == "partially_filled":
            fq = int(status.get("exec_quantity", 0) or 0)
            fp = float(status.get("avg_fill_price", price))
            if elapsed_min >= 60:
                try: client.cancel_order(oid)
                except Exception: pass
                if fq > 0:
                    return {"status": "filled", "order_id": oid, "fill_price": fp,
                            "quantity": fq, "partial": True}
                return {"status": "canceled", "order_id": oid}
        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # ── re-quote the live option ──
        try:
            q = client.get_quote(option_symbol)
            nb = float(q.get("bid", bid) or bid); na = float(q.get("ask", ask) or ask)
            if nb > 0 and na > 0: bid, ask = nb, na
        except Exception:
            pass
        live_mid = round((bid + ask) / 2, 2)

        # ── re-fetch UVXY for micro-regime ──
        try:
            uq = client.get_quote("UVXY")
            uvxy_now = float(uq.get("last") or uq.get("close") or uvxy_base)
        except Exception:
            uvxy_now = uvxy_base

        if is_buy:
            # buys: simple ratchet toward ask (kept minimal; buys are the protective long)
            new_price = min(round(price + 0.05, 2), ask)
        else:
            new_price, tier = _reprice_target(phase_label, mid0, live_mid, bid, ask,
                                              elapsed_min, uvxy_open, uvxy_now)

        if new_price == price:
            LOG.log(f"   Holding ${price:.2f} (mid {live_mid}, bid {bid})")
            continue
        LOG.log(f"   Repricing ${price:.2f} \\u2192 ${new_price:.2f} (mid {live_mid}, bid {bid})")

        if client.sandbox:
            try: client.cancel_order(oid)
            except Exception: pass
            try:
                result = client.place_order(underlying, option_symbol, side, quantity, new_price)
                noid = result.get("order", {}).get("id")
                if noid: oid, price = noid, new_price
            except Exception as e:
                LOG.log(f"   \\u26a0\\ufe0f Re-place failed: {e}")
        else:
            try:
                client.modify_order(oid, new_price); price = new_price
            except Exception as e:
                LOG.log(f"   \\u26a0\\ufe0f Modify failed: {e}")

    LOG.log(f"   \\u26a0\\ufe0f Max attempts \\u2014 order {oid} working @ ${price:.2f}")
    return {"status": "working", "order_id": oid, "last_price": price}


'''
src = src[:fn_start] + new_fn + src[fn_end:]
open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --preview")
print(f"Revert: cp {bak} {TARGET}")
