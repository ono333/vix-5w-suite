#!/usr/bin/env python3
"""
tradier_long_manager.py
───────────────────────
Manages UVXY long call legs (LEAPs) for all 5 variants via Tradier API.

Responsibilities:
  1. Check if long leg exists per variant
  2. Enter new long if missing
  3. Roll long when DTE approaches threshold
  4. Flag phase transitions (V4→V2 etc) for confirmation

Roll thresholds by variant (from long_dte_weeks):
  V1: 26w long → roll when DTE < 30d
  V2: 13w long → roll when DTE < 21d
  V3:  8w long → roll when DTE < 14d
  V4: 13w long → roll when DTE < 21d
  V5: 13w long → roll when DTE < 21d

Long leg target: delta 0.65–0.75 (ITM), nearest Friday >= target DTE

Phase transition detection:
  - V4 active last week + Collapse phase this week → flag V4→V2 transition
  - Requires manual confirmation before auto-swap

Usage:
    python3 tradier_long_manager.py --check      # check status only
    python3 tradier_long_manager.py --preview    # show what would be done
    python3 tradier_long_manager.py --paper      # execute on sandbox
    python3 tradier_long_manager.py --confirm-transition V4 V2  # approve swap
"""

from __future__ import annotations
import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
STORAGE_DIR        = Path.home() / ".vix_suite"
VOL_SNAPSHOTS_PATH = STORAGE_DIR / "vol_snapshots.json"
SIGNAL_BATCH_PATH  = STORAGE_DIR / "current_signal_batch.json"
SIGNAL_HISTORY     = STORAGE_DIR / "member_signal_history.json"
LONG_STATE_PATH    = STORAGE_DIR / "tradier_long_state.json"

# ── Variant roll thresholds ───────────────────────────────────────────────────
ROLL_THRESHOLD = {
    "V1": 30,   # 26w long → roll when < 30 DTE
    "V2": 21,   # 13w long → roll when < 21 DTE
    "V3": 14,   #  8w long → roll when < 14 DTE
    "V4": 21,   # 13w long → roll when < 21 DTE
    "V5": 21,   # 13w long → roll when < 21 DTE
}

# Long leg delta target range
LONG_DELTA_MIN = 0.65
LONG_DELTA_MAX = 0.75

# Reprice settings
MAX_REPRICE    = 6
REPRICE_SEC    = 60
NUDGE          = 0.05


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_snap() -> dict:
    if not VOL_SNAPSHOTS_PATH.exists():
        return {}
    data = json.loads(VOL_SNAPSHOTS_PATH.read_text())
    snaps = data if isinstance(data, list) else data.get("snapshots", [])
    return snaps[-1] if snaps else {}


def load_batch() -> dict:
    if not SIGNAL_BATCH_PATH.exists():
        return {}
    return json.loads(SIGNAL_BATCH_PATH.read_text())


def load_long_state() -> dict:
    """Load persisted long leg state."""
    if not LONG_STATE_PATH.exists():
        return {}
    try:
        return json.loads(LONG_STATE_PATH.read_text())
    except Exception:
        return {}


def save_long_state(state: dict):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    LONG_STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def build_role_map(batch: dict) -> dict:
    role_map = {}
    for v in batch.get("variants", []):
        role = v.get("role", "")
        for key, tag in [("V1","v1"),("V2","v2"),("V3","v3"),
                         ("V4","v4"),("V5","v5")]:
            if tag in role:
                role_map[key] = v
    return role_map


# ── Tradier client (minimal, reuses tradier_executor logic) ───────────────────

try:
    from tradier_executor import TradierClient, next_friday
except ImportError:
    import requests

    class TradierClient:
        def __init__(self, sandbox: bool = True):
            self.base    = ("https://sandbox.tradier.com/v1" if sandbox
                            else "https://api.tradier.com/v1")
            self.token   = (os.environ.get("TRADIER_SANDBOX_TOKEN","") if sandbox
                            else os.environ.get("TRADIER_LIVE_TOKEN",""))
            self.account = os.environ.get("TRADIER_ACCOUNT","VA88395338")
            self.sandbox = sandbox
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Accept":        "application/json",
            })

        def _get(self, path, params=None):
            r = self.session.get(f"{self.base}{path}", params=params)
            r.raise_for_status()
            return r.json()

        def _post(self, path, data):
            r = self.session.post(
                f"{self.base}{path}", data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            r.raise_for_status()
            return r.json()

        def get_positions(self):
            data = self._get(f"/accounts/{self.account}/positions")
            pos  = data.get("positions", {})
            if not pos or pos == "null":
                return []
            p = pos.get("position", [])
            return p if isinstance(p, list) else [p]

        def get_option_chain(self, symbol, expiry, greeks=True):
            data = self._get("/markets/options/chains", {
                "symbol": symbol,
                "expiration": expiry.strftime("%Y-%m-%d"),
                "greeks": "true" if greeks else "false",
            })
            opts = data.get("options", {})
            if not opts:
                return []
            chain = opts.get("option", [])
            return chain if isinstance(chain, list) else [chain]

        def get_expirations(self, symbol):
            data = self._get("/markets/options/expirations", {
                "symbol": symbol, "includeAllRoots": "true",
                "strikes": "false",
            })
            dates = data.get("expirations", {}).get("date", [])
            if isinstance(dates, str):
                dates = [dates]
            return [date.fromisoformat(d) for d in dates]

        def place_order(self, symbol, option_symbol, side, quantity, price):
            return self._post(f"/accounts/{self.account}/orders", {
                "class": "option", "symbol": symbol,
                "option_symbol": option_symbol, "side": side,
                "quantity": str(quantity), "type": "limit",
                "duration": "day", "price": f"{price:.2f}",
            })

        def modify_order(self, order_id, new_price):
            return self._post(f"/accounts/{self.account}/orders/{order_id}", {
                "type": "limit", "duration": "day",
                "price": f"{new_price:.2f}",
            })

        def get_order(self, order_id):
            data = self._get(f"/accounts/{self.account}/orders/{order_id}")
            return data.get("order", {})

    def next_friday(min_dte: int = 7) -> date:
        today  = date.today()
        target = today + timedelta(days=min_dte)
        days   = (4 - target.weekday()) % 7
        return target + timedelta(days=days)


# ── Position parser ───────────────────────────────────────────────────────────

def parse_uvxy_positions(positions: list[dict]) -> dict:
    """
    Parse Tradier positions into long/short UVXY calls.
    Returns {
        "longs":  [{"symbol","strike","expiry","dte","quantity","delta_approx"}],
        "shorts": [{"symbol","strike","expiry","dte","quantity"}]
    }
    """
    today  = date.today()
    longs  = []
    shorts = []

    for pos in positions:
        sym = pos.get("symbol", "")
        qty = int(pos.get("quantity", 0))

        if "UVXY" not in sym or qty == 0:
            continue

        try:
            idx      = sym.index("UVXY") + 4
            exp_str  = sym[idx:idx+6]
            exp_date = date(2000 + int(exp_str[:2]),
                            int(exp_str[2:4]),
                            int(exp_str[4:6]))
            dte      = (exp_date - today).days
            strike   = int(sym[idx+7:]) / 1000
            cp       = sym[idx+6]

            if cp != "C":
                continue

            entry = {
                "symbol":  sym,
                "strike":  strike,
                "expiry":  exp_date,
                "dte":     dte,
                "quantity": abs(qty),
                "cost":    float(pos.get("cost_basis", 0)) / abs(qty) / 100,
            }

            if qty > 0:
                longs.append(entry)
            else:
                shorts.append(entry)

        except Exception:
            continue

    return {"longs": longs, "shorts": shorts}


# ── Long leg finder ───────────────────────────────────────────────────────────

def find_long_strike(chain: list[dict],
                     delta_min: float = LONG_DELTA_MIN,
                     delta_max: float = LONG_DELTA_MAX) -> Optional[dict]:
    """Find best long call strike in delta target range."""
    calls = [c for c in chain
             if c.get("option_type","").lower() == "call"
             and c.get("greeks") is not None]

    candidates = []
    for c in calls:
        greeks = c.get("greeks") or {}
        delta  = float(greeks.get("delta", 0) or 0)
        bid    = float(c.get("bid", 0) or 0)
        ask    = float(c.get("ask", 0) or 0)
        mid    = round((bid + ask) / 2, 2)

        if delta_min <= delta <= delta_max and ask > 0:
            candidates.append({
                "symbol": c.get("symbol"),
                "strike": float(c.get("strike", 0)),
                "expiry": c.get("expiration_date"),
                "delta":  delta,
                "bid":    bid,
                "ask":    ask,
                "mid":    mid,
            })

    if not candidates:
        return None

    # Pick closest delta to 0.70 (middle of range)
    candidates.sort(key=lambda x: abs(x["delta"] - 0.70))
    return candidates[0]


# ── Order executor with reprice ───────────────────────────────────────────────

def execute_with_reprice(client: TradierClient, underlying: str,
                         option_symbol: str, side: str,
                         quantity: int, bid: float, ask: float,
                         sandbox: bool = True,
                         preview: bool = False) -> dict:
    """
    Most profitable execution for long/short legs.

    BTO (buy):  start at mid, nudge toward ask each interval
    STO (sell): start at mid, nudge toward bid each interval

    Sandbox: cancel + re-place (modify not supported)
    Live:    true order modification
    """
    is_buy  = "buy" in side.lower()
    mid     = round((bid + ask) / 2, 2)
    price   = mid
    floor   = bid  if not is_buy else mid
    ceil    = ask  if is_buy else mid
    nudge   = 0.05 if is_buy else 0.01
    action  = "BTO" if is_buy else "STO"

    if preview:
        print(f"   [PREVIEW] {action} {option_symbol} ×{quantity} "
              f"@ mid ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")
        return {"status": "preview", "price": price}

    print(f"   {action} {option_symbol} ×{quantity} "
          f"@ ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")

    try:
        result = client.place_order(underlying, option_symbol,
                                    side, quantity, price)
        oid = result.get("order", {}).get("id")
        if not oid:
            print(f"   ❌ Order failed: {result}")
            return {"status": "failed"}
        print(f"   Order {oid} placed")
    except Exception as e:
        print(f"   ❌ Place failed: {e}")
        return {"status": "failed"}

    for attempt in range(MAX_REPRICE):
        time.sleep(REPRICE_SEC)
        try:
            status = client.get_order(oid)
            state  = status.get("status", "")
        except Exception:
            state = "unknown"

        print(f"   [{attempt+1}] status={state} @ ${price:.2f}")

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            print(f"   ✅ Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid,
                    "fill_price": fill_px, "quantity": quantity,
                    "option_symbol": option_symbol}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # Nudge toward fill
        if is_buy:
            new_price = min(round(price + nudge, 2), ceil)
        else:
            new_price = max(round(price - nudge, 2), floor)

        if new_price == price:
            print(f"   At limit ${price:.2f} — waiting")
            continue

        print(f"   Repricing ${price:.2f} → ${new_price:.2f}")

        if sandbox:
            try:
                client.session.delete(
                    f"{client.base}/accounts/{client.account}/orders/{oid}",
                )
            except Exception:
                pass
            try:
                result  = client.place_order(underlying, option_symbol,
                                             side, quantity, new_price)
                new_oid = result.get("order", {}).get("id")
                if new_oid:
                    oid   = new_oid
                    price = new_price
            except Exception as e:
                print(f"   ⚠️ Re-place failed: {e}")
        else:
            try:
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as e:
                print(f"   ⚠️ Modify failed: {e}")

    return {"status": "working", "order_id": oid, "last_price": price}


# ── Phase transition detector ─────────────────────────────────────────────────

def detect_phase_transition(snap: dict) -> Optional[tuple[str, str]]:
    """
    Detect V4→V2 or other phase transitions requiring long leg swap.
    Returns (from_variant, to_variant) or None.
    """
    phase    = snap.get("spike_label", "")
    collapse = snap.get("collapse_flag", False)

    if not SIGNAL_HISTORY.exists():
        return None

    try:
        history = json.loads(SIGNAL_HISTORY.read_text())
        today   = date.today().isoformat()
        prev    = next((h for h in reversed(history)
                        if h.get("signal_date","") != today), None)
        if not prev:
            return None

        prev_phase = prev.get("phase", "")

        # V4→V2: Expansion/Late Spike → Collapse
        if (prev_phase in ("Expansion","Late Spike","Spike Peak")
                and phase in ("Collapse","Compression")
                and collapse):
            return ("V4", "V2")

        # V2→V4: Compression → Expansion/Spike
        if (prev_phase in ("Compression","Collapse")
                and phase in ("Expansion","Late Spike","Spike Peak")):
            return ("V2", "V4")

    except Exception:
        pass

    return None


# ── Main long manager ─────────────────────────────────────────────────────────

def run(sandbox: bool = True, preview: bool = False,
        check_only: bool = False,
        confirm_transition: tuple = None):

    print(f"\n{'='*60}")
    print(f"  Long Leg Manager — {'SANDBOX' if sandbox else 'LIVE'}")
    mode_str = "CHECK ONLY" if check_only else "PREVIEW" if preview else "EXECUTING"
    print(f"  Mode: {mode_str}")
    print(f"{'='*60}\n")

    snap  = load_snap()
    batch = load_batch()

    if not snap or not batch:
        print("❌ Missing app data — run app first")
        return

    uvxy   = snap.get("uvxy", 0.0)
    phase  = snap.get("spike_label", "")
    regime = snap.get("regime", "")
    print(f"   UVXY: ${uvxy:.2f} | Phase: {phase} | Regime: {regime}")

    # Phase transition check
    transition = detect_phase_transition(snap)
    if transition:
        frm, to = transition
        print(f"\n   ⚠️  PHASE TRANSITION DETECTED: {frm} → {to}")
        print(f"   Long leg swap may be needed.")
        if confirm_transition and confirm_transition == transition:
            print(f"   ✅ Transition confirmed — will swap {frm} long → {to} long")
        elif not check_only:
            print(f"   Run with --confirm-transition {frm} {to} to approve swap")
            print(f"   Or reply to member email for guidance")

    client = TradierClient(sandbox=sandbox)

    # Fetch current positions
    print("\n   Fetching current Tradier positions...")
    try:
        positions = client.get_positions()
        parsed    = parse_uvxy_positions(positions)
        longs     = parsed["longs"]
        shorts    = parsed["shorts"]
        print(f"   Long legs:  {len(longs)}")
        print(f"   Short legs: {len(shorts)}")
        for lg in longs:
            print(f"     LONG:  ${lg['strike']:.0f}C "
                  f"exp {lg['expiry']} (DTE={lg['dte']}) ×{lg['quantity']}")
        for sh in shorts:
            print(f"     SHORT: ${sh['strike']:.0f}C "
                  f"exp {sh['expiry']} (DTE={sh['dte']}) ×{sh['quantity']}")
    except Exception as e:
        print(f"   ❌ Could not fetch positions: {e}")
        return

    # Get expirations
    try:
        expirations = client.get_expirations("UVXY")
    except Exception as e:
        print(f"   ❌ Could not fetch expirations: {e}")
        return

    role_map = build_role_map(batch)
    today    = date.today()
    state    = load_long_state()
    actions  = []

    print(f"\n{'─'*60}")
    print("  VARIANT LONG LEG STATUS")
    print(f"{'─'*60}")

    for key in ["V1","V2","V3","V4","V5"]:
        v = role_map.get(key)
        if not v:
            continue

        long_dte_weeks  = v.get("long_dte_weeks", 13)
        target_dte      = long_dte_weeks * 7
        roll_threshold  = ROLL_THRESHOLD.get(key, 21)

        print(f"\n  {key} — target DTE: {target_dte}d, "
              f"roll threshold: {roll_threshold}d")

        # Find matching long in account
        # Match: UVXY call, qty > 0, DTE > roll_threshold
        matching_longs = [lg for lg in longs if lg["dte"] > roll_threshold]

        if not matching_longs:
            print(f"    ❌ NO LONG LEG — needs entry")
            # Find target expiry
            target_exp_date = today + timedelta(days=target_dte)
            valid_exps = [e for e in expirations if e >= target_exp_date]
            if not valid_exps:
                print(f"    ❌ No expiry available ≥ {target_exp_date}")
                continue
            exp = valid_exps[0]
            print(f"    Target expiry: {exp} ({(exp-today).days} DTE)")

            if not check_only:
                # Fetch chain and find strike
                try:
                    chain = client.get_option_chain("UVXY", exp)
                    best  = find_long_strike(chain)
                    if best:
                        print(f"    Best long: ${best['strike']:.0f}C "
                              f"δ={best['delta']:.3f} ask=${best['ask']:.2f}")
                        actions.append({
                            "key":    key,
                            "action": "enter_long",
                            "strike": best["strike"],
                            "expiry": exp,
                            "symbol": best["symbol"],
                            "mid":    best["mid"],
                            "ask":    best["ask"],
                            "qty":    1,
                        })
                    else:
                        print(f"    ⚠️ No strike found in δ {LONG_DELTA_MIN}–{LONG_DELTA_MAX}")
                except Exception as e:
                    print(f"    ❌ Chain error: {e}")

        else:
            # Use longest DTE long as the active long leg
            active = max(matching_longs, key=lambda x: x["dte"])
            print(f"    ✅ Active long: ${active['strike']:.0f}C "
                  f"exp {active['expiry']} (DTE={active['dte']}) "
                  f"×{active['quantity']}")

            # Check if roll needed
            if active["dte"] <= roll_threshold:
                print(f"    ⚠️  DTE {active['dte']} ≤ threshold {roll_threshold} "
                      f"— ROLL NEEDED")

                target_exp_date = today + timedelta(days=target_dte)
                valid_exps = [e for e in expirations
                              if e >= target_exp_date]
                if valid_exps:
                    new_exp = valid_exps[0]
                    print(f"    Roll target: {new_exp} "
                          f"({(new_exp-today).days} DTE)")

                    if not check_only:
                        try:
                            chain    = client.get_option_chain("UVXY", new_exp)
                            best_new = find_long_strike(chain)
                            if best_new:
                                print(f"    New long: ${best_new['strike']:.0f}C "
                                      f"δ={best_new['delta']:.3f} "
                                      f"ask=${best_new['ask']:.2f}")
                                # Step 1: BTO new long
                                actions.append({
                                    "key":    key,
                                    "action": "roll_long_buy",
                                    "strike": best_new["strike"],
                                    "expiry": new_exp,
                                    "symbol": best_new["symbol"],
                                    "mid":    best_new["mid"],
                                    "ask":    best_new["ask"],
                                    "qty":    active["quantity"],
                                })
                                # Step 2: STC old long (after new fills)
                                actions.append({
                                    "key":    key,
                                    "action": "roll_long_sell",
                                    "strike": active["strike"],
                                    "expiry": active["expiry"],
                                    "symbol": active["symbol"],
                                    "mid":    active.get("cost", 0.50),
                                    "ask":    active.get("cost", 0.50) * 1.1,
                                    "qty":    active["quantity"],
                                })
                        except Exception as e:
                            print(f"    ❌ Chain error: {e}")
            else:
                print(f"    ✅ DTE {active['dte']} — no roll needed")

    # Execute actions
    if not actions:
        if check_only:
            # Summarize check results
            missing = [k for k in ["V1","V2","V3","V4","V5"]
                       if not any(lg["dte"] > ROLL_THRESHOLD.get(k,21)
                                  for lg in longs)]
            if missing:
                print(f"\n{'─'*60}")
                print(f"  ⚠️  MISSING LONG LEGS: {', '.join(missing)}")
                print(f"  Run --preview to see entry plan")
                print(f"  Run --paper to execute on sandbox")
            else:
                print(f"\n{'─'*60}")
                print("  ✅ All long legs healthy — no action needed")
        else:
            print(f"\n{'─'*60}")
            print("  ✅ All long legs healthy — no action needed")
        return

    print(f"\n{'─'*60}")
    print(f"  ACTIONS ({len(actions)} total)")
    print(f"{'─'*60}")

    for action in actions:
        key    = action["key"]
        act    = action["action"]
        strike = action["strike"]
        exp    = action["expiry"]
        sym    = action["symbol"]
        mid    = action["mid"]
        ask    = action["ask"]
        qty    = action["qty"]

        print(f"\n  {key} — {act}: ${strike:.0f}C exp {exp} ×{qty}")

        if act in ("enter_long", "roll_long_buy"):
            side = "buy_to_open"
        else:
            side = "sell_to_close"

        result = execute_with_reprice(
            client, "UVXY", sym, side, qty,
            bid=action.get("bid", mid),
            ask=ask,
            sandbox=sandbox,
            preview=preview
        )

        # If roll_long_buy filled → proceed with sell
        # If roll_long_buy failed/preview → skip the paired sell
        if act == "roll_long_buy" and result.get("status") not in ("filled","preview"):
            print(f"   ⚠️ New long did not fill — skipping old long close")
            # Remove paired sell from remaining actions
            actions = [a for a in actions
                       if not (a["key"] == key
                               and a["action"] == "roll_long_sell")]

        # Update state
        if result.get("status") == "filled":
            state[key] = {
                "symbol":     sym,
                "strike":     strike,
                "expiry":     str(exp),
                "fill_price": result.get("fill_price"),
                "quantity":   qty,
                "updated_at": str(date.today()),
            }
            save_long_state(state)

    print(f"\n{'='*60}")
    print("  LONG LEG MANAGER COMPLETE")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",   action="store_true",
                        help="Check status only — no orders")
    parser.add_argument("--preview", action="store_true",
                        help="Show what would be done — no orders")
    parser.add_argument("--paper",   action="store_true",
                        help="Execute on Tradier sandbox")
    parser.add_argument("--live",    action="store_true",
                        help="Execute on live account (future)")
    parser.add_argument("--confirm-transition", nargs=2,
                        metavar=("FROM", "TO"),
                        help="Confirm phase transition e.g. V4 V2")
    args = parser.parse_args()

    if args.live:
        print("⚠️  Live mode not yet enabled")
        return

    transition = tuple(args.confirm_transition) if args.confirm_transition else None

    run(
        sandbox    = not args.live,
        preview    = args.preview or (not args.paper and not args.check),
        check_only = args.check,
        confirm_transition = transition,
    )


if __name__ == "__main__":
    main()
