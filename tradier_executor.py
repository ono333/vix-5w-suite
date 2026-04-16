#!/usr/bin/env python3
"""
tradier_executor.py
───────────────────
Automated short leg execution for UVXY diagonal spreads via Tradier API.
Reads signal from app's existing data sources — never invents parameters.

Sandbox: https://sandbox.tradier.com
Live:    https://api.tradier.com

Usage:
    python3 tradier_executor.py --preview          # show what would be ordered
    python3 tradier_executor.py --paper            # execute on sandbox
    python3 tradier_executor.py --live             # execute on live account (future)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

# ── Config ────────────────────────────────────────────────────────────────────
SANDBOX_BASE  = "https://sandbox.tradier.com/v1"
LIVE_BASE     = "https://api.tradier.com/v1"
SANDBOX_TOKEN = os.environ.get("TRADIER_SANDBOX_TOKEN", "")
LIVE_TOKEN    = os.environ.get("TRADIER_LIVE_TOKEN", "")

SANDBOX_ACCOUNT = "VA88395338"

STORAGE_DIR        = Path.home() / ".vix_suite"
VOL_SNAPSHOTS_PATH = STORAGE_DIR / "vol_snapshots.json"
SIGNAL_BATCH_PATH  = STORAGE_DIR / "current_signal_batch.json"

# Mid-price reprice settings
MAX_REPRICE_ATTEMPTS = 8
REPRICE_INTERVAL_SEC = 90    # reprice every 90 seconds
NUDGE_PER_ATTEMPT    = 0.02  # nudge toward ask by $0.02 each attempt


# ── OCC Symbol builder ────────────────────────────────────────────────────────

def occ_symbol(underlying: str, expiry: date, call_put: str,
               strike: float) -> str:
    """
    Build OCC option symbol.
    Format: UVXY260424C00044000
    """
    exp_str   = expiry.strftime("%y%m%d")
    cp        = call_put.upper()[0]
    strike_i  = int(round(strike * 1000))
    return f"{underlying}{exp_str}{cp}{strike_i:08d}"


# ── Tradier API client ────────────────────────────────────────────────────────

class TradierClient:
    def __init__(self, sandbox: bool = True):
        self.base    = SANDBOX_BASE if sandbox else LIVE_BASE
        self.token   = SANDBOX_TOKEN if sandbox else LIVE_TOKEN
        self.account = SANDBOX_ACCOUNT
        self.sandbox = sandbox
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept":        "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        r = self.session.get(f"{self.base}{path}", params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = self.session.post(
            f"{self.base}{path}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        r.raise_for_status()
        return r.json()

    def get_quote(self, symbol: str) -> float:
        """Get last price for a symbol."""
        data = self._get("/markets/quotes", {"symbols": symbol, "greeks": "false"})
        quotes = data.get("quotes", {}).get("quote", {})
        if isinstance(quotes, list):
            quotes = quotes[0]
        return float(quotes.get("last", 0) or quotes.get("bid", 0))

    def get_option_chain(self, symbol: str, expiry: date,
                         greeks: bool = True) -> list[dict]:
        """Get option chain for a specific expiry."""
        data = self._get("/markets/options/chains", {
            "symbol":     symbol,
            "expiration": expiry.strftime("%Y-%m-%d"),
            "greeks":     "true" if greeks else "false",
        })
        options = data.get("options", {})
        if not options:
            return []
        chain = options.get("option", [])
        return chain if isinstance(chain, list) else [chain]

    def get_expirations(self, symbol: str) -> list[date]:
        """Get available expiration dates."""
        data = self._get("/markets/options/expirations", {
            "symbol":           symbol,
            "includeAllRoots":  "true",
            "strikes":          "false",
        })
        dates = data.get("expirations", {}).get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return [date.fromisoformat(d) for d in dates]

    def preview_order(self, symbol: str, option_symbol: str, side: str,
                      quantity: int, price: float) -> dict:
        """Preview an option order — no execution."""
        return self._post(f"/accounts/{self.account}/orders", {
            "class":         "option",
            "symbol":        symbol,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(quantity),
            "type":          "limit",
            "duration":      "day",
            "price":         f"{price:.2f}",
            "preview":       "true",
        })

    def place_order(self, symbol: str, option_symbol: str, side: str,
                    quantity: int, price: float) -> dict:
        """Place a live limit order."""
        return self._post(f"/accounts/{self.account}/orders", {
            "class":         "option",
            "symbol":        symbol,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(quantity),
            "type":          "limit",
            "duration":      "day",
            "price":         f"{price:.2f}",
        })

    def modify_order(self, order_id: int, new_price: float) -> dict:
        """Modify existing order price."""
        return self._post(f"/accounts/{self.account}/orders/{order_id}", {
            "type":     "limit",
            "duration": "day",
            "price":    f"{new_price:.2f}",
        })

    def get_order(self, order_id: int) -> dict:
        """Get order status."""
        data = self._get(f"/accounts/{self.account}/orders/{order_id}")
        return data.get("order", {})

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        data = self._get(f"/accounts/{self.account}/positions")
        positions = data.get("positions", {})
        if not positions or positions == "null":
            return []
        pos = positions.get("position", [])
        return pos if isinstance(pos, list) else [pos]

    def get_balances(self) -> dict:
        """Get account balances."""
        data = self._get(f"/accounts/{self.account}/balances")
        return data.get("balances", {})


# ── Signal readers ────────────────────────────────────────────────────────────

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


def build_role_map(batch: dict) -> dict:
    role_map = {}
    for v in batch.get("variants", []):
        role = v.get("role", "")
        for key, tag in [("V1","v1"),("V2","v2"),("V3","v3"),
                         ("V4","v4"),("V5","v5")]:
            if tag in role:
                role_map[key] = v
    return role_map


def delta_ceil(v: dict) -> float:
    sm = v.get("sigma_mult", 1.0)
    if sm <= 0.8: return 0.28
    if sm <= 0.9: return 0.25
    if sm <= 1.0: return 0.22
    if sm <= 1.2: return 0.18
    return 0.15


# ── Next Friday expiry ────────────────────────────────────────────────────────

def next_friday(min_dte: int = 7) -> date:
    """Return nearest Friday with at least min_dte days."""
    today   = date.today()
    target  = today + timedelta(days=min_dte)
    days    = (4 - target.weekday()) % 7
    return target + timedelta(days=days)


# ── Strike finder ─────────────────────────────────────────────────────────────

def find_best_strike(chain: list[dict], dc: float,
                     min_credit: float = 0.30) -> Optional[dict]:
    """
    Find the best strike from option chain:
    - Delta ≤ ceiling
    - Credit (bid) ≥ min_credit
    - Furthest OTM that still pays target credit
    """
    calls = [c for c in chain
             if c.get("option_type", "").lower() == "call"
             and c.get("greeks") is not None]

    candidates = []
    for c in calls:
        greeks = c.get("greeks") or {}
        delta  = abs(float(greeks.get("delta", 1.0) or 1.0))
        bid    = float(c.get("bid", 0) or 0)
        ask    = float(c.get("ask", 0) or 0)
        mid    = round((bid + ask) / 2, 2)

        if delta <= dc and bid >= min_credit:
            candidates.append({
                "symbol":  c.get("symbol"),
                "strike":  float(c.get("strike", 0)),
                "expiry":  c.get("expiration_date"),
                "delta":   delta,
                "bid":     bid,
                "ask":     ask,
                "mid":     mid,
            })

    if not candidates:
        return None

    # Sort by delta descending — closest to ceiling = most premium
    candidates.sort(key=lambda x: x["delta"], reverse=True)
    return candidates[0]


# ── Mid-price executor ────────────────────────────────────────────────────────

def execute_with_reprice(client: TradierClient, underlying: str,
                         option_symbol: str, side: str,
                         quantity: int, bid: float, ask: float,
                         sandbox: bool = True,
                         dry_run: bool = False) -> dict:
    """
    Most profitable execution algorithm for UVXY options.

    STO (sell): start at mid, nudge toward bid each interval
    BTO (buy):  start at mid, nudge toward ask each interval

    Sandbox: cancel + re-place (no modify API)
    Live:    true order modification (single cancel-free reprice)

    Never goes below bid (STO) or above ask (BTO).
    Skips if bid ≤ $0.05 (no liquidity).
    """
    is_sell = "sell" in side.lower()

    # Liquidity check for sells
    if is_sell and bid <= 0.05:
        print(f"   ⚠️ Bid ${bid:.2f} too low — no liquidity, skipping")
        return {"status": "skipped_low_bid"}

    mid   = round((bid + ask) / 2, 2)
    price = mid
    floor = bid   if is_sell else mid   # never go below bid on sells
    ceil  = mid   if is_sell else ask   # never go above ask on buys
    direction = "toward bid" if is_sell else "toward ask"
    nudge     = 0.01  # $0.01 per interval — fine-grained

    if dry_run:
        action = "STO" if is_sell else "BTO"
        print(f"   [DRY RUN] {action} {option_symbol} ×{quantity} "
              f"@ mid ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")
        return {"status": "dry_run", "price": price}

    action = "STO" if is_sell else "BTO"
    print(f"   {action} {option_symbol} ×{quantity} "
          f"@ ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")

    # Place initial order
    try:
        result = client.place_order(underlying, option_symbol,
                                    side, quantity, price)
        oid = result.get("order", {}).get("id")
        if not oid:
            print(f"   ❌ Order failed: {result}")
            return {"status": "failed"}
        print(f"   Order {oid} placed @ ${price:.2f}")
    except Exception as e:
        print(f"   ❌ Place failed: {e}")
        return {"status": "failed"}

    for attempt in range(MAX_REPRICE_ATTEMPTS):
        time.sleep(REPRICE_INTERVAL_SEC)

        try:
            status = client.get_order(oid)
            state  = status.get("status", "")
        except Exception:
            state = "unknown"

        print(f"   [{attempt+1}] status={state} price=${price:.2f}")

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            print(f"   ✅ Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid,
                    "fill_price": fill_px, "quantity": quantity}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # Calculate new price — nudge toward floor/ceil
        if is_sell:
            new_price = max(round(price - nudge, 2), floor)
        else:
            new_price = min(round(price + nudge, 2), ceil)

        if new_price == price:
            print(f"   At limit (${price:.2f}) — waiting for fill")
            continue

        print(f"   Repricing ${price:.2f} → ${new_price:.2f} ({direction})")

        if sandbox:
            # Sandbox: cancel + re-place
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
                    print(f"   New order {oid} @ ${price:.2f}")
            except Exception as e:
                print(f"   ⚠️ Re-place failed: {e}")
        else:
            # Live: true modify
            try:
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as e:
                print(f"   ⚠️ Modify failed: {e}")

    print(f"   ⚠️ Max attempts — order {oid} still working @ ${price:.2f}")
    return {"status": "working", "order_id": oid, "last_price": price}


# ── Main execution logic ──────────────────────────────────────────────────────

def run(sandbox: bool = True, preview: bool = False,
        variant_filter: list[str] = None):
    """
    Main executor — reads signal batch, finds strikes, executes orders.
    """
    print(f"\n{'='*60}")
    print(f"  Tradier Executor — {'SANDBOX' if sandbox else 'LIVE'} mode")
    print(f"  {'PREVIEW ONLY' if preview else 'EXECUTING'}")
    print(f"{'='*60}\n")

    # Load app data
    snap  = load_snap()
    batch = load_batch()

    if not snap:
        print("❌ No vol snapshot — run app first")
        return
    if not batch:
        print("❌ No signal batch — generate signals first")
        return

    uvxy  = snap.get("uvxy", 0.0)
    phase = snap.get("spike_label", "")
    regime = snap.get("regime", "")
    print(f"   UVXY: ${uvxy:.2f} | Phase: {phase} | Regime: {regime}")

    # Connect
    client = TradierClient(sandbox=sandbox)

    # Get available expiries
    print("\n   Fetching UVXY expirations...")
    try:
        expirations = client.get_expirations("UVXY")
        print(f"   Available: {[e.strftime('%b %d') for e in expirations[:6]]}")
    except Exception as e:
        print(f"   ❌ Failed to get expirations: {e}")
        return

    # Target expiry — next Friday ≥ 7 DTE
    target_exp = next_friday(min_dte=7)
    # Find nearest available expiry ≥ target
    valid_exps = [e for e in expirations if e >= target_exp]
    if not valid_exps:
        print(f"   ❌ No expiry found ≥ {target_exp}")
        return
    exp = valid_exps[0]
    print(f"   Target expiry: {exp.strftime('%b %d, %Y')}")

    # Get chain
    print(f"\n   Fetching option chain for {exp}...")
    try:
        chain = client.get_option_chain("UVXY", exp)
        print(f"   Chain: {len(chain)} options")
    except Exception as e:
        print(f"   ❌ Chain fetch failed: {e}")
        return

    # Get balances
    try:
        bal = client.get_balances()
        margin = bal.get("margin", {})
        bp = margin.get("option_buying_power", 0)
        print(f"\n   Buying power: ${float(bp):,.0f}")
    except Exception as e:
        print(f"   ⚠️ Could not fetch balances: {e}")

    # ── Long leg safety check ─────────────────────────────────────────────
    print("\n   Checking existing positions for long legs...")
    try:
        positions = client.get_positions()
        print(f"   Found {len(positions)} position(s) in account")
    except Exception as e:
        print(f"   ❌ Could not fetch positions: {e}")
        return

    # Parse long legs — UVXY calls with DTE > 21 (LEAPs / longer dated)
    today = date.today()
    long_legs = []
    for pos in positions:
        sym = pos.get("symbol", "")
        qty = int(pos.get("quantity", 0))
        if "UVXY" in sym and qty > 0:  # long = positive quantity
            # Parse expiry from OCC symbol e.g. UVXY260918C00045000
            try:
                # OCC format: UVXYYYMMDDCSSSSSS
                uvxy_idx = sym.index("UVXY") + 4
                exp_str  = sym[uvxy_idx:uvxy_idx+6]
                exp_date = date(2000 + int(exp_str[:2]),
                                int(exp_str[2:4]),
                                int(exp_str[4:6]))
                dte      = (exp_date - today).days
                strike   = int(sym[uvxy_idx+7:]) / 1000
                cp       = sym[uvxy_idx+6]
                if cp == "C" and dte > 21:  # call with > 21 DTE = long leg
                    long_legs.append({
                        "symbol":   sym,
                        "strike":   strike,
                        "expiry":   exp_date,
                        "dte":      dte,
                        "quantity": qty,
                    })
                    print(f"   ✅ Long leg found: ${strike:.0f}C "
                          f"exp {exp_date} (DTE={dte}) ×{qty}")
            except Exception:
                pass

    if not long_legs:
        print("   ⚠️  NO LONG LEGS FOUND IN TRADIER ACCOUNT")
        print("   You must own UVXY long calls before selling shorts.")
        print("   Buy long legs first, then re-run executor.")
        return

    # Use highest-strike long as reference (most protective)
    best_long = max(long_legs, key=lambda x: x["strike"])
    print(f"   Using long: ${best_long['strike']:.0f}C "
          f"exp {best_long['expiry']} as protection")

    # Process each variant
    role_map = build_role_map(batch)
    variants_to_run = variant_filter or ["V1","V2","V3","V4","V5"]

    print(f"\n{'─'*60}")
    results = {}
    pending_orders = {}  # key → order details

    # ── Phase 1: Place all orders simultaneously ───────────────────────────
    print("\n  Phase 1 — Placing all orders...")
    for key in variants_to_run:
        v = role_map.get(key)
        if not v:
            print(f"  {key}: not in signal batch — skip")
            continue

        dc         = delta_ceil(v)
        offset     = v.get("short_strike_offset", 2.0)
        ref_strike = round(uvxy + offset)
        contracts  = 1

        print(f"\n  {key} — delta ceiling: {dc:.2f}, ref strike: ${ref_strike}")

        best = find_best_strike(chain, dc, min_credit=0.25)

        if not best:
            print(f"  ⚠️  No strike found for {key}")
            results[key] = {"status": "no_strike"}
            continue

        if best["strike"] <= best_long["strike"]:
            print(f"  ❌ BLOCKED: short ${best['strike']:.0f}C ≤ "
                  f"long ${best_long['strike']:.0f}C")
            results[key] = {"status": "blocked_long_safety"}
            continue

        print(f"  {key}: ${best['strike']:.0f}C δ={best['delta']:.3f} "
              f"bid=${best['bid']:.2f} ask=${best['ask']:.2f} "
              f"mid=${best['mid']:.2f}")

        if preview:
            print(f"  [PREVIEW] STO ${best['strike']:.0f}C ×{contracts} "
                  f"@ ${best['mid']:.2f}")
            results[key] = {"status": "preview", "strike": best["strike"],
                            "expiry": str(exp), "mid": best["mid"],
                            "delta": best["delta"]}
            continue

        # Place at bid for sell orders (guaranteed fill for STO)
        place_price = best["bid"]
        try:
            result = client.place_order("UVXY", best["symbol"],
                                        "sell_to_open", contracts, place_price)
            order  = result.get("order", {})
            oid    = order.get("id")
        except Exception as _oe:
            print(f"  {key}: ❌ Timeout: {_oe} — retrying in 5s")
            time.sleep(5)
            try:
                result = client.place_order("UVXY", best["symbol"],
                                            "sell_to_open", contracts, place_price)
                order = result.get("order", {})
                oid   = order.get("id")
            except Exception as _oe2:
                print(f"  {key}: ❌ Retry failed: {_oe2}")
                results[key] = {"status": "failed"}
                continue
        if oid:
            print(f"  {key}: Order {oid} placed @ ${place_price:.2f}")
            pending_orders[key] = {
                "order_id": oid,
                "strike":   best["strike"],
                "expiry":   str(exp),
                "delta":    best["delta"],
                "price":    place_price,
                "ask":      best["ask"],
                "symbol":   best["symbol"],
            }
        else:
            print(f"  {key}: ❌ Order failed: {result}")
            results[key] = {"status": "failed"}

    if not pending_orders or preview:
        # Skip polling for preview mode
        pass
    else:
        # ── Phase 2: Poll all orders together ─────────────────────────────
        print(f"\n  Phase 2 — Polling {len(pending_orders)} orders...")
        for attempt in range(MAX_REPRICE_ATTEMPTS):
            time.sleep(10)  # short poll interval since all placed at ask
            all_done = True
            for key, od in list(pending_orders.items()):
                if key in results:
                    continue
                status = client.get_order(od["order_id"])
                state  = status.get("status", "")
                if state == "filled":
                    fill_px = float(status.get("avg_fill_price", od["price"]))
                    print(f"  ✅ {key}: filled @ ${fill_px:.2f}")
                    results[key] = {
                        "status":     "filled",
                        "order_id":   od["order_id"],
                        "fill_price": fill_px,
                        "strike":     od["strike"],
                        "expiry":     od["expiry"],
                        "delta":      od["delta"],
                    }
                elif state in ("canceled","expired","rejected"):
                    print(f"  ❌ {key}: {state}")
                    results[key] = {"status": state}
                else:
                    all_done = False
            if all_done:
                break
        # Any still pending
        for key, od in pending_orders.items():
            if key not in results:
                results[key] = {"status": "working",
                                "order_id": od["order_id"],
                                "strike": od["strike"]}

    # Legacy single-order loop no longer needed — skip to summary
    # Summary
    print(f"\n{'='*60}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*60}")
    for key, r in results.items():
        status = r.get("status", "?")
        if status == "filled":
            print(f"  {key}: ✅ Filled ${r['strike']:.0f}C @ ${r['fill_price']:.2f}")
        elif status == "preview":
            print(f"  {key}: 👁  Preview ${r['strike']:.0f}C @ ${r['mid']:.2f} "
                  f"(δ={r['delta']:.3f})")
        elif status == "no_strike":
            print(f"  {key}: ⚠️  No valid strike found")
        elif status == "dry_run":
            print(f"  {key}: 🔵 Dry run @ ${r['price']:.2f}")
        else:
            print(f"  {key}: ❌ {status}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview",  action="store_true",
                        help="Show what would be ordered — no execution")
    parser.add_argument("--paper",    action="store_true",
                        help="Execute on Tradier sandbox")
    parser.add_argument("--live",     action="store_true",
                        help="Execute on live account (future)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to run e.g. --variants V4 V5")
    args = parser.parse_args()

    if args.live:
        print("⚠️  Live mode not yet enabled — use --paper for sandbox")
        return

    sandbox = not args.live
    run(sandbox=sandbox,
        preview=args.preview or (not args.paper),
        variant_filter=args.variants)


if __name__ == "__main__":
    main()
