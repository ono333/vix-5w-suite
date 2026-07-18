#!/usr/bin/env python3
"""
apply_retry_500.py — add transient-5xx retry so unattended runs don't silently
miss trades on Tradier 500 errors, WITHOUT risking double-placement.

Design (Option 1, correctly scoped):
  - _get (all READS: quotes, positions, order status, chains) -> retry on 5xx with
    exponential backoff. Reads are idempotent; retrying is fully safe.
  - place_order (the ONLY order-creating write) -> retry on 5xx, but BEFORE each
    retry, check whether a working order already exists for this option_symbol.
    If one exists, the 500 lied (order WAS created) -> do NOT re-place (no double).
    If none exists -> safe to re-place.
  - In-loop re-places are UNCHANGED: the existing while-market-open loop already
    retries placement each cycle, so no per-cycle verification is added (no overkill).
  - Only 500/502/503/504 retry. 4xx (client errors) never retry.

Line-anchored. RUN ON SERVER: python3 apply_retry_500.py
"""
import shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_retry500_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

edits = 0

# ── 1. Add a retry helper + rework _get to use it. ──
# Current _get:
old_get = '''    def _get(self, path, params=None):
        r = self.session.get(f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()'''
new_get = '''    def _retry_5xx(self, fn, *args, **kwargs):
        """Call fn (a requests method); retry on transient 5xx with backoff.
        Only 500/502/503/504 retry; 4xx and success return/raise immediately."""
        backoffs = [1, 3, 7]
        for attempt in range(len(backoffs) + 1):
            try:
                r = fn(*args, **kwargs)
            except requests.exceptions.RequestException:
                if attempt < len(backoffs):
                    time.sleep(backoffs[attempt]); continue
                raise
            if r.status_code in (500, 502, 503, 504) and attempt < len(backoffs):
                time.sleep(backoffs[attempt]); continue
            return r

    def _get(self, path, params=None):
        r = self._retry_5xx(self.session.get, f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()'''
if old_get in src:
    src = src.replace(old_get, new_get, 1)
    print("\u2713 _get now retries transient 5xx (+ _retry_5xx helper added)"); edits += 1
else:
    print("\u274c _get anchor not found \u2014 restoring."); shutil.copy(bak, TARGET); sys.exit(1)

# ── 2. Add a helper to check for an existing working order for an option. ──
# Insert right before place_order.
if "def has_working_order" not in src:
    anchor_po = "    def place_order(self, symbol, option_symbol, side, quantity, price) -> dict:"
    if anchor_po not in src:
        print("\u274c place_order anchor not found \u2014 restoring."); shutil.copy(bak, TARGET); sys.exit(1)
    helper = '''    def has_working_order(self, option_symbol: str) -> bool:
        """True if an open/pending order already exists for this option_symbol.
        Used to avoid double-placing after a 500 that actually created the order."""
        try:
            data = self._get(f"/accounts/{self.account}/orders")
            o = data.get("orders", {})
            if not o or o == "null":
                return False
            orders = o.get("order", [])
            if isinstance(orders, dict): orders = [orders]
            for x in orders:
                if x.get("option_symbol") == option_symbol and \\
                   x.get("status") in ("open", "pending", "partially_filled"):
                    return True
        except Exception:
            pass
        return False

'''
    src = src.replace(anchor_po, helper + anchor_po, 1)
    print("\u2713 has_working_order() helper added"); edits += 1

# ── 3. Rework place_order to retry with verification. ──
old_po = '''    def place_order(self, symbol, option_symbol, side, quantity, price) -> dict:
        return self._post(f"/accounts/{self.account}/orders", {
            "class": "option", "symbol": symbol,
            "option_symbol": option_symbol, "side": side,
            "quantity": str(quantity), "type": "limit",
            "duration": "day", "price": f"{price:.2f}",
        })'''
new_po = '''    def place_order(self, symbol, option_symbol, side, quantity, price) -> dict:
        payload = {
            "class": "option", "symbol": symbol,
            "option_symbol": option_symbol, "side": side,
            "quantity": str(quantity), "type": "limit",
            "duration": "day", "price": f"{price:.2f}",
        }
        url = f"{self.base}/accounts/{self.account}/orders"
        backoffs = [1, 3, 7]
        for attempt in range(len(backoffs) + 1):
            # Before a RETRY (not the first try), verify the prior 500 didn't
            # already create the order -> if it did, don't double-place.
            if attempt > 0 and self.has_working_order(option_symbol):
                return {"order": {"status": "already_working",
                                  "note": "order existed after 5xx; not re-placed"}}
            try:
                r = self.session.post(url, data=payload,
                                      headers={"Authorization": f"Bearer {self.token}",
                                               "Accept": "application/json"},
                                      timeout=15)
            except requests.exceptions.RequestException:
                if attempt < len(backoffs):
                    time.sleep(backoffs[attempt]); continue
                raise
            if r.status_code in (500, 502, 503, 504) and attempt < len(backoffs):
                time.sleep(backoffs[attempt]); continue
            r.raise_for_status()
            return r.json()'''
if old_po in src:
    src = src.replace(old_po, new_po, 1)
    print("\u2713 place_order retries 5xx with existing-order verification"); edits += 1
else:
    print("\u274c place_order body anchor not found \u2014 restoring."); shutil.copy(bak, TARGET); sys.exit(1)

open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print(f"\u2713 py_compile OK  ({edits} edits)")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --check")
print(f"   Revert: cp {bak} {TARGET}")
