#!/usr/bin/env python3
"""
tradier_orchestrator.py
───────────────────────
Fully autonomous daily execution loop for UVXY diagonal spreads.
Runs without human intervention — monitors, decides, executes, logs.

Decision loop per variant:
  1. Sync state from Tradier API (source of truth)
  2. Long leg check:
     - Missing → BTO per variant parameters
     - DTE < threshold → roll (BTO new first, STC old after fill)
     - Phase transition detected → swap long leg
  3. Short leg check (Friday/Monday only):
     - Missing + long confirmed → STO
     - DTE ≤ 2 + OTM → let expire, queue for Monday
     - Delta ≥ ceiling → emergency roll
  4. Log all fills to tradier_long_state.json
  5. Send confirmation email on any action

Usage:
    python3 tradier_orchestrator.py --check     # status only, no orders
    python3 tradier_orchestrator.py --preview   # show decisions, no orders
    python3 tradier_orchestrator.py --paper     # execute on sandbox
    python3 tradier_orchestrator.py --live      # execute on live (future)

Designed to run via systemd timer at 10:05am ET Mon/Fri,
with retry logic until Yahoo data is available.
"""

from __future__ import annotations
import argparse
import json
import os
import smtplib
import sys
import time
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from tradier_liquidity import check_liquidity, log_skip
from tradier_exec_log import log_fill

# ── Paths ─────────────────────────────────────────────────────────────────────
STORAGE_DIR        = Path.home() / ".vix_suite"
VOL_SNAPSHOTS_PATH = STORAGE_DIR / "vol_snapshots.json"
SIGNAL_BATCH_PATH  = STORAGE_DIR / "current_signal_batch.json"
LONG_STATE_PATH    = STORAGE_DIR / "tradier_long_state.json"
ORCH_LOG_PATH      = STORAGE_DIR / "orchestrator.log"

# ── Roll thresholds ───────────────────────────────────────────────────────────
ROLL_THRESHOLD = {"V1": 30, "V2": 21, "V3": 14, "V4": 21, "V5": 21}
LONG_DELTA_MIN = 0.65
LONG_DELTA_MAX = 0.75
SHORT_EMERGENCY_DELTA = 0.50
REPRICE_SEC    = 900
MAX_REPRICE    = 20
NUDGE_BUY      = 0.05
NUDGE_SELL     = 0.01
STO_FLOOR_DROP = 0.20  # max drop from mid before giving up


# ── Logger ────────────────────────────────────────────────────────────────────

class OrchestratorLog:
    def __init__(self):
        self.entries = []

    def log(self, msg: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.entries.append(line)
        # Append to log file
        with open(ORCH_LOG_PATH, "a") as f:
            f.write(f"[{date.today()}] {line}\n")

    def summary(self) -> str:
        return "\n".join(self.entries)


LOG = OrchestratorLog()


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


def load_state() -> dict:
    if not LONG_STATE_PATH.exists():
        return {"updated_at": str(date.today()), "variants": {}}
    try:
        return json.loads(LONG_STATE_PATH.read_text())
    except Exception:
        return {"updated_at": str(date.today()), "variants": {}}


def save_state(state: dict):
    state["updated_at"] = str(date.today())
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


def delta_ceil(v: dict) -> float:
    sm = v.get("sigma_mult", 1.0)
    if sm <= 0.8: return 0.28
    if sm <= 0.9: return 0.25
    if sm <= 1.0: return 0.22
    if sm <= 1.2: return 0.18
    return 0.15


# ── Tradier client ────────────────────────────────────────────────────────────

import requests

class TradierClient:
    def __init__(self, sandbox: bool = True):
        self.sandbox = sandbox
        self.base    = ("https://sandbox.tradier.com/v1" if sandbox
                        else "https://api.tradier.com/v1")
        self.token   = (os.environ.get("TRADIER_SANDBOX_TOKEN","") if sandbox
                        else os.environ.get("TRADIER_LIVE_TOKEN",""))
        self.account = os.environ.get("TRADIER_ACCOUNT","VA88395338")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept":        "application/json",
        })

    def _get(self, path, params=None):
        r = self.session.get(f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path, data):
        r = self.session.post(
            f"{self.base}{path}", data=data, timeout=15,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> list[dict]:
        data = self._get(f"/accounts/{self.account}/positions")
        pos  = data.get("positions", {})
        if not pos or pos == "null": return []
        p = pos.get("position", [])
        return p if isinstance(p, list) else [p]

    def get_option_chain(self, symbol: str, expiry: date) -> list[dict]:
        data = self._get("/markets/options/chains", {
            "symbol": symbol,
            "expiration": expiry.strftime("%Y-%m-%d"),
            "greeks": "true",
        })
        opts = data.get("options", {})
        if not opts: return []
        chain = opts.get("option", [])
        return chain if isinstance(chain, list) else [chain]

    def get_expirations(self, symbol: str) -> list[date]:
        data = self._get("/markets/options/expirations", {
            "symbol": symbol, "includeAllRoots": "true", "strikes": "false",
        })
        dates = data.get("expirations", {}).get("date", [])
        if isinstance(dates, str): dates = [dates]
        return [date.fromisoformat(d) for d in dates]

    def get_quote(self, symbol: str) -> dict:
        data = self._get("/markets/quotes", {"symbols": symbol, "greeks": "false"})
        q = data.get("quotes", {}).get("quote", {})
        return q if isinstance(q, dict) else {}

    def place_order(self, symbol, option_symbol, side, quantity, price) -> dict:
        return self._post(f"/accounts/{self.account}/orders", {
            "class": "option", "symbol": symbol,
            "option_symbol": option_symbol, "side": side,
            "quantity": str(quantity), "type": "limit",
            "duration": "gtc", "price": f"{price:.2f}",
        })

    def cancel_order(self, order_id: int) -> dict:
        r = self.session.delete(
            f"{self.base}/accounts/{self.account}/orders/{order_id}",
            timeout=15)
        r.raise_for_status()
        return r.json()

    def get_order(self, order_id: int) -> dict:
        data = self._get(f"/accounts/{self.account}/orders/{order_id}")
        return data.get("order", {})

    def modify_order(self, order_id: int, new_price: float) -> dict:
        return self._post(f"/accounts/{self.account}/orders/{order_id}", {
            "type": "limit", "duration": "gtc",
            "price": f"{new_price:.2f}",
        })


# ── Position parser ───────────────────────────────────────────────────────────

def parse_positions(positions: list[dict]) -> tuple[list, list]:
    """Parse Tradier positions into (longs, shorts)."""
    today  = date.today()
    longs, shorts = [], []
    for pos in positions:
        sym = pos.get("symbol", "")
        qty = float(pos.get("quantity", 0))
        if "UVXY" not in sym or qty == 0: continue
        try:
            idx      = sym.index("UVXY") + 4
            exp_str  = sym[idx:idx+6]
            exp_date = date(2000+int(exp_str[:2]),int(exp_str[2:4]),int(exp_str[4:6]))
            dte      = (exp_date - today).days
            strike   = int(sym[idx+7:]) / 1000
            cp       = sym[idx+6]
            if cp != "C": continue
            cost     = float(pos.get("cost_basis", 0))
            fill_px  = abs(cost) / abs(qty) / 100
            entry = {"symbol": sym, "strike": strike, "expiry": exp_date,
                     "dte": dte, "quantity": int(abs(qty)),
                     "fill_price": round(fill_px, 2)}
            if qty > 0: longs.append(entry)
            else: shorts.append(entry)
        except Exception:
            continue
    return longs, shorts


# ── Expiry finder ─────────────────────────────────────────────────────────────

def next_friday(expirations: list[date], min_dte: int = 7) -> Optional[date]:
    today  = date.today()
    target = today + timedelta(days=min_dte)
    valid  = [e for e in expirations if e >= target and e.weekday() == 4]
    return valid[0] if valid else None


def target_long_expiry(expirations: list[date], long_dte_weeks: int) -> Optional[date]:
    today  = date.today()
    target = today + timedelta(weeks=long_dte_weeks)
    valid  = [e for e in expirations if e >= target]
    return valid[0] if valid else None


# ── Strike finders ────────────────────────────────────────────────────────────

def find_long_strike(chain: list[dict]) -> Optional[dict]:
    candidates = []
    for c in chain:
        if c.get("option_type","").lower() != "call": continue
        greeks = c.get("greeks") or {}
        delta  = float(greeks.get("delta", 0) or 0)
        bid    = float(c.get("bid", 0) or 0)
        ask    = float(c.get("ask", 0) or 0)
        if LONG_DELTA_MIN <= delta <= LONG_DELTA_MAX and ask > 0:
            candidates.append({
                "symbol": c.get("symbol"), "strike": float(c.get("strike",0)),
                "delta": delta, "bid": bid, "ask": ask,
                "mid": round((bid+ask)/2, 2),
            })
    if not candidates: return None
    candidates.sort(key=lambda x: abs(x["delta"] - 0.70))
    return candidates[0]


def find_short_strike(chain: list[dict], dc: float,
                      min_credit: float = 0.25) -> Optional[dict]:
    candidates = []
    for c in chain:
        if c.get("option_type","").lower() != "call": continue
        greeks = c.get("greeks") or {}
        delta  = abs(float(greeks.get("delta", 1.0) or 1.0))
        bid    = float(c.get("bid", 0) or 0)
        ask    = float(c.get("ask", 0) or 0)
        if delta <= dc and bid >= min_credit:
            candidates.append({
                "symbol": c.get("symbol"), "strike": float(c.get("strike",0)),
                "delta": delta, "bid": bid, "ask": ask,
                "mid": round((bid+ask)/2, 2),
            })
    if not candidates: return None
    candidates.sort(key=lambda x: x["delta"], reverse=True)
    return candidates[0]


# ── Order executor ────────────────────────────────────────────────────────────

def place_with_reprice(client: TradierClient, underlying: str,
                       option_symbol: str, side: str,
                       quantity: int, bid: float, ask: float) -> dict:
    """
    Place order at mid, reprice toward fill price each interval.
    BTO: nudge toward ask. STO: nudge toward bid.
    Sandbox: cancel + re-place. Live: modify.
    """
    is_buy  = "buy" in side.lower()
    mid     = round((bid + ask) / 2, 2)
    price   = mid
    limit   = ask if is_buy else bid
    nudge   = NUDGE_BUY if is_buy else NUDGE_SELL

    if not is_buy and bid <= 0.05:
        LOG.log(f"   ⚠️ Bid ${bid:.2f} too low — skipping")
        return {"status": "skipped_low_bid"}

    LOG.log(f"   {'BTO' if is_buy else 'STO'} {option_symbol} ×{quantity} "
            f"@ ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")

    try:
        result = client.place_order(underlying, option_symbol, side, quantity, price)
        oid    = result.get("order", {}).get("id")
        if not oid:
            LOG.log(f"   ❌ Place failed: {result}")
            return {"status": "failed"}
        LOG.log(f"   Order {oid} placed")
    except Exception as e:
        LOG.log(f"   ❌ Place error: {e}")
        return {"status": "failed"}

    for attempt in range(MAX_REPRICE):
        time.sleep(REPRICE_SEC)
        try:
            status = client.get_order(oid)
            state  = status.get("status", "")
        except Exception:
            state = "unknown"

        LOG.log(f"   [{attempt+1}] status={state} @ ${price:.2f}")

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            LOG.log(f"   ✅ Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid,
                    "fill_price": fill_px, "quantity": quantity}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # Nudge toward fill
        if is_buy:
            new_price = min(round(price + nudge, 2), limit)
        else:
            # STO: floor = mid - STO_FLOOR_DROP
            sto_floor = round(mid - STO_FLOOR_DROP, 2)
            new_price = max(round(price - nudge, 2), sto_floor)
            if new_price <= sto_floor and new_price == price:
                LOG.log(f"   ⛔ Floor ${sto_floor:.2f} reached — giving up")
                try: client.cancel_order(oid)
                except Exception: pass
                return {"status": "floor_reached", "order_id": oid, "floor": sto_floor}

        if new_price == price:
            LOG.log(f"   At limit ${price:.2f} — waiting")
            continue

        LOG.log(f"   Repricing ${price:.2f} → ${new_price:.2f}")

        if client.sandbox:
            # Cancel + re-place
            try: client.cancel_order(oid)
            except Exception: pass
            try:
                result  = client.place_order(underlying, option_symbol,
                                             side, quantity, new_price)
                new_oid = result.get("order", {}).get("id")
                if new_oid:
                    oid = new_oid
                    price = new_price
            except Exception as e:
                LOG.log(f"   ⚠️ Re-place failed: {e}")
        else:
            try:
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as e:
                LOG.log(f"   ⚠️ Modify failed: {e}")

    LOG.log(f"   ⚠️ Max attempts — order {oid} working @ ${price:.2f}")
    return {"status": "working", "order_id": oid, "last_price": price}


# ── State sync from Tradier ───────────────────────────────────────────────────

def sync_state_from_tradier(client: TradierClient) -> dict:
    """
    Rebuild tradier_long_state.json from actual Tradier positions.
    Source of truth is always the broker, not the local file.
    """
    positions = client.get_positions()
    longs, shorts = parse_positions(positions)
    today = date.today()

    LOG.log(f"   Synced {len(longs)} longs, {len(shorts)} shorts from Tradier")

    state = load_state()
    variants = state.get("variants", {
        k: {"long": None, "short": None} for k in ["V1","V2","V3","V4","V5"]
    })

    # Map longs to variants by expiry
    for lg in longs:
        exp_str = str(lg["expiry"])
        if "2026-12" in exp_str:
            variants.setdefault("V1", {})["long"] = lg
        elif "2026-06" in exp_str:
            variants.setdefault("V3", {})["long"] = lg
        elif "2026-07" in exp_str or "2026-08" in exp_str:
            for key in ["V2","V4","V5"]:
                variants.setdefault(key, {})["long"] = dict(lg, quantity=1)

    # Map shorts to variants by strike
    strike_to_variant = {
        43.0: "V1", 44.5: ["V2","V5"], 46.0: "V3", 48.0: "V4"
    }
    for sh in shorts:
        mapping = strike_to_variant.get(sh["strike"])
        if not mapping: continue
        if isinstance(mapping, list):
            for k in mapping:
                variants.setdefault(k, {})["short"] = sh
        else:
            variants.setdefault(mapping, {})["short"] = sh

    # Clear expired shorts
    for key, v in variants.items():
        sh = v.get("short")
        if sh:
            exp = sh.get("expiry")
            if isinstance(exp, str):
                exp = date.fromisoformat(exp)
            if isinstance(exp, date) and exp < today:
                v["short"] = None
                LOG.log(f"   {key}: short expired — cleared")

    state["variants"] = variants
    save_state(state)
    return state


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run(sandbox: bool = True, preview: bool = False, check_only: bool = False):
    today   = date.today()
    weekday = today.weekday()  # 0=Mon, 4=Fri

    LOG.log(f"{'='*60}")
    LOG.log(f"Tradier Orchestrator — {today} ({'SANDBOX' if sandbox else 'LIVE'})")
    mode_str = "CHECK" if check_only else "PREVIEW" if preview else "EXECUTE"
    LOG.log(f"Mode: {mode_str} | Weekday: {today.strftime('%A')}")
    LOG.log(f"{'='*60}")

    # Load app data
    snap  = load_snap()
    batch = load_batch()

    if not snap:
        LOG.log("❌ No snapshot — waiting for Yahoo data")
        return {"status": "no_snapshot"}

    uvxy   = snap.get("uvxy", 0.0)
    phase  = snap.get("spike_label", "")
    regime = snap.get("regime", "")
    collapse = snap.get("collapse_flag", False)
    LOG.log(f"UVXY: ${uvxy:.2f} | Phase: {phase} | Regime: {regime} | Collapse: {collapse}")

    # Check snapshot freshness
    try:
        snap_time = datetime.fromisoformat(snap.get("captured_at",""))
        age_min   = (datetime.now() - snap_time).seconds / 60
        if age_min > 60:
            LOG.log(f"⚠️ Snapshot is {age_min:.0f} min old — data may be stale")
    except Exception:
        pass

    client = TradierClient(sandbox=sandbox)
    role_map = build_role_map(batch)

    # ── Sync state from Tradier ────────────────────────────────────────────
    LOG.log("\nSyncing state from Tradier...")
    try:
        state = sync_state_from_tradier(client)
    except Exception as e:
        LOG.log(f"❌ Sync failed: {e}")
        state = load_state()

    variants = state.get("variants", {})

    # ── Get expirations ────────────────────────────────────────────────────
    try:
        expirations = client.get_expirations("UVXY")
        LOG.log(f"Expirations available: {[e.strftime('%b %d') for e in expirations[:6]]}")
    except Exception as e:
        LOG.log(f"❌ Cannot fetch expirations: {e}")
        return {"status": "no_expirations"}

    actions_taken = []
    is_short_day  = weekday in (0, 4)  # Monday or Friday

    LOG.log(f"\n{'─'*60}")
    LOG.log(f"Short entry day: {'YES' if is_short_day else 'NO (hold)'}")
    LOG.log(f"{'─'*60}")

    # ── Process each variant ───────────────────────────────────────────────
    for key in ["V1","V2","V3","V4","V5"]:
        v = role_map.get(key)
        if not v:
            LOG.log(f"\n{key}: not in signal batch — skip")
            continue

        long_dte_weeks = v.get("long_dte_weeks", 13)
        roll_threshold = ROLL_THRESHOLD.get(key, 21)
        dc             = delta_ceil(v)
        variant_state  = variants.get(key, {"long": None, "short": None})
        long_pos       = variant_state.get("long")
        short_pos      = variant_state.get("short")

        LOG.log(f"\n{key} — long_dte={long_dte_weeks}w roll<{roll_threshold}d δ≤{dc:.2f}")

        # ── Long leg decision ──────────────────────────────────────────────
        if not long_pos:
            LOG.log(f"  ❌ No long leg — needs entry")
            if not check_only:
                # ── DTE step-down: try target DTE first, step down if illiquid ──
                # Step down: target → target-4w → target-8w → min 4w
                _dte_candidates = []
                _base = long_dte_weeks
                for _step in [0, 4, 8]:
                    _w = max(_base - _step, 4)
                    _e = target_long_expiry(expirations, _w)
                    if _e and _e not in _dte_candidates:
                        _dte_candidates.append(_e)

                exp  = None
                best = None
                for _cand_exp in _dte_candidates:
                    _dte_days = (_cand_exp - date.today()).days
                    LOG.log(f"  Trying {_cand_exp} ({_dte_days}d)...")
                    try:
                        _chain = client.get_option_chain("UVXY", _cand_exp)
                        _best  = find_long_strike(_chain)
                        if not _best:
                            LOG.log(f"  ⚠️ No strike in delta range")
                            continue
                        _liq_ok, _liq_reason = check_liquidity(_best["bid"], _best["ask"])
                        if _liq_ok:
                            exp  = _cand_exp
                            best = _best
                            if _cand_exp != _dte_candidates[0]:
                                LOG.log(f"  ℹ️ DTE adjusted: target was "
                                        f"{_dte_candidates[0]}, using {_cand_exp} "
                                        f"(better liquidity)")
                            break
                        else:
                            LOG.log(f"  ⚠️ {_cand_exp} illiquid: {_liq_reason} "
                                    f"(bid=${_best['bid']:.2f} ask=${_best['ask']:.2f})")
                            log_skip(key, _best["strike"], str(_cand_exp),
                                     _best["bid"], _best["ask"], _liq_reason)
                    except Exception as _e:
                        LOG.log(f"  ❌ Chain error for {_cand_exp}: {_e}")

                if not exp or not best:
                    LOG.log(f"  ❌ No liquid expiry found after step-down")
                    continue

                try:
                    LOG.log(f"  BTO ${best['strike']:.0f}C {exp} "
                            f"δ={best['delta']:.3f} ask=${best['ask']:.2f}")
                    liq_ok, liq_reason = True, None  # already checked above
                    if not preview:
                        result = place_with_reprice(
                            client, "UVXY", best["symbol"],
                            "buy_to_open", 1,
                            best["bid"], best["ask"])
                        if result["status"] == "filled":
                            log_fill(key, best["strike"], str(exp), "BTO",
                                     best["mid"], result["fill_price"], 1,
                                     str(result.get("order_id", "")))
                            variant_state["long"] = {
                                "symbol":     best["symbol"],
                                "strike":     best["strike"],
                                "expiry":     str(exp),
                                "dte":        (exp - today).days,
                                "quantity":   1,
                                "fill_price": result["fill_price"],
                            }
                            variants[key] = variant_state
                            save_state({**state, "variants": variants})
                            actions_taken.append(
                                f"{key}: BTO ${best['strike']:.0f}C {exp} "
                                f"@ ${result['fill_price']:.2f}")
                except Exception as e:
                    LOG.log(f"  ❌ Long entry error: {e}")

        else:
            long_exp = long_pos.get("expiry")
            if isinstance(long_exp, str):
                long_exp = date.fromisoformat(long_exp)
            long_dte = (long_exp - today).days if long_exp else 999
            LOG.log(f"  ✅ Long: ${long_pos['strike']:.0f}C exp {long_exp} "
                    f"(DTE={long_dte})")

            # Roll if DTE < threshold
            if long_dte <= roll_threshold:
                LOG.log(f"  ⚠️ DTE {long_dte} ≤ {roll_threshold} — ROLL NEEDED")
                if not check_only:
                    new_exp = target_long_expiry(expirations, long_dte_weeks)
                    if new_exp and new_exp != long_exp:
                        try:
                            chain    = client.get_option_chain("UVXY", new_exp)
                            best_new = find_long_strike(chain)
                            if best_new:
                                LOG.log(f"  Roll to ${best_new['strike']:.0f}C {new_exp}")
                                liq_ok, liq_reason = check_liquidity(
                                    best_new["bid"], best_new["ask"])
                                if not liq_ok:
                                    log_skip(key, best_new["strike"], str(new_exp),
                                             best_new["bid"], best_new["ask"], liq_reason)
                                    LOG.log(f"  ⚠️ Roll liquidity check failed: {liq_reason}")
                                    continue
                                if not preview:
                                    # BTO new long first
                                    r_buy = place_with_reprice(
                                        client, "UVXY", best_new["symbol"],
                                        "buy_to_open", long_pos["quantity"],
                                        best_new["bid"], best_new["ask"])
                                    if r_buy["status"] == "filled":
                                        log_fill(key, best_new["strike"], str(new_exp), "BTO",
                                                 best_new["mid"], r_buy["fill_price"],
                                                 long_pos["quantity"],
                                                 str(r_buy.get("order_id", "")))
                                        # STC old long
                                        old_quote = client.get_quote(long_pos["symbol"])
                                        old_bid   = float(old_quote.get("bid", 0.10))
                                        old_ask   = float(old_quote.get("ask", 0.20))
                                        r_sell = place_with_reprice(
                                            client, "UVXY", long_pos["symbol"],
                                            "sell_to_close", long_pos["quantity"],
                                            old_bid, old_ask)
                                        variant_state["long"] = {
                                            "symbol":     best_new["symbol"],
                                            "strike":     best_new["strike"],
                                            "expiry":     str(new_exp),
                                            "dte":        (new_exp-today).days,
                                            "quantity":   long_pos["quantity"],
                                            "fill_price": r_buy["fill_price"],
                                        }
                                        variants[key] = variant_state
                                        save_state({**state, "variants": variants})
                                        actions_taken.append(
                                            f"{key}: Rolled long to "
                                            f"${best_new['strike']:.0f}C {new_exp}")
                        except Exception as e:
                            LOG.log(f"  ❌ Roll error: {e}")

        # ── Short leg decision ─────────────────────────────────────────────
        long_confirmed = variant_state.get("long") is not None

        if not long_confirmed:
            LOG.log(f"  ⚠️ No long confirmed — cannot sell short")
            continue

        long_strike = variant_state["long"]["strike"]

        if not short_pos:
            if is_short_day:
                LOG.log(f"  📋 No short — entering today (short day)")
                if not check_only:
                    sh_exp = next_friday(expirations, min_dte=7)
                    if not sh_exp:
                        LOG.log(f"  ❌ No short expiry found")
                        continue
                    try:
                        chain = client.get_option_chain("UVXY", sh_exp)
                        best  = find_short_strike(chain, dc)
                        if not best:
                            LOG.log(f"  ⚠️ No short strike found for δ≤{dc:.2f}")
                            continue
                        if best["strike"] <= long_strike:
                            LOG.log(f"  ❌ Short ${best['strike']:.0f}C ≤ "
                                    f"long ${long_strike:.0f}C — blocked")
                            continue
                        LOG.log(f"  STO ${best['strike']:.0f}C {sh_exp} "
                                f"δ={best['delta']:.3f} bid=${best['bid']:.2f}")
                        liq_ok, liq_reason = check_liquidity(best["bid"], best["ask"])
                        if not liq_ok:
                            log_skip(key, best["strike"], str(sh_exp),
                                     best["bid"], best["ask"], liq_reason)
                            LOG.log(f"  ⚠️ Liquidity check failed: {liq_reason}")
                            continue
                        if not preview:
                            result = place_with_reprice(
                                client, "UVXY", best["symbol"],
                                "sell_to_open", 1,
                                best["bid"], best["ask"])
                            if result["status"] == "filled":
                                log_fill(key, best["strike"], str(sh_exp), "STO",
                                         best["mid"], result["fill_price"], 1,
                                         str(result.get("order_id", "")))
                                variant_state["short"] = {
                                    "symbol":     best["symbol"],
                                    "strike":     best["strike"],
                                    "expiry":     str(sh_exp),
                                    "dte":        (sh_exp-today).days,
                                    "quantity":   1,
                                    "fill_price": result["fill_price"],
                                }
                                variants[key] = variant_state
                                save_state({**state, "variants": variants})
                                actions_taken.append(
                                    f"{key}: STO ${best['strike']:.0f}C {sh_exp} "
                                    f"@ ${result['fill_price']:.2f}")
                    except Exception as e:
                        LOG.log(f"  ❌ Short entry error: {e}")
            else:
                LOG.log(f"  ℹ️ No short — not a short entry day")

        else:
            short_exp = short_pos.get("expiry")
            if isinstance(short_exp, str):
                short_exp = date.fromisoformat(short_exp)
            short_dte = (short_exp - today).days if short_exp else 0
            LOG.log(f"  ✅ Short: ${short_pos['strike']:.0f}C exp {short_exp} "
                    f"(DTE={short_dte})")

            # Emergency roll check — get live delta
            if short_dte > 0:
                try:
                    chain = client.get_option_chain("UVXY", short_exp)
                    live  = next((c for c in chain
                                  if float(c.get("strike",0)) == short_pos["strike"]
                                  and c.get("option_type","").lower() == "call"), None)
                    if live:
                        greeks = live.get("greeks") or {}
                        live_delta = abs(float(greeks.get("delta", 0) or 0))
                        LOG.log(f"  Live delta: {live_delta:.3f} (ceiling: {dc:.2f})")
                        if live_delta >= SHORT_EMERGENCY_DELTA:
                            LOG.log(f"  🚨 EMERGENCY ROLL — delta {live_delta:.3f} ≥ 0.50")
                            actions_taken.append(
                                f"{key}: ⚠️ Emergency roll needed "
                                f"(delta={live_delta:.3f})")
                except Exception:
                    pass

            # Let expire if DTE 1-2
            if short_dte > 0 and short_dte <= 2:
                LOG.log(f"  ⏳ DTE={short_dte} — letting expire Friday")
                continue

            # DTE=0 — expiry day, enter new short now
            if short_dte == 0:
                LOG.log(f"  ♻️  DTE=0 — expiry day, entering new short")
                if is_short_day and not check_only:
                    try:
                        sh_exp = next_friday(expirations, min_dte=7)
                        if not sh_exp:
                            LOG.log(f"  ❌ No short expiry found")
                        else:
                            chain = client.get_option_chain("UVXY", sh_exp)
                            best  = find_short_strike(chain, dc)
                            if not best:
                                LOG.log(f"  ⚠️ No short strike found in delta range")
                            else:
                                liq_ok, liq_reason = check_liquidity(best["bid"], best["ask"])
                                if not liq_ok:
                                    log_skip(key, best["strike"], str(sh_exp),
                                             best["bid"], best["ask"], liq_reason)
                                    LOG.log(f"  ⚠️ Liquidity check failed: {liq_reason}")
                                else:
                                    LOG.log(f"  STO ${best['strike']:.0f}C {sh_exp} "
                                            f"δ={best['delta']:.3f} bid=${best['bid']:.2f}")
                                    if not preview:
                                        result = place_with_reprice(
                                            client, "UVXY", best["symbol"],
                                            "sell_to_open", 1,
                                            best["bid"], best["ask"])
                                        if result["status"] == "filled":
                                            log_fill(key, best["strike"], str(sh_exp), "STO",
                                                     best["mid"], result["fill_price"], 1,
                                                     str(result.get("order_id", "")))
                                            variant_state["short"] = {
                                                "symbol":     best["symbol"],
                                                "strike":     best["strike"],
                                                "expiry":     str(sh_exp),
                                                "quantity":   1,
                                                "fill_price": result["fill_price"],
                                            }
                                            variants[key] = variant_state
                                            save_state({**state, "variants": variants})
                                            actions_taken.append(
                                                f"{key}: STO ${best['strike']:.0f}C {sh_exp} "
                                                f"@ ${result['fill_price']:.2f}")
                                        elif result["status"] == "floor_reached":
                                            LOG.log(f"  ⛔ {key}: floor reached — skipping")
                    except Exception as e:
                        LOG.log(f"  ❌ Short entry error: {e}")
                elif preview:
                    LOG.log(f"  [PREVIEW] Would enter new short for {key}")

    # ── Summary ────────────────────────────────────────────────────────────
    LOG.log(f"\n{'='*60}")
    LOG.log(f"ORCHESTRATOR COMPLETE — {len(actions_taken)} action(s)")
    for a in actions_taken:
        LOG.log(f"  • {a}")
    LOG.log(f"{'='*60}")

    # Send confirmation email if actions taken
    if actions_taken and not preview and not check_only:
        _send_confirmation(actions_taken, uvxy, phase)

    return {"status": "ok", "actions": actions_taken}


# ── Confirmation email ────────────────────────────────────────────────────────

def _send_confirmation(actions: list[str], uvxy: float, phase: str):
    try:
        smtp_user = os.environ.get("SMTP_USER","")
        smtp_pass = os.environ.get("SMTP_PASS","")
        if not smtp_user or not smtp_pass:
            return

        subject = (f"[Tradier Auto] {len(actions)} action(s) executed "
                   f"— UVXY ${uvxy:.2f} · {phase}")
        body    = "<br>".join([f"• {a}" for a in actions])
        html    = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                    padding:20px;">
          <h3 style="color:#1E4D2B;">🤖 Tradier Auto Execution Report</h3>
          <p>UVXY ${uvxy:.2f} · Phase: {phase} · {date.today()}</p>
          <div style="background:#f0f7f2;border-left:4px solid #1E4D2B;
                      padding:12px;border-radius:4px;">
            {body}
          </div>
          <p style="font-size:11px;color:#999;margin-top:20px;">
            Auto-generated by Tradier Orchestrator · Sandbox mode
          </p>
        </div>"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = smtp_user
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, smtp_user, msg.as_string())
        LOG.log("📧 Confirmation email sent")
    except Exception as e:
        LOG.log(f"⚠️ Email failed: {e}")


# ── Live delta check (Layer 6) ────────────────────────────────────────────────

def check_position_deltas(sandbox: bool = True) -> list[dict]:
    """
    Check live deltas for all open short legs via Tradier chain.
    Sends alert email if any delta >= 0.35.
    """
    today = date.today()
    if today.weekday() >= 5:
        return []

    client   = TradierClient(sandbox=sandbox)
    state    = load_state()
    variants = state.get("variants", {})
    batch    = load_batch()
    role_map = build_role_map(batch)

    LOG.log("\n── check_position_deltas ──")
    flagged = []

    for key in ["V1", "V2", "V3", "V4", "V5"]:
        v_state = variants.get(key, {})
        short   = v_state.get("short")
        if not short:
            continue

        short_exp = short.get("expiry")
        if isinstance(short_exp, str):
            try:
                short_exp = date.fromisoformat(short_exp)
            except Exception:
                continue
        if not short_exp or short_exp < today:
            continue

        short_strike = float(short.get("strike", 0))
        v  = role_map.get(key, {})
        dc = delta_ceil(v) if v else 0.22

        try:
            chain    = client.get_option_chain("UVXY", short_exp)
            live_opt = next(
                (c for c in chain
                 if abs(float(c.get("strike", 0)) - short_strike) < 0.01
                 and c.get("option_type", "").lower() == "call"),
                None,
            )
            if not live_opt:
                LOG.log(f"  {key}: no chain match for ${short_strike}C {short_exp}")
                continue

            greeks     = live_opt.get("greeks") or {}
            live_delta = abs(float(greeks.get("delta", 0) or 0))
            LOG.log(f"  {key}: ${short_strike}C {short_exp} "
                    f"δ={live_delta:.3f} (ceil={dc:.2f})")

            if live_delta >= 0.35:
                flagged.append({
                    "variant":    key,
                    "strike":     short_strike,
                    "expiry":     short_exp,
                    "delta":      live_delta,
                    "delta_ceil": dc,
                })
        except Exception as e:
            LOG.log(f"  {key}: delta check error: {e}")

    if flagged:
        _send_delta_alert(flagged)

    return flagged


def _send_delta_alert(flagged: list[dict]):
    try:
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        if not smtp_user or not smtp_pass:
            LOG.log("⚠️ SMTP not configured — delta alert not sent")
            return

        now  = datetime.now().strftime("%b %d %Y %I:%M %p ET")
        rows = "".join(
            f"<tr>"
            f"<td style='padding:8px 14px;color:#fff;font-weight:700'>{f['variant']}</td>"
            f"<td style='padding:8px 14px;color:#ff9800'>${f['strike']:.0f}C {f['expiry']}</td>"
            f"<td style='padding:8px 14px;color:#ff3366;font-weight:700'>δ={f['delta']:.3f}</td>"
            f"<td style='padding:8px 14px;color:#aaa'>ceil {f['delta_ceil']:.2f}</td>"
            f"</tr>"
            for f in flagged
        )
        html = f"""
        <div style="background:#05080a;color:#ccc;font-family:'IBM Plex Mono',monospace;
                    padding:24px;max-width:700px;margin:0 auto">
          <div style="font-size:20px;font-weight:800;color:#ff3366;margin-bottom:4px">
            ⚡ Delta Alert — Roll Review Needed
          </div>
          <div style="font-size:11px;color:#555;margin-bottom:20px">{now}</div>
          <table style="width:100%;border-collapse:collapse;background:#0c1215;
                        border:1px solid #1a252c;border-radius:6px;overflow:hidden">
            <thead>
              <tr style="background:#111820">
                <th style="padding:8px 14px;text-align:left;font-size:10px;color:#444;
                           text-transform:uppercase;letter-spacing:2px">Variant</th>
                <th style="padding:8px 14px;text-align:left;font-size:10px;color:#444;
                           text-transform:uppercase;letter-spacing:2px">Short</th>
                <th style="padding:8px 14px;text-align:left;font-size:10px;color:#444;
                           text-transform:uppercase;letter-spacing:2px">Live Delta</th>
                <th style="padding:8px 14px;text-align:left;font-size:10px;color:#444;
                           text-transform:uppercase;letter-spacing:2px">Ceiling</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <div style="margin-top:16px;padding:12px;background:#1a0000;
                      border:1px solid #ff3366;border-radius:4px;
                      color:#ff6666;font-size:12px">
            ⚠ Delta ≥ 0.35 detected. Review positions — consider early roll.
          </div>
          <div style="margin-top:20px;font-size:10px;color:#333">
            VIX 5W Suite · Tradier Orchestrator · Sandbox
          </div>
        </div>"""

        subject = (f"[VIX DELTA] ⚡ {len(flagged)} position(s) δ≥0.35 — "
                   f"{now[:12]}")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = smtp_user
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, smtp_user, msg.as_string())
        LOG.log(f"📧 Delta alert sent: {subject[:60]}")
    except Exception as e:
        LOG.log(f"⚠️ Delta alert email failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",   action="store_true", help="Status only")
    parser.add_argument("--preview", action="store_true", help="Show decisions only")
    parser.add_argument("--paper",   action="store_true", help="Execute on sandbox")
    parser.add_argument("--live",    action="store_true", help="Execute on live (future)")
    args = parser.parse_args()

    if args.live:
        print("⚠️ Live mode not yet enabled")
        return

    run(
        sandbox    = not args.live,
        preview    = args.preview or (not args.paper and not args.check),
        check_only = args.check,
    )


if __name__ == "__main__":
    main()
