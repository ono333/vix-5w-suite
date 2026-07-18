#!/usr/bin/env python3
"""
reconcile.py
────────────
Reconciles real_trade_log.json against a Fidelity CSV export.

Usage:
    python3 reconcile.py --csv /path/to/Portfolio_Positions.csv [--apply]

Without --apply: prints a dry-run diff only.
With    --apply: writes mutations to real_trade_log.json (backup created first).

Logic:
  1. Parse Fidelity CSV → ground truth of open option legs
  2. Parse real_trade_log.json → what the system thinks is open
  3. Diff: find shorts/longs that need closing, opening, or updating
  4. Generate mutations
  5. Apply (or dry-run)
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path

TRADE_LOG_PATH = Path.home() / ".vix_suite" / "real_trade_log_fidelity.json"
BACKUP_DIR     = Path.home() / ".vix_suite" / "backups"

# ─── Fidelity CSV parser ──────────────────────────────────────────────────────

TICKER_RE = re.compile(
    r"(?P<sym>[A-Z]+)(?P<yr>\d{2})(?P<mo>\d{2})(?P<day>\d{2})(?P<cp>[CP])(?P<strike>[\d.]+)"
)

def _clean(v) -> float:
    try:
        if v is None: return 0.0
        s = str(v).replace("$","").replace(",","").replace("%","").replace("+","").strip()
        return float(s) if s not in ("", "--", "N/A", "None") else 0.0
    except: return 0.0

def parse_fidelity_csv(csv_path: str) -> dict:
    """
    Returns dict of actual open positions keyed by (underlying, expiry, strike, cp).
    Only includes UVXY options.
    """
    import csv
    positions = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym_raw = row.get("Symbol") or ""
            if not isinstance(sym_raw, str): continue
            sym_raw = sym_raw.strip().lstrip("-").strip()
            if not sym_raw: continue
            m = TICKER_RE.match(sym_raw)
            if not m or m.group("sym") != "UVXY":
                continue

            yr     = 2000 + int(m.group("yr"))
            mo     = int(m.group("mo"))
            dy     = int(m.group("day"))
            expiry = date(yr, mo, dy).isoformat()
            strike = float(m.group("strike"))
            cp     = m.group("cp")
            qty    = int(_clean(row.get("Quantity", 0)))
            cost   = _clean(row.get("Average Cost Basis") or 0)
            last   = _clean(row.get("Last Price") or 0)
            pnl    = _clean(row.get("Total Gain/Loss Dollar") or 0)

            key = (expiry, strike, cp)
            positions[key] = {
                "expiry":    expiry,
                "strike":    strike,
                "cp":        cp,
                "qty":       qty,          # negative = short, positive = long
                "avg_cost":  cost,
                "last":      last,
                "total_pnl": pnl,
                "is_short":  qty < 0,
                "contracts": abs(qty),
            }
    return positions


# ─── Trade log reader ─────────────────────────────────────────────────────────

def load_trade_log(path: Path = TRADE_LOG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def get_open_shorts(log: dict) -> list[dict]:
    """Return all short legs with status='open' across all positions."""
    result = []
    for pid, pos in log["diagonal_positions"].items():
        for leg in pos.get("short_legs", []):
            if leg.get("status") == "open":
                result.append({
                    "position_id": pid,
                    "leg_id":      leg["leg_id"],
                    "strike":      leg["strike"],
                    "expiry":      leg["expiration_date"],
                    "contracts":   leg.get("contracts", pos.get("contracts", 1)),
                    "credit":      leg.get("entry_credit", 0),
                })
    return result


def get_open_longs(log: dict) -> list[dict]:
    """Return all long legs with long_status='open'."""
    result = []
    for pid, pos in log["diagonal_positions"].items():
        if pos.get("long_status") == "open":
            result.append({
                "position_id": pid,
                "variant":     pos.get("variant_id"),
                "strike":      pos["long_strike"],
                "expiry":      pos["long_expiration"],
                "contracts":   pos.get("contracts", 1),
            })
    return result


# ─── Diff logic ───────────────────────────────────────────────────────────────

def diff(fidelity: dict, log: dict) -> dict:
    """
    Compare ground truth vs log.
    Returns mutations: {
        close_shorts: [...],   # log shows open, Fidelity doesn't have them
        add_shorts:   [...],   # Fidelity has shorts not in log
        update_longs: [...],   # log long doesn't match Fidelity long
        fix_corrupted:[...],   # duplicate open/rolled legs
    }
    """
    mutations = {
        "close_shorts":  [],
        "add_shorts":    [],
        "update_longs":  [],
        "fix_corrupted": [],
    }

    # ── 1. Shorts in log that Fidelity no longer shows → close them ───────────
    log_open_shorts = get_open_shorts(log)
    fidelity_shorts = {k: v for k, v in fidelity.items() if v["is_short"] and k[2] == "C"}

    for ls in log_open_shorts:
        key = (ls["expiry"], ls["strike"], "C")
        if key not in fidelity_shorts:
            mutations["close_shorts"].append({
                "position_id": ls["position_id"],
                "leg_id":      ls["leg_id"],
                "strike":      ls["strike"],
                "expiry":      ls["expiry"],
                "reason":      "not_in_fidelity",
                "exit_price":  0.0,
                "exit_reason": "expired_worthless",
            })

    # ── 2. Shorts in Fidelity not in log → need adding ────────────────────────
    log_short_keys = {(ls["expiry"], ls["strike"]) for ls in log_open_shorts}
    for key, fpos in fidelity_shorts.items():
        expiry, strike, cp = key
        if (expiry, strike) not in log_short_keys:
            mutations["add_shorts"].append({
                "expiry":    expiry,
                "strike":    strike,
                "contracts": fpos["contracts"],
                "avg_cost":  fpos["avg_cost"],
                "last":      fpos["last"],
            })

    # ── 3. Longs: check if log long matches any Fidelity long ─────────────────
    fidelity_longs = {k: v for k, v in fidelity.items()
                      if not v["is_short"] and k[2] == "C"}
    log_open_longs = get_open_longs(log)

    for ll in log_open_longs:
        key = (ll["expiry"], ll["strike"], "C")
        if key not in fidelity_longs:
            # Find what Fidelity actually has for this position's contracts
            mutations["update_longs"].append({
                "position_id":   ll["position_id"],
                "variant":       ll["variant"],
                "old_strike":    ll["strike"],
                "old_expiry":    ll["expiry"],
                "new_long":      None,  # filled below
            })

    # Try to suggest replacement longs from Fidelity
    unmatched_longs = list(fidelity_longs.items())
    for mut in mutations["update_longs"]:
        for key, fpos in unmatched_longs:
            expiry, strike, cp = key
            mut["new_long"] = {
                "strike":    strike,
                "expiry":    expiry,
                "contracts": fpos["contracts"],
                "avg_cost":  fpos["avg_cost"],
            }
            break  # assign first unmatched; human confirms

    # ── 4. Find corrupted legs (same expiry+strike with both open and rolled) ──
    for pid, pos in log["diagonal_positions"].items():
        seen: dict[tuple, list] = {}
        for leg in pos.get("short_legs", []):
            k = (leg.get("expiration_date"), leg.get("strike"))
            seen.setdefault(k, []).append(leg)
        for k, legs in seen.items():
            statuses = [l["status"] for l in legs]
            if "open" in statuses and "rolled" in statuses:
                mutations["fix_corrupted"].append({
                    "position_id": pid,
                    "expiry":      k[0],
                    "strike":      k[1],
                    "legs":        [l["leg_id"] for l in legs],
                    "fix":         "keep_rolled_remove_open_duplicate",
                })

    return mutations


# ─── Apply mutations ──────────────────────────────────────────────────────────

def apply_mutations(log: dict, mutations: dict,
                    fidelity: dict, today: str = None) -> dict:
    """Apply computed mutations to the log dict. Returns modified log."""
    if today is None:
        today = date.today().isoformat()
    now   = datetime.now().isoformat()
    diags = log["diagonal_positions"]

    # ── Close stale shorts ────────────────────────────────────────────────────
    for mut in mutations["close_shorts"]:
        pid = mut["position_id"]
        for leg in diags[pid]["short_legs"]:
            if leg["leg_id"] == mut["leg_id"]:
                leg["status"]          = mut["exit_reason"]
                leg["exit_date"]       = today
                leg["exit_price"]      = mut["exit_price"]
                leg["exit_fill_price"] = mut["exit_price"]
                leg["exit_reason"]     = mut["exit_reason"]
                # P&L = credit collected (exit at 0)
                leg["pnl"] = round(
                    leg.get("entry_credit", 0) * leg.get("contracts", 1) * 100, 2
                )
                print(f"  ✅ Closed short leg {leg['leg_id']} "
                      f"${mut['strike']} exp {mut['expiry']} → {mut['exit_reason']}")

    # ── Add new short legs ────────────────────────────────────────────────────
    for mut in mutations["add_shorts"]:
        # Match to best position by contracts
        strike    = mut["strike"]
        expiry    = mut["expiry"]
        contracts = mut["contracts"]
        credit    = mut["avg_cost"]

        # Try to match by contracts count
        matched_pid = None
        for pid, pos in diags.items():
            if pos.get("contracts") == contracts and pos.get("long_status") == "open":
                # Check no existing open short for this expiry
                has_open = any(
                    l["status"] == "open" for l in pos.get("short_legs", [])
                )
                if not has_open:
                    matched_pid = pid
                    break

        if matched_pid is None:
            # Fall back to first position with open long
            for pid, pos in diags.items():
                if pos.get("long_status") == "open":
                    has_open = any(
                        l["status"] == "open" for l in pos.get("short_legs", [])
                    )
                    if not has_open:
                        matched_pid = pid
                        break

        if matched_pid:
            pos      = diags[matched_pid]
            leg_num  = len(pos["short_legs"]) + 1
            new_leg  = {
                "leg_id":           f"{matched_pid}-S{leg_num}",
                "position_id":      matched_pid,
                "entry_date":       today,
                "strike":           strike,
                "expiration_date":  expiry,
                "entry_credit":     credit,
                "fill_price":       credit,
                "contracts":        contracts,
                "broker":           pos.get("broker", "Fidelity"),
                "account_id":       pos.get("account_id", "686168"),
                "commission":       0.65 * contracts,
                "slippage":         0.0,
                "status":           "open",
                "current_price":    mut.get("last", credit),
                "exit_date":        None,
                "exit_price":       None,
                "exit_fill_price":  None,
                "exit_commission":  0.65 * contracts,
                "exit_reason":      None,
                "pnl":              0.0,
                "notes":            f"reconciled from Fidelity CSV {today}",
                "created_at":       now,
            }
            pos["short_legs"].append(new_leg)
            print(f"  ✅ Added short leg ${strike} exp {expiry} "
                  f"× {contracts} to {matched_pid}")
        else:
            print(f"  ⚠️  Could not match short ${strike} exp {expiry} "
                  f"to any position — manual review needed")

    # ── Update mismatched longs ───────────────────────────────────────────────
    for mut in mutations["update_longs"]:
        pid  = mut["position_id"]
        new  = mut.get("new_long")
        if new and pid in diags:
            pos = diags[pid]
            print(f"  ✅ Updated long for {pid}: "
                  f"${mut['old_strike']} {mut['old_expiry']} → "
                  f"${new['strike']} {new['expiry']}")
            pos["long_strike"]      = new["strike"]
            pos["long_expiration"]  = new["expiry"]
            pos["long_entry_price"] = new["avg_cost"]
            pos["long_fill_price"]  = new["avg_cost"]
            pos["long_current_price"] = new["avg_cost"]
            pos["long_status"]      = "open"

    # ── Fix corrupted legs ────────────────────────────────────────────────────
    for mut in mutations["fix_corrupted"]:
        pid = mut["position_id"]
        pos = diags[pid]
        legs = pos["short_legs"]
        # Remove the duplicate open leg, keep rolled
        cleaned = []
        seen_key: set = set()
        for leg in legs:
            k = (leg.get("expiration_date"), leg.get("strike"))
            if k in seen_key and leg["status"] == "open":
                print(f"  ✅ Removed duplicate open leg {leg['leg_id']} "
                      f"(already recorded as rolled)")
                continue
            seen_key.add(k)
            cleaned.append(leg)
        pos["short_legs"] = cleaned

    return log


# ─── Backup ───────────────────────────────────────────────────────────────────

def backup(path: Path = TRADE_LOG_PATH):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"real_trade_log_{ts}.json"
    shutil.copy2(path, dst)
    print(f"  📦 Backup created: {dst}")


# ─── Pretty print diff ────────────────────────────────────────────────────────

def print_diff(mutations: dict):
    print("\n" + "="*60)
    print("RECONCILIATION DIFF")
    print("="*60)

    cs = mutations["close_shorts"]
    print(f"\n🔴 Shorts to CLOSE ({len(cs)}):")
    for m in cs:
        print(f"   ${m['strike']} exp {m['expiry']} [{m['position_id']}] → {m['exit_reason']}")

    as_ = mutations["add_shorts"]
    print(f"\n🟢 Shorts to ADD ({len(as_)}):")
    for m in as_:
        print(f"   ${m['strike']} exp {m['expiry']} × {m['contracts']} @ ${m['avg_cost']}")

    ul = mutations["update_longs"]
    print(f"\n🔵 Longs to UPDATE ({len(ul)}):")
    for m in ul:
        new = m.get("new_long")
        new_str = f"→ ${new['strike']} {new['expiry']}" if new else "→ ?"
        print(f"   [{m['variant']}] ${m['old_strike']} {m['old_expiry']} {new_str}")

    fc = mutations["fix_corrupted"]
    print(f"\n⚠️  Corrupted legs to FIX ({len(fc)}):")
    for m in fc:
        print(f"   [{m['position_id']}] ${m['strike']} exp {m['expiry']} — {m['fix']}")

    print("\n" + "="*60)
    total = len(cs) + len(as_) + len(ul) + len(fc)
    print(f"Total mutations: {total}")
    print("="*60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconcile trade log vs Fidelity CSV")
    parser.add_argument("--csv",   required=True, help="Path to Fidelity CSV export")
    parser.add_argument("--apply", action="store_true",
                        help="Apply mutations (default: dry-run only)")
    args = parser.parse_args()

    print(f"\n📂 Loading Fidelity CSV: {args.csv}")
    fidelity = parse_fidelity_csv(args.csv)
    print(f"   Found {len(fidelity)} UVXY option legs")

    print(f"\n📂 Loading trade log: {TRADE_LOG_PATH}")
    log = load_trade_log()
    n_pos = len(log["diagonal_positions"])
    print(f"   Found {n_pos} diagonal positions")

    mutations = diff(fidelity, log)
    print_diff(mutations)

    total = sum(len(v) for v in mutations.values())
    if total == 0:
        print("\n✅ Trade log is in sync with Fidelity. Nothing to do.")
        return

    if not args.apply:
        print("\n⚡ Dry-run complete. Run with --apply to write changes.")
        return

    print("\n🔧 Applying mutations...")
    backup()
    log = apply_mutations(log, mutations, fidelity)

    with open(TRADE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ Trade log updated: {TRADE_LOG_PATH}")


if __name__ == "__main__":
    main()
