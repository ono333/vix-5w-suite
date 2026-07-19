#!/usr/bin/env python3
"""fidelity_import.py — stage 2 of the broker-truth ledger: rebuild
real_trade_log_fidelity.json from a Fidelity activity CSV.

Broker is truth. Every leg in the output traces to a broker fill; nothing is
hand-entered. Variant labels are carried over ONLY where the old ledger's leg
was broker-verified (Feb 19 - Apr 29 window, mapping table below); all later
activity books to a consolidated position under the 46C Sep18 long.

Known deliberate exclusions (visible in fidelity_ledger.py output instead):
  - 3x 29C Jun26 long, bought 06/15 sold 06/18 (-$73): not a diagonal leg.

Usage:
  python3 fidelity_import.py --csv <History_for_Account.csv>
      [--out ~/.vix_suite/real_trade_log_fidelity.rebuilt.json]
      [--apply]     # back up live file, then replace it
Validation: loads the written file through real_trade_log.RealTradeLog and
prints per-position summaries. Run --apply only after reviewing --out.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity_ledger import parse_fills, build_roundtrips  # noqa: E402

LIVE = os.path.expanduser("~/.vix_suite/real_trade_log_fidelity.json")

# ── Positions: one per long (strike, expiry, variant). True broker fills. ──
POSITIONS = {
    "V1": dict(position_id="RB-V1-45C-SEP18", variant="V1_INCOME_HARVESTER",
               long_strike=45.0, long_expiration="2026-09-18",
               entry_regime="CALM", entry_vix_level=21.31, entry_percentile=0.13),
    "V5A": dict(position_id="RB-V5A-51C-JUN18", variant="V5_REGIME_ALLOCATOR",
                long_strike=51.0, long_expiration="2026-06-18",
                entry_regime="CALM", entry_vix_level=19.86, entry_percentile=0.13),
    "V4": dict(position_id="RB-V4-65C-APR17", variant="V4_TAIL_HUNTER",
               long_strike=65.0, long_expiration="2026-04-17",
               entry_regime="EXTREME", entry_vix_level=21.93, entry_percentile=0.83),
    "V5B": dict(position_id="RB-V5B-65C-APR17", variant="V5_REGIME_ALLOCATOR",
                long_strike=65.0, long_expiration="2026-04-17",
                entry_regime="EXTREME", entry_vix_level=22.43, entry_percentile=0.85),
    "CONS": dict(position_id="RB-CONS-46C-SEP18", variant="CONSOLIDATED",
                 long_strike=46.0, long_expiration="2026-09-18",
                 entry_regime="UNKNOWN", entry_vix_level=0.0, entry_percentile=0.0),
}

# Long open fills -> position book. (date, strike, expiry, price)
LONG_MAP = {
    ("2026-02-19", 45.0, "2026-09-18", 13.50): "V1",
    ("2026-02-27", 45.0, "2026-09-18", 13.42): "V1",
    ("2026-02-27", 51.0, "2026-06-18", 8.70): "V5A",
    ("2026-03-17", 65.0, "2026-04-17", 2.68): "V4",
    ("2026-03-17", 65.0, "2026-04-17", 2.89): "V5B",
    ("2026-04-08", 46.0, "2026-09-18", 14.65): "CONS",
}

# Short open fills -> book, keyed (open_date, strike, expiry, open_price).
# Broker-verified against the old ledger's variant labels (recon Jul 19).
# Duplicate-key fills (two identical opens, different books) carry a count.
SHORT_MAP = {
    ("2026-02-19", 42.0, "2026-02-27", 2.18): ["V1"],
    ("2026-02-27", 43.0, "2026-03-06", 1.66): ["V1"],
    ("2026-02-27", 43.0, "2026-03-06", 1.72): ["V1"],
    ("2026-03-02", 45.0, "2026-03-06", 1.04): ["V5A"],
    ("2026-03-03", 47.0, "2026-03-13", 4.23): ["V1"],
    ("2026-03-03", 46.5, "2026-03-13", 4.70): ["V5A"],
    ("2026-03-06", 50.0, "2026-03-20", 5.57): ["V1"],
    ("2026-03-06", 51.0, "2026-03-20", 5.36): ["V5A"],
    ("2026-03-17", 47.0, "2026-03-27", 2.56): ["V4"],
    ("2026-03-17", 53.0, "2026-03-27", 1.62): ["V5A"],
    ("2026-03-20", 53.0, "2026-03-27", 3.29): ["V1"],
    ("2026-03-20", 54.0, "2026-03-27", 2.67): ["V5B"],
    ("2026-03-24", 54.0, "2026-04-02", 3.26): ["V1"],
    ("2026-03-24", 56.0, "2026-04-10", 5.21): ["V4"],
    ("2026-03-26", 56.0, "2026-04-10", 5.86): ["V1"],
    ("2026-03-26", 57.0, "2026-04-10", 5.31): ["V5B"],
    ("2026-04-09", 45.0, "2026-04-17", 1.45): ["V1"],
    ("2026-04-09", 46.0, "2026-04-17", 1.22): ["V5B"],
    ("2026-04-09", 46.0, "2026-04-17", 1.28): ["V4"],
    ("2026-04-17", 39.0, "2026-04-24", 1.30): ["V1", "V5A"],  # two 1-lots
    ("2026-04-17", 39.0, "2026-04-24", 1.33): ["V5B"],
    ("2026-04-17", 40.0, "2026-04-24", 1.07): ["V1"],
    ("2026-04-22", 43.0, "2026-05-01", 1.32): ["V1"],
    ("2026-04-22", 47.0, "2026-05-08", 1.38): ["V5A"],
    ("2026-04-29", 40.0, "2026-05-08", 1.10): ["V1"],
    # May 7 onward: no verified variant records exist -> CONS (default)
}

SKIP_LONGS = {(29.0, "2026-06-26")}   # 3-day 29C trade: not a diagonal leg


def book_for_short(trip, used_counts):
    key = (str(trip["open_date"]), trip["strike"], trip["expiry"],
           trip["open_price"])
    books = SHORT_MAP.get(key)
    if not books:
        return "CONS"
    i = used_counts.get(key, 0)
    used_counts[key] = i + 1
    return books[min(i, len(books) - 1)]


def build(csv_path):
    fills = parse_fills(csv_path, "UVXY")
    trips = build_roundtrips(fills)

    pos = {}
    for bk, meta in POSITIONS.items():
        pos[bk] = {
            "position_id": meta["position_id"],
            "variant_id": meta["variant"], "variant_name": meta["variant"],
            "entry_date": None,  # set from first long fill
            "entry_regime": meta["entry_regime"],
            "entry_vix_level": meta["entry_vix_level"],
            "entry_percentile": meta["entry_percentile"],
            "contracts": 0,
            "broker": "Fidelity", "account_id": "Z31686168",
            "long_strike": meta["long_strike"],
            "long_expiration": meta["long_expiration"],
            "long_entry_price": 0.0, "long_fill_price": 0.0,
            "long_commission": 0.65, "long_current_price": 0.0,
            "long_status": "open", "short_legs": [], "roll_history": [],
            "status": "open", "close_date": None, "close_reason": None,
            "notes": "", "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "_long_fills": [],
        }

    unmapped_longs = []
    for t in trips:
        if t["side"] != "LONG":
            continue
        if (t["strike"], t["expiry"]) in SKIP_LONGS:
            continue
        key = (str(t["open_date"]), t["strike"], t["expiry"], t["open_price"])
        bk = LONG_MAP.get(key)
        if bk is None:
            unmapped_longs.append(t)
            continue
        p = pos[bk]
        p["_long_fills"].append(t)
        if t["close_kind"] == "EXP":
            p["long_status"] = "expired"
            p["status"] = "closed"
            p["close_date"] = str(t["close_date"])
            p["close_reason"] = "long_expired"

    for p in pos.values():
        lf = p["_long_fills"]
        if not lf:
            continue
        n = sum(t["qty"] for t in lf)
        wavg = sum(t["open_price"] * t["qty"] for t in lf) / n
        p["contracts"] = n
        p["long_fill_price"] = round(wavg, 4)
        p["long_entry_price"] = round(wavg, 4)
        p["entry_date"] = str(min(t["open_date"] for t in lf))
        p["notes"] = "broker fills: " + "; ".join(
            f"+{t['qty']}@{t['open_price']:g} {t['open_date']}" for t in lf)

    used = {}
    counters = {bk: 0 for bk in pos}
    for t in sorted((t for t in trips if t["side"] == "SHORT"),
                    key=lambda x: (x["open_date"], x["strike"])):
        bk = book_for_short(t, used)
        p = pos[bk]
        counters[bk] += 1
        n_open = abs(t["qty"])
        open_ct = round(t["costs"] / (2 * n_open), 2) if t["close_kind"] != "OPEN" \
            else round(t["costs"] / n_open, 2)
        status = ("open" if t["close_kind"] == "OPEN" else
                  "expired" if t["close_kind"] == "EXP" else "closed")
        leg = {
            "leg_id": f"{p['position_id']}-S{counters[bk]}",
            "position_id": p["position_id"],
            "entry_date": str(t["open_date"]),
            "strike": t["strike"], "expiration_date": t["expiry"],
            "entry_credit": t["open_price"], "fill_price": t["open_price"],
            "contracts": n_open, "broker": "Fidelity",
            "account_id": "Z31686168", "commission": open_ct,
            "slippage": 0.0, "status": status, "current_price": 0.0,
            "exit_date": str(t["close_date"]) if t["close_date"] else None,
            "exit_price": t["close_price"], "exit_fill_price": t["close_price"],
            "exit_commission": open_ct if status == "closed" else 0.0,
            "exit_reason": {"BTC": "closed", "EXP": "expired",
                            "ASGN": "assigned"}.get(t["close_kind"]),
            "pnl": t["pnl"] if t["pnl"] is not None else 0.0,
            "notes": "broker", "created_at": datetime.now().isoformat(),
        }
        p["short_legs"].append(leg)

    for p in pos.values():
        del p["_long_fills"]
        if p["entry_date"] is None:
            p["entry_date"] = (min((l["entry_date"] for l in p["short_legs"]),
                                   default=str(date.today())))
    return ({"diagonal_positions": {p["position_id"]: p for p in pos.values()
                                    if p["contracts"] or p["short_legs"]},
             "history": [],
             "updated_at": datetime.now().isoformat()},
            unmapped_longs)


def validate(path):
    try:
        import real_trade_log as rtlmod
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        try:
            import real_trade_log as rtlmod
        except ImportError:
            print("validate: real_trade_log.py not importable — skipped")
            return
    from pathlib import Path
    rtl = rtlmod.RealTradeLog(path=Path(path))
    print(f"\nvalidate: loaded {len(rtl.diagonal_positions)} positions "
          f"through RealTradeLog")
    for pid, p in rtl.diagonal_positions.items():
        print(f"  {pid}: {p.status:<7} long {p.long_strike:g} "
              f"{p.long_expiration} ({p.long_status})  shorts={len(p.short_legs)} "
              f"gross_cr=${p.gross_short_credits:,.0f} "
              f"net_cr=${p.net_short_credits:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/.vix_suite/real_trade_log_fidelity.rebuilt.json"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    data, unmapped = build(args.csv)
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    n_legs = sum(len(p["short_legs"]) for p in data["diagonal_positions"].values())
    print(f"Wrote {out}: {len(data['diagonal_positions'])} positions, "
          f"{n_legs} short legs")
    if unmapped:
        print("WARNING unmapped long fills (excluded):")
        for t in unmapped:
            print(f"  {t['open_date']} {t['strike']}C {t['expiry']} "
                  f"@{t['open_price']}")

    validate(out)

    if args.apply:
        if os.path.exists(LIVE):
            bak = LIVE + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(LIVE, bak)
            print(f"backed up live ledger -> {bak}")
        shutil.copy2(out, LIVE)
        print(f"APPLIED -> {LIVE}")


if __name__ == "__main__":
    main()
