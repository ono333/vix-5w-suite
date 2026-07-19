#!/usr/bin/env python3
"""fidelity_ledger.py — stage 1 of the broker-truth ledger: parse a Fidelity
activity CSV and reconstruct option round-trips (FIFO, quantity-splitting).

Broker is truth: every row in the output traces to broker fills; nothing is
averaged, inferred, or hand-entered.

Usage:
  python3 fidelity_ledger.py --csv <History_for_Account.csv> [--symbol UVXY]
                             [--out ~/.vix_suite/broker_ledger_uvxy.csv]

Output columns:
  side (SHORT/LONG), strike, expiry, open_date, open_price, qty,
  close_date, close_price, close_kind (BTC/STC/EXP/OPEN), commissions,
  fees, pnl  (pnl blank for still-open legs)
"""

import argparse
import csv
import os
import re
from collections import deque
from datetime import date, datetime

OPT_RE = re.compile(r"-?([A-Z]+)(\d{6})([CP])([\d.]+)")


def parse_fills(path, symbol):
    fills = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header_seen = False
        for row in rdr:
            if not row or not any(row):
                continue
            if not header_seen:
                if row[0].strip() == "Run Date":
                    header_seen = True
                continue
            if len(row) < 9:
                continue
            action, sym = row[1].strip(), row[2].strip()
            m = OPT_RE.search(sym.replace(" ", ""))
            if not m or m.group(1) != symbol:
                continue
            _, ymd, right, strike = m.groups()
            expiry = f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}"
            try:
                d = datetime.strptime(row[0].strip(), "%m/%d/%Y").date()
            except ValueError:
                continue
            a = action.upper()
            if a.startswith("YOU SOLD OPENING"):
                kind = "STO"
            elif a.startswith("YOU BOUGHT CLOSING"):
                kind = "BTC"
            elif a.startswith("YOU BOUGHT OPENING"):
                kind = "BTO"
            elif a.startswith("YOU SOLD CLOSING"):
                kind = "STC"
            elif a.startswith("EXPIRED"):
                kind = "EXP"
                m2 = re.search(r"as of (\d{4}-\d{2}-\d{2})", action)
                if m2:
                    d = date.fromisoformat(m2.group(1))
            elif a.startswith("ASSIGNED"):
                kind = "ASGN"
            else:
                continue
            qty = int(float(row[6].replace(",", "") or 0))
            price = float(row[5]) if row[5] not in ("", None) else 0.0
            comm = float(row[7]) if row[7] not in ("", None) else 0.0
            fees = float(row[8]) if row[8] not in ("", None) else 0.0
            fills.append({"date": d, "kind": kind, "expiry": expiry,
                          "strike": float(strike), "right": right, "qty": qty,
                          "price": price, "comm": comm, "fees": fees})
    fills.sort(key=lambda x: (x["date"], x["kind"] not in ("STO", "BTO")))
    return fills


def build_roundtrips(fills):
    """FIFO-pair opens with closes per (strike, expiry, side).
    Per-contract commission/fees are prorated across splits."""
    opens = {}   # (side, strike, expiry) -> deque of open lots
    trips = []

    def per_ct(f):
        n = abs(f["qty"]) or 1
        return (f["comm"] + f["fees"]) / n

    for f in fills:
        k = f["kind"]
        if k in ("STO", "BTO"):
            side = "SHORT" if k == "STO" else "LONG"
            opens.setdefault((side, f["strike"], f["expiry"]), deque()).append({
                "date": f["date"], "price": f["price"],
                "qty": abs(f["qty"]), "cost_ct": per_ct(f)})
            continue
        if k in ("BTC", "STC", "EXP", "ASGN"):
            if k == "BTC":
                side = "SHORT"
            elif k == "STC":
                side = "LONG"
            else:  # EXP/ASGN: Fidelity qty sign — negative removes a long lot
                side = "LONG" if f["qty"] < 0 else "SHORT"
            key = (side, f["strike"], f["expiry"])
            dq = opens.get(key)
            remaining = abs(f["qty"])
            close_ct = per_ct(f)
            close_px = 0.0 if k in ("EXP",) else f["price"]
            while remaining > 0 and dq:
                lot = dq[0]
                take = min(lot["qty"], remaining)
                costs = (lot["cost_ct"] + close_ct) * take
                if side == "SHORT":
                    pnl = (lot["price"] - close_px) * take * 100 - costs
                else:
                    pnl = (close_px - lot["price"]) * take * 100 - costs
                trips.append({
                    "side": side, "strike": f["strike"], "expiry": f["expiry"],
                    "open_date": lot["date"], "open_price": lot["price"],
                    "qty": take, "close_date": f["date"],
                    "close_price": close_px, "close_kind": k,
                    "costs": round(costs, 2), "pnl": round(pnl, 2)})
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] == 0:
                    dq.popleft()
            if remaining > 0:
                trips.append({
                    "side": side, "strike": f["strike"], "expiry": f["expiry"],
                    "open_date": None, "open_price": None, "qty": remaining,
                    "close_date": f["date"], "close_price": close_px,
                    "close_kind": k, "costs": round(close_ct * remaining, 2),
                    "pnl": None, "note": "UNMATCHED CLOSE (pre-CSV open?)"})

    for (side, strike, expiry), dq in opens.items():
        for lot in dq:
            if lot["qty"] > 0:
                trips.append({
                    "side": side, "strike": strike, "expiry": expiry,
                    "open_date": lot["date"], "open_price": lot["price"],
                    "qty": lot["qty"], "close_date": None, "close_price": None,
                    "close_kind": "OPEN",
                    "costs": round(lot["cost_ct"] * lot["qty"], 2),
                    "pnl": None})
    trips.sort(key=lambda t: (t["open_date"] or t["close_date"]))
    return trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", default="UVXY")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fills = parse_fills(args.csv, args.symbol)
    trips = build_roundtrips(fills)

    closed = [t for t in trips if t["pnl"] is not None]
    open_ = [t for t in trips if t["close_kind"] == "OPEN"]
    unmatched = [t for t in trips if t.get("note")]

    print(f"{args.symbol}: {len(fills)} fills -> {len(closed)} closed "
          f"round-trips, {len(open_)} open lots, {len(unmatched)} unmatched")
    hdr = (f"{'side':<6}{'strike':>7} {'expiry':<11}{'open':<11}{'o.px':>6} "
           f"{'qty':>3} {'close':<11}{'c.px':>6} {'how':<5}{'costs':>7}{'pnl':>9}")
    print("\n" + hdr)
    for t in trips:
        print(f"{t['side']:<6}{t['strike']:>7g} {t['expiry']:<11}"
              f"{str(t['open_date'] or '?'):<11}"
              f"{t['open_price'] if t['open_price'] is not None else '':>6} "
              f"{t['qty']:>3} {str(t['close_date'] or '—'):<11}"
              f"{t['close_price'] if t['close_price'] is not None else '':>6} "
              f"{t['close_kind']:<5}{t['costs']:>7.2f}"
              f"{t['pnl'] if t['pnl'] is not None else '':>9}")

    total = sum(t["pnl"] for t in closed)
    by_side = {}
    for t in closed:
        by_side[t["side"]] = by_side.get(t["side"], 0.0) + t["pnl"]
    print(f"\nRealized P&L (closed trips): ${total:,.2f}   "
          + "  ".join(f"{s}: ${v:,.2f}" for s, v in sorted(by_side.items())))
    months = {}
    for t in closed:
        mkey = str(t["close_date"])[:7]
        months[mkey] = months.get(mkey, 0.0) + t["pnl"]
    print("By close month: " + "  ".join(f"{m}: {v:+,.0f}"
                                         for m, v in sorted(months.items())))

    if args.out:
        out = os.path.expanduser(args.out)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "side", "strike", "expiry", "open_date", "open_price", "qty",
                "close_date", "close_price", "close_kind", "costs", "pnl",
                "note"])
            w.writeheader()
            for t in trips:
                w.writerow({k: t.get(k, "") for k in w.fieldnames})
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
