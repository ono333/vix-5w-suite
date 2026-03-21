#!/usr/bin/env python3
"""
import_ib_trades.py — Mar 20 IB roll import
Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")

OPEN_SHORTS = [
    {
        "contract":   "27MAR26 55C",
        "expiry":     "2026-03-27",
        "strike":     55.0,
        "credit":     4.175,
        "variant_id": "V1_INCOME_HARVESTER",
        "position_id":"V1-REAL-20260223144523",
        "contracts":  7,
    },
    {
        "contract":   "27MAR26 55C",
        "expiry":     "2026-03-27",
        "strike":     55.0,
        "credit":     4.175,
        "variant_id": "V3_SHOCK_ABSORBER",
        "position_id":"V3-20260112123803-R10",
        "contracts":  7,
    },
    {
        "contract":   "27MAR26 56C",
        "expiry":     "2026-03-27",
        "strike":     56.0,
        "credit":     3.850,
        "variant_id": "V2_MEAN_REVERSION",
        "position_id":"",          # ← set if V2 has existing position
        "contracts":  8,
    },
    {
        "contract":   "27MAR26 57C",
        "expiry":     "2026-03-27",
        "strike":     57.0,
        "credit":     3.587,
        "variant_id": "V4_TAIL_HUNTER",
        "position_id":"V4-REAL-20260317111104",
        "contracts":  5,
    },
    {
        "contract":   "27MAR26 59C",
        "expiry":     "2026-03-27",
        "strike":     59.0,
        "credit":     3.126,
        "variant_id": "V5_REGIME_ALLOCATOR",
        "position_id":"V5-REAL-20260317145121",
        "contracts":  5,
    },
]

ASSIGNED_SUMMARY = [
    ("06MAR26 48C", "2026-03-06",  6,  3544.76),
    ("06MAR26 49C", "2026-03-06",  8,  3917.28),
    ("13MAR26 47C", "2026-03-13", 18,  8022.34),
    ("13MAR26 49C", "2026-03-13",  2,   488.08),
    ("20MAR26 52C", "2026-03-20", 14,  2432.93),
]


def main():
    from real_trade_log import get_real_trade_log, reset_real_trade_log_cache
    rtl = get_real_trade_log()
    open_pos = rtl.open_positions()

    print("\n=== IB Trade Import — Mar 20 Rolls ===\n")
    print("Current open positions:")
    for pid, pos in open_pos.items():
        short = pos.current_short_leg
        print(f"  {pid}: {pos.variant_name} | "
              f"short ${short.strike if short else '—'} "
              f"exp {short.expiration_date if short else '—'} "
              f"x{pos.contracts}")

    print("\n--- Importing short legs ---\n")

    for t in OPEN_SHORTS:
        pid = t["position_id"]
        if not pid:
            print(f"⚠️  {t['contract']} ({t['variant_id']}): No position_id — skipping")
            continue
        if pid not in open_pos:
            print(f"❌  {t['contract']}: Position {pid} not found")
            continue
        pos = open_pos[pid]
        short = pos.current_short_leg
        if (short and short.expiration_date == t["expiry"]
                and float(short.strike) == t["strike"]):
            print(f"✅  {t['contract']}: Already recorded for {pid}")
            continue
        result = rtl.add_short_leg(
            pid, t["strike"], t["expiry"], t["credit"], t["contracts"]
        )
        if result:
            print(f"✅  {t['contract']}: ${t['strike']}C exp {t['expiry']} "
                  f"x{t['contracts']} @ ${t['credit']:.3f} → {pid} ({t['variant_id']})")
        else:
            print(f"❌  add_short_leg failed for {pid}")

    rtl._save()
    reset_real_trade_log_cache()
    print("\n✅ Saved to real_trade_log.\n")

    print("--- Historical assigned/closed P&L ---")
    total = sum(a[3] for a in ASSIGNED_SUMMARY)
    for contract, dt, qty, pnl in ASSIGNED_SUMMARY:
        print(f"  {contract}  {dt}  x{qty}  P&L: ${pnl:,.2f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total realized: ${total:,.2f}")


if __name__ == "__main__":
    main()
