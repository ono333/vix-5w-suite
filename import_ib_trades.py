#!/usr/bin/env python3
"""
import_ib_trades.py
Full IB paper trade history import — Feb 27 through Mar 20.
Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")

# ── Position IDs ──────────────────────────────────────────────────────────
POS = {
    "V1": "V1-20260112111933",
    "V2": "V2-20260120161227",
    "V3": "V3-20260112123803",
    "V4": "V4-20260112123937",
    "V5": "V5-20260124124054",
}

# ── Roll history to record ────────────────────────────────────────────────
# Each entry: (variant, roll_date, buyback_price, new_strike, new_expiry,
#              new_credit, contracts, uvxy_approx, vix_approx, notes)
#
# Variant assignment per cycle (confirmed by user):
#   55C → V1 + V3 (7 each)
#   56C → V2 (8)
#   57C → V4 (5)
#   59C → V5 (5)
#
# For earlier cycles we use 5 contracts each (original paper sizing)

ROLLS = [
    # ── Feb 27: Close 27FEB26, open 06MAR26 ──────────────────────────────
    ("V1","2026-02-27", 0.010, 43.0,"2026-03-06", 1.800, 5, 38.0, 20.0, "27FEB→06MAR roll"),
    ("V2","2026-02-27", 0.056, 44.0,"2026-03-06", 1.650, 5, 38.0, 20.0, "27FEB→06MAR roll"),
    ("V3","2026-02-27", 0.210, 45.0,"2026-03-06", 1.500, 5, 38.0, 20.0, "27FEB→06MAR roll"),
    # V4 and V5 shared the 43C position (15 total, split)
    ("V4","2026-02-27", 0.010, 43.0,"2026-03-06", 1.685, 5, 38.0, 20.0, "27FEB→06MAR roll"),
    ("V5","2026-02-27", 0.010, 43.0,"2026-03-06", 1.685, 5, 38.0, 20.0, "27FEB→06MAR roll"),

    # ── Mar 5: Close 06MAR26, open 06MAR26 48/49C and 13MAR26 47C ────────
    # V1 rolled to 13MAR26 47C (18 total — V1+V3 combined, 9 each approx)
    ("V1","2026-03-05", 2.100, 47.0,"2026-03-13", 2.516, 5, 42.0, 22.0, "06MAR→13MAR 47C roll"),
    ("V3","2026-03-05", 1.908, 48.0,"2026-03-06", 0.573, 5, 42.0, 22.0, "06MAR→06MAR 48C roll"),
    ("V2","2026-03-05", 1.503, 49.0,"2026-03-06", 0.415, 5, 42.0, 22.0, "06MAR→06MAR 49C roll"),
    ("V4","2026-03-05", 2.100, 47.0,"2026-03-13", 2.516, 5, 42.0, 22.0, "06MAR→13MAR 47C roll"),
    ("V5","2026-03-05", 2.100, 49.0,"2026-03-13", 2.060, 5, 42.0, 22.0, "06MAR→13MAR 49C roll"),

    # ── Mar 13: Close 13MAR26 (assigned), open 20MAR26 52C ───────────────
    ("V1","2026-03-13", 0.0,  52.0,"2026-03-20", 4.000, 5, 47.0, 24.0, "13MAR assigned→20MAR 52C"),
    ("V4","2026-03-13", 0.0,  52.0,"2026-03-20", 4.000, 5, 47.0, 24.0, "13MAR assigned→20MAR 52C"),

    # ── Mar 20: Close 20MAR26 52C, open 27MAR26 positions ────────────────
    # Using ACTUAL fill prices from IB (T. Price column)
    ("V1","2026-03-20", 2.250, 55.0,"2026-03-27", 3.725, 7, 54.10, 24.06, "20MAR→27MAR 55C roll"),
    ("V3","2026-03-20", 0.0,   55.0,"2026-03-27", 3.725, 7, 54.10, 24.06, "20MAR→27MAR 55C roll"),
    ("V2","2026-03-20", 0.0,   56.0,"2026-03-27", 3.200, 8, 54.10, 24.06, "20MAR→27MAR 56C roll"),
    ("V4","2026-03-20", 2.250, 57.0,"2026-03-27", 2.900, 5, 54.10, 24.06, "20MAR→27MAR 57C roll"),
    ("V5","2026-03-20", 0.0,   59.0,"2026-03-27", 2.170, 5, 54.10, 24.06, "20MAR→27MAR 59C roll"),
]


def main():
    from trade_log import get_trade_log
    tl = get_trade_log()

    print("\n=== Paper Trade Full History Import ===\n")
    print("Positions:")
    for v, pid in POS.items():
        pos = tl.diagonal_positions.get(pid)
        if pos:
            short = pos.current_short_leg
            print(f"  {v} ({pid}): short ${short.strike if short else '—'} "
                  f"exp {short.expiration_date if short else '—'}")
        else:
            print(f"  {v} ({pid}): ❌ NOT FOUND")

    print(f"\nRecording {len(ROLLS)} roll entries...\n")

    ok = 0
    for (variant, roll_date, buyback, new_strike,
         new_expiry, new_credit, contracts,
         uvxy, vix, notes) in ROLLS:

        pid = POS.get(variant)
        if not pid:
            print(f"❌  {variant}: no position ID"); continue

        pos = tl.diagonal_positions.get(pid)
        if not pos:
            print(f"❌  {pid} not found"); continue

        # Skip if this roll is already the current short
        short = pos.current_short_leg
        if (short and short.expiration_date == new_expiry
                and float(short.strike) == new_strike):
            print(f"⏭  {variant} {roll_date}: already at "
                  f"${new_strike} exp {new_expiry} — skip")
            continue

        try:
            new_leg, roll = tl.roll_diagonal_short(
                position_id      = pid,
                exit_price       = buyback,
                new_strike       = new_strike,
                new_expiration   = new_expiry,
                new_credit       = new_credit,
                underlying_price = uvxy,
                regime           = "EXTREME",
                vix_level        = vix,
                vix_percentile   = 0.85,
                contracts        = contracts,
                roll_date        = roll_date,
            )
            net = new_credit - buyback
            print(f"✅  {variant} {roll_date}: "
                  f"buyback ${buyback:.3f} → ${new_strike}C "
                  f"exp {new_expiry} @ ${new_credit:.3f} "
                  f"net ${net:+.3f} | {notes}")
            ok += 1
        except Exception as e:
            print(f"⚠️  {variant} {roll_date}: {e}")
            # Fallback — just update current short
            try:
                tl.add_short_leg(pid, new_strike, new_expiry,
                                 new_credit, contracts)
                print(f"   ↳ Added via add_short_leg")
                ok += 1
            except Exception as e2:
                print(f"   ↳ Failed: {e2}")

    tl._save()
    print(f"\n✅ Done — {ok}/{len(ROLLS)} rolls recorded.")
    print("Check Streamlit → Paper Trading → Trade Log")

    # P&L summary
    print("\n--- Realized P&L from IB report ---")
    realized = [
        ("06MAR26 43C closed",  -426.68),
        ("06MAR26 44C closed",  -151.79),
        ("06MAR26 45C closed",    40.70),
        ("06MAR26 48C assigned",   0.00),
        ("06MAR26 49C assigned",   0.00),
        ("13MAR26 47C assigned",   0.00),
        ("13MAR26 49C assigned",   0.00),
        ("20MAR26 52C closed",  2432.93),
        ("02APR26 46P closed",  -531.08),
    ]
    total = sum(v for _, v in realized)
    for label, pnl in realized:
        print(f"  {label:30s}  ${pnl:+,.2f}")
    print(f"  {'─'*42}")
    print(f"  {'Net realized P&L':30s}  ${total:+,.2f}")


if __name__ == "__main__":
    main()
