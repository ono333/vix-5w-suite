#!/usr/bin/env python3
"""import_ib_trades.py — Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")

POS = {
    "V1": "V1-20260112111933",
    "V2": "V2-20260120161227",
    "V3": "V3-20260112123803",
    "V4": "V4-20260112123937",
    "V5": "V5-20260112124054",
}

ROLLS = [
    ("V3","2026-02-27", 0.210, 45.0,"2026-03-06", 1.500, 5, 38.0,"27FEB→06MAR"),
    ("V1","2026-03-05", 2.100, 47.0,"2026-03-13", 2.516, 5, 42.0,"06MAR→13MAR 47C"),
    ("V2","2026-03-05", 1.908, 49.0,"2026-03-13", 2.060, 5, 42.0,"06MAR→13MAR 49C"),
    ("V3","2026-03-05", 1.503, 48.0,"2026-03-06", 0.573, 5, 42.0,"06MAR→06MAR 48C"),
    ("V4","2026-03-05", 2.100, 47.0,"2026-03-13", 2.516, 5, 42.0,"06MAR→13MAR 47C"),
    ("V5","2026-03-05", 1.503, 49.0,"2026-03-06", 0.415, 5, 42.0,"06MAR→06MAR 49C"),
    ("V1","2026-03-13", 0.0,  52.0,"2026-03-20", 4.000, 5, 47.0,"13MAR→20MAR 52C"),
    ("V4","2026-03-13", 0.0,  52.0,"2026-03-20", 4.000, 5, 47.0,"13MAR→20MAR 52C"),
    ("V1","2026-03-20", 2.250, 55.0,"2026-03-27", 3.725, 7, 54.10,"20MAR→27MAR 55C"),
    ("V3","2026-03-20", 0.0,   55.0,"2026-03-27", 3.725, 7, 54.10,"20MAR→27MAR 55C"),
    ("V2","2026-03-20", 0.0,   56.0,"2026-03-27", 3.200, 8, 54.10,"20MAR→27MAR 56C"),
    ("V4","2026-03-20", 2.250, 57.0,"2026-03-27", 2.900, 5, 54.10,"20MAR→27MAR 57C"),
    ("V5","2026-03-20", 0.0,   59.0,"2026-03-27", 2.170, 5, 54.10,"20MAR→27MAR 59C"),
]

def main():
    from trade_log import get_trade_log
    tl = get_trade_log()
    print("\n=== Paper Trade Import ===\n")
    ok = 0
    for (variant, roll_date, buyback, new_strike,
         new_expiry, new_credit, contracts, uvxy, notes) in ROLLS:
        pid = POS[variant]
        pos = tl.diagonal_positions.get(pid)
        if not pos:
            print(f"❌  {variant} ({pid}): not found"); continue
        short = pos.current_short_leg
        if (short and short.expiration_date == new_expiry
                and float(short.strike) == new_strike):
            print(f"⏭  {variant} {roll_date}: already at ${new_strike} exp {new_expiry}")
            continue
        try:
            tl.roll_diagonal_short(
                position_id      = pid,
                exit_price       = buyback,
                new_strike       = new_strike,
                new_expiration   = new_expiry,
                new_credit       = new_credit,
                underlying_price = uvxy,
                regime           = "EXTREME",
                contracts        = contracts,
                notes            = f"{roll_date} | {notes}",
            )
            print(f"✅  {variant} {roll_date}: ${new_strike}C exp {new_expiry} "
                  f"@ ${new_credit:.3f} net ${new_credit-buyback:+.3f}")
            ok += 1
        except Exception as e:
            print(f"❌  {variant} {roll_date}: {e}")
    tl._save()
    print(f"\n✅ {ok}/{len(ROLLS)} rolls recorded.")

if __name__ == "__main__":
    main()
