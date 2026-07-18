#!/usr/bin/env python3
"""
import_ib_trades.py — Full IB paper trade history import
Uses add_short_leg + manual RollRecord creation to bypass roll_short constraints.
Run from ~/vix_suite/
"""
import sys
from datetime import datetime
sys.path.insert(0, ".")

POS = {
    "V1": "V1-20260112111933",
    "V2": "V2-20260120161227",
    "V3": "V3-20260112123803",
    "V4": "V4-20260112123937",
    "V5": "V5-20260112124054",
}

# Full roll sequence — each entry closes old short and opens new one
# (roll_date, variant, buyback, new_strike, new_expiry, new_credit, contracts)
ROLLS = [
    ("2026-02-27","V3", 0.210, 45.0,"2026-03-06", 1.500, 5,"27FEB→06MAR"),
    ("2026-03-05","V1", 2.100, 47.0,"2026-03-13", 2.516, 5,"06MAR→13MAR 47C"),
    ("2026-03-05","V2", 1.908, 49.0,"2026-03-13", 2.060, 5,"06MAR→13MAR 49C"),
    ("2026-03-05","V3", 1.503, 48.0,"2026-03-06", 0.573, 5,"06MAR→06MAR 48C"),
    ("2026-03-05","V4", 2.100, 47.0,"2026-03-13", 2.516, 5,"06MAR→13MAR 47C"),
    ("2026-03-05","V5", 1.503, 49.0,"2026-03-06", 0.415, 5,"06MAR→06MAR 49C"),
    ("2026-03-13","V1", 0.0,  52.0,"2026-03-20", 4.000, 5,"13MAR→20MAR 52C"),
    ("2026-03-13","V4", 0.0,  52.0,"2026-03-20", 4.000, 5,"13MAR→20MAR 52C"),
    ("2026-03-20","V1", 2.250,55.0,"2026-03-27", 3.725, 7,"20MAR→27MAR 55C"),
    ("2026-03-20","V3", 0.0,  55.0,"2026-03-27", 3.725, 7,"20MAR→27MAR 55C"),
    ("2026-03-20","V2", 0.0,  56.0,"2026-03-27", 3.200, 8,"20MAR→27MAR 56C"),
    ("2026-03-20","V4", 2.250,57.0,"2026-03-27", 2.900, 5,"20MAR→27MAR 57C"),
    ("2026-03-20","V5", 0.0,  59.0,"2026-03-27", 2.170, 5,"20MAR→27MAR 59C"),
]


def main():
    from trade_log import get_trade_log, ShortLeg, RollRecord
    tl = get_trade_log()
    print("\n=== Paper Trade Import ===\n")
    ok = 0

    for (roll_date, variant, buyback, new_strike,
         new_expiry, new_credit, contracts, notes) in ROLLS:

        pid = POS[variant]
        pos = tl.diagonal_positions.get(pid)
        if not pos:
            print(f"❌  {variant} ({pid}): not found"); continue

        # Check if already recorded
        existing = [l for l in pos.short_legs
                    if l.expiration_date == new_expiry
                    and float(l.strike) == new_strike]
        if existing:
            print(f"⏭  {variant} {roll_date}: ${new_strike}C {new_expiry} already exists")
            continue

        # Close current open short if any (mark as rolled)
        for leg in pos.short_legs:
            if leg.status == "open":
                leg.status = "rolled"
                leg.exit_date = roll_date
                leg.exit_price = buyback
                leg.exit_reason = "rolled"

        # Add new short leg directly
        leg_num = len(pos.short_legs) + 1
        new_leg = ShortLeg(
            leg_id=f"{pid}-S{leg_num}",
            position_id=pid,
            entry_date=roll_date,
            strike=new_strike,
            expiration_date=new_expiry,
            entry_credit=new_credit,
            contracts=contracts,
            status="open",
        )
        pos.short_legs.append(new_leg)
        pos.total_short_credits += new_credit * contracts

        # Add roll record
        roll_credit = new_credit - buyback
        roll_num = len(pos.roll_history) + 1
        roll = RollRecord(
            roll_id=f"{pid}-R{roll_num}",
            position_id=pid,
            roll_date=roll_date,
            old_strike=0.0,
            old_expiration="",
            old_exit_price=buyback,
            new_strike=new_strike,
            new_expiration=new_expiry,
            new_credit=new_credit,
            roll_credit=roll_credit,
            underlying_price=0.0,
            contracts=contracts,
            regime="EXTREME",
            notes=f"{roll_date} | {notes}",
        )
        pos.roll_history.append(roll)
        pos.total_rolls += 1
        pos.total_roll_credits += roll_credit * contracts

        print(f"✅  {variant} {roll_date}: ${new_strike}C exp {new_expiry} "
              f"@ ${new_credit:.3f} net ${roll_credit:+.3f} | {notes}")
        ok += 1

    tl._save()
    print(f"\n✅ {ok}/{len(ROLLS)} rolls recorded.")
    print("Restart Streamlit to see changes.")


if __name__ == "__main__":
    main()
