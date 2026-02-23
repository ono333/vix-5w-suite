#!/usr/bin/env python3
"""
Fix missing roll records by inferring from short_legs history.

Based on analysis:
- V1: 5 short legs, 1 roll record → Missing 3 rolls (S1→S2, S2→S3, S3→S4)
- V2: 3 short legs, 2 roll records → Missing 1 roll (S1→S2)
- V3: 4 short legs, 4 roll records → OK (3 short + 1 long)
- V4: 5 short legs, 4 roll records → Missing 1 roll (S1→S2)
- V5: 4 short legs, 3 roll records → OK

Run this on Ubuntu:
    cd ~/vix_suite
    python3 fix_missing_rolls.py
"""

import json
from pathlib import Path
from datetime import datetime

TRADE_LOG_PATH = Path.home() / ".vix_suite" / "trade_log.json"

def load_trade_log():
    with open(TRADE_LOG_PATH, 'r') as f:
        return json.load(f)

def save_trade_log(data):
    # Backup first
    backup_path = TRADE_LOG_PATH.parent / f"trade_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, 'w') as f:
        json.dump(load_trade_log(), f, indent=2)
    print(f"✅ Backup saved to {backup_path}")
    
    # Save updated
    with open(TRADE_LOG_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated {TRADE_LOG_PATH}")

def find_missing_rolls(pos_data):
    """
    Compare short_legs transitions with roll_history to find missing rolls.
    Returns list of missing roll info.
    """
    short_legs = pos_data.get("short_legs", [])
    roll_history = pos_data.get("roll_history", [])
    position_id = pos_data["position_id"]
    contracts = pos_data.get("contracts", 5)
    
    if len(short_legs) < 2:
        return []
    
    # Build a set of existing roll transitions (old_strike, old_exp) -> (new_strike, new_exp)
    existing_transitions = set()
    for roll in roll_history:
        if roll.get("roll_type", "short") == "short":
            key = (
                roll.get("old_strike"),
                roll.get("old_expiration"),
                roll.get("new_strike"),
                roll.get("new_expiration"),
            )
            existing_transitions.add(key)
    
    # Find missing transitions
    missing = []
    for i in range(len(short_legs) - 1):
        old_leg = short_legs[i]
        new_leg = short_legs[i + 1]
        
        transition_key = (
            old_leg.get("strike"),
            old_leg.get("expiration_date"),
            new_leg.get("strike"),
            new_leg.get("expiration_date"),
        )
        
        if transition_key not in existing_transitions:
            # This transition is missing from roll_history
            buyback = old_leg.get("exit_price", 0.0) or 0.0
            new_credit = new_leg.get("entry_credit", 0.0)
            
            missing.append({
                "index": i + 1,
                "roll_date": new_leg.get("entry_date", ""),
                "old_strike": old_leg.get("strike", 0.0),
                "old_expiration": old_leg.get("expiration_date", ""),
                "old_exit_price": buyback,
                "new_strike": new_leg.get("strike", 0.0),
                "new_expiration": new_leg.get("expiration_date", ""),
                "new_credit": new_credit,
                "roll_credit": new_credit - buyback,
                "contracts": contracts,
            })
    
    return missing

def add_missing_rolls(pos_data, missing_rolls):
    """Add missing roll records to a position."""
    position_id = pos_data["position_id"]
    roll_history = pos_data.get("roll_history", [])
    contracts = pos_data.get("contracts", 5)
    
    for missing in missing_rolls:
        new_roll = {
            "roll_id": f"{position_id}-R{len(roll_history) + 1}",  # Temporary ID
            "position_id": position_id,
            "roll_date": missing["roll_date"],
            "old_strike": missing["old_strike"],
            "old_expiration": missing["old_expiration"],
            "old_exit_price": missing["old_exit_price"],
            "new_strike": missing["new_strike"],
            "new_expiration": missing["new_expiration"],
            "new_credit": missing["new_credit"],
            "roll_credit": missing["roll_credit"],
            "underlying_price": 0.0,
            "contracts": missing["contracts"],
            "roll_type": "short",
            "regime": pos_data.get("entry_regime", ""),
            "notes": "Auto-inferred from short_legs history",
        }
        roll_history.append(new_roll)
    
    # Sort by date
    roll_history.sort(key=lambda r: r.get("roll_date", ""))
    
    # Renumber roll IDs sequentially
    short_roll_num = 0
    long_roll_num = 0
    for roll in roll_history:
        if roll.get("roll_type", "short") == "long":
            long_roll_num += 1
            roll["roll_id"] = f"{position_id}-RL{long_roll_num}"
        else:
            short_roll_num += 1
            roll["roll_id"] = f"{position_id}-R{short_roll_num}"
    
    pos_data["roll_history"] = roll_history
    
    # Recalculate totals (only count short rolls)
    total_rolls = 0
    total_credits = 0.0
    for roll in roll_history:
        if roll.get("roll_type", "short") == "short":
            total_rolls += 1
            roll_contracts = roll.get("contracts", contracts)
            total_credits += roll.get("roll_credit", 0) * roll_contracts
    
    pos_data["total_rolls"] = total_rolls
    pos_data["total_roll_credits"] = total_credits
    
    return len(missing_rolls)

def analyze_position(pos_id, pos_data):
    """Analyze and report on a position's roll history."""
    short_legs = pos_data.get("short_legs", [])
    roll_history = pos_data.get("roll_history", [])
    
    # Count short vs long rolls
    short_rolls = [r for r in roll_history if r.get("roll_type", "short") == "short"]
    long_rolls = [r for r in roll_history if r.get("roll_type") == "long"]
    
    expected_short_rolls = len(short_legs) - 1
    
    print(f"\n{'='*60}")
    print(f"📊 {pos_id}")
    print(f"{'='*60}")
    print(f"   Short legs:          {len(short_legs)}")
    print(f"   Expected short rolls: {expected_short_rolls}")
    print(f"   Actual short rolls:   {len(short_rolls)}")
    print(f"   Long rolls:           {len(long_rolls)}")
    print(f"   total_rolls field:    {pos_data.get('total_rolls', 0)}")
    
    # Find missing rolls
    missing = find_missing_rolls(pos_data)
    
    if missing:
        print(f"\n   ⚠️  Missing {len(missing)} roll(s):")
        for m in missing:
            print(f"      • {m['roll_date']}: ${m['old_strike']} → ${m['new_strike']}, net ${m['roll_credit']:.2f}")
    else:
        print(f"\n   ✅ All rolls accounted for")
    
    return missing

def main():
    print("=" * 60)
    print("Fix Missing Roll Records - All Variants")
    print("=" * 60)
    
    data = load_trade_log()
    diagonal_positions = data.get("diagonal_positions", {})
    
    if not diagonal_positions:
        print("No diagonal positions found.")
        return
    
    # Analyze all positions first
    all_missing = {}
    for pos_id, pos_data in diagonal_positions.items():
        missing = analyze_position(pos_id, pos_data)
        if missing:
            all_missing[pos_id] = missing
    
    if not all_missing:
        print("\n" + "=" * 60)
        print("✅ All positions have complete roll history!")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_missing = sum(len(m) for m in all_missing.values())
    print(f"Found {total_missing} missing roll(s) across {len(all_missing)} position(s):")
    for pos_id, missing in all_missing.items():
        print(f"   • {pos_id}: {len(missing)} missing")
    
    # Confirm before making changes
    print("\n" + "-" * 60)
    confirm = input("Add missing roll records? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("❌ Cancelled. No changes made.")
        return
    
    # Add missing rolls
    print("\nAdding missing rolls...")
    for pos_id, missing in all_missing.items():
        pos_data = diagonal_positions[pos_id]
        added = add_missing_rolls(pos_data, missing)
        print(f"   ✅ {pos_id}: Added {added} roll(s)")
        print(f"      New totals: {pos_data['total_rolls']} rolls, ${pos_data['total_roll_credits']:.2f} credits")
    
    # Save
    data["updated_at"] = datetime.now().isoformat()
    save_trade_log(data)
    
    print("\n" + "=" * 60)
    print("✅ Done! Restart the app to see changes:")
    print("   sudo systemctl restart streamlit-app")
    print("=" * 60)

if __name__ == "__main__":
    main()
