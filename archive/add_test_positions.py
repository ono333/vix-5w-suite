#!/usr/bin/env python3
"""
Test script to add sample positions to the trade log.
This demonstrates the management mode in emails.

Run: python3 add_test_positions.py
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_log import get_trade_log, Position


def main():
    print("📝 Adding test positions to trade log...\n")
    
    trade_log = get_trade_log()
    
    # Show current state
    summary = trade_log.get_summary()
    print(f"Current state: {summary['open_positions']} open positions")
    
    # Add V1 Income Harvester position (in profit)
    if not trade_log.has_open_position("V1_INCOME_HARVESTER"):
        pos = trade_log.open_position(
            variant_id="V1_INCOME_HARVESTER",
            variant_name="V1 Income Harvester",
            entry_price=1.85,  # Credit received
            entry_regime="CALM",
            entry_vix_level=32.50,
            entry_percentile=0.05,
            strike=38.0,
            expiration_date=(datetime.now() + timedelta(days=23)).strftime("%Y-%m-%d"),
            contracts=5,
            allocation_pct=2.0,
            allocation_dollars=5000.0,
            target_pct=0.40,
            stop_pct=0.60,
        )
        # Simulate current P&L (in profit - price dropped)
        pos.update_pnl(1.25)  # Current price lower = profit for short premium
        print(f"✅ Added V1 Income Harvester position")
        print(f"   Entry: ${pos.entry_price:.2f} credit")
        print(f"   Target: ${pos.target_price:.2f}")
        print(f"   Stop: ${pos.stop_price:.2f}")
        print(f"   Current P&L: ${pos.current_pnl:+,.0f} ({pos.current_pnl_pct:.1%})")
    else:
        print("⚠️ V1 Income Harvester already has position")
    
    print()
    
    # Add V5 Regime Allocator position (slight loss)
    if not trade_log.has_open_position("V5_REGIME_ALLOCATOR"):
        pos = trade_log.open_position(
            variant_id="V5_REGIME_ALLOCATOR",
            variant_name="V5 Regime Allocator",
            entry_price=2.10,  # Credit received
            entry_regime="CALM",
            entry_vix_level=34.00,
            entry_percentile=0.03,
            strike=42.0,
            expiration_date=(datetime.now() + timedelta(days=16)).strftime("%Y-%m-%d"),
            contracts=4,
            allocation_pct=2.5,
            allocation_dollars=6250.0,
            target_pct=0.35,
            stop_pct=0.50,
        )
        # Simulate current P&L (slight loss - price went up)
        pos.update_pnl(2.35)  # Current price higher = loss for short premium
        print(f"✅ Added V5 Regime Allocator position")
        print(f"   Entry: ${pos.entry_price:.2f} credit")
        print(f"   Target: ${pos.target_price:.2f}")
        print(f"   Stop: ${pos.stop_price:.2f}")
        print(f"   Current P&L: ${pos.current_pnl:+,.0f} ({pos.current_pnl_pct:.1%})")
    else:
        print("⚠️ V5 Regime Allocator already has position")
    
    print()
    
    # Summary
    print("=" * 50)
    summary = trade_log.get_summary()
    print(f"📊 Trade Log Summary:")
    print(f"   Open positions: {summary['open_positions']}")
    print(f"   Open P&L: ${summary['open_pnl']:+,.0f}")
    print(f"   Total trades: {summary['total_trades']}")
    print(f"   Realized P&L: ${summary['total_realized_pnl']:+,.0f}")
    
    print()
    print("🔄 Open positions:")
    for pos in trade_log.get_all_open_positions():
        dte = pos.days_to_expiry()
        print(f"   • {pos.variant_name}")
        print(f"     P&L: ${pos.current_pnl:+,.0f} | DTE: {dte} days")
        print(f"     Target: ${pos.target_price:.2f} | Stop: ${pos.stop_price:.2f}")
    
    print()
    print(f"💾 Trade log saved to: {trade_log.storage_path}")
    print()
    print("Now run: python3 daily_signal.py --dry-run")
    print("You should see MANAGEMENT mode for V1 and V5!")


if __name__ == "__main__":
    main()
