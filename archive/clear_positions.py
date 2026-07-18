#!/usr/bin/env python3
"""
Clear all positions from trade log.
Use this to reset for testing.

Run: python3 clear_positions.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_log import get_trade_log


def main():
    trade_log = get_trade_log()
    
    # Show current state
    open_positions = trade_log.get_all_open_positions()
    print(f"📊 Current state: {len(open_positions)} open positions")
    
    if not open_positions:
        print("   No positions to clear.")
        return
    
    print("\nPositions to clear:")
    for pos in open_positions:
        print(f"   • {pos.variant_name} (P&L: ${pos.current_pnl:+,.0f})")
    
    confirm = input("\n⚠️ Clear all positions? [y/N]: ").strip().lower()
    
    if confirm == 'y':
        # Close all positions with 'manual_clear' reason
        for pos in open_positions:
            trade_log.close_position(
                variant_id=pos.variant_id,
                exit_price=pos.current_price,
                exit_reason="manual_clear",
                exit_regime="MANUAL",
            )
        print(f"\n✅ Cleared {len(open_positions)} positions (moved to history)")
    else:
        print("\n❌ Cancelled")


if __name__ == "__main__":
    main()
