#!/usr/bin/env python3
"""
Diagnostic script for VIX Suite trade log issues.
Run this on Ubuntu to check why trades aren't showing.
"""

import json
import os
from pathlib import Path

print("=" * 60)
print("VIX Suite Trade Log Diagnostic")
print("=" * 60)

# 1. Check expected path
expected_path = Path.home() / ".vix_suite" / "trade_log.json"
print(f"\n1. Expected path: {expected_path}")
print(f"   Exists: {expected_path.exists()}")

if expected_path.exists():
    stat = expected_path.stat()
    print(f"   Size: {stat.st_size} bytes")
    print(f"   Modified: {stat.st_mtime}")

# 2. Try to load the file
print("\n2. Loading trade_log.json...")
try:
    with open(expected_path, 'r') as f:
        data = json.load(f)
    
    print(f"   ✅ JSON loaded successfully")
    print(f"   Keys: {list(data.keys())}")
    
    # Check diagonal positions
    diagonals = data.get("diagonal_positions", {})
    print(f"\n3. Diagonal Positions: {len(diagonals)}")
    
    for pos_id, pos in diagonals.items():
        status = pos.get("status", "unknown")
        variant = pos.get("variant_name", "unknown")
        contracts = pos.get("contracts", 0)
        short_legs = pos.get("short_legs", [])
        open_shorts = [s for s in short_legs if s.get("status") == "open"]
        
        print(f"   - {pos_id}: {variant}")
        print(f"     Status: {status}, Contracts: {contracts}")
        print(f"     Short legs: {len(short_legs)} total, {len(open_shorts)} open")
    
    # Check positions (old format)
    positions = data.get("positions", {})
    print(f"\n4. Positions (old format): {len(positions)}")
    
    # Check history
    history = data.get("history", [])
    print(f"\n5. Trade History: {len(history)} records")

except json.JSONDecodeError as e:
    print(f"   ❌ JSON parse error: {e}")
except FileNotFoundError:
    print(f"   ❌ File not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check what the app's TradeLog class sees
print("\n" + "=" * 60)
print("Testing TradeLog class...")
print("=" * 60)

try:
    # Add current directory to path
    import sys
    sys.path.insert(0, os.getcwd())
    
    from trade_log import TradeLog, get_trade_log
    
    trade_log = get_trade_log()
    print(f"\n6. TradeLog storage path: {trade_log.storage_path}")
    
    all_diagonals = trade_log.get_all_diagonals()
    open_diagonals = trade_log.get_open_diagonals()
    
    print(f"   All diagonals: {len(all_diagonals)}")
    print(f"   Open diagonals: {len(open_diagonals)}")
    
    for pos in all_diagonals:
        print(f"   - {pos.position_id}: {pos.variant_name} ({pos.status})")

except ImportError as e:
    print(f"   ⚠️ Could not import trade_log module: {e}")
    print("   Run this from your vix_suite directory")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
