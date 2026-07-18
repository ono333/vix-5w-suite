#!/usr/bin/env python3
"""
Test script to verify Polygon.io API connection and historical options data access.

Your API key in keychain (MASSIVE_API_KEY) should be a Polygon.io API key.

Run this to diagnose if your API key has historical options aggregates access.

Usage:
    python test_massive_connection.py
"""

import os
import sys
import platform
import subprocess
from datetime import datetime, timedelta


def get_api_key_from_keychain():
    """Try to get API key from macOS Keychain."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "MASSIVE_API_KEY", "-a", "MASSIVE_API_KEY", "-w"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            print(f"Keychain access failed: {e}")
    
    return os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")


def test_connection():
    print("=" * 60)
    print("Polygon.io API Connection Test")
    print("=" * 60)
    
    # Step 1: Get API key
    print("\n1. Checking API key...")
    api_key = get_api_key_from_keychain()
    
    if not api_key:
        print("   ❌ No API key found!")
        print("   Add to keychain: security add-generic-password -s MASSIVE_API_KEY -a MASSIVE_API_KEY -w 'your-key'")
        print("   Or set env var: export POLYGON_API_KEY='your-key'")
        return False
    
    print(f"   ✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Step 2: Import polygon client
    print("\n2. Importing polygon-api-client...")
    try:
        from polygon import RESTClient
        print("   ✅ polygon-api-client imported successfully")
    except ImportError:
        print("   ❌ polygon-api-client not installed!")
        print("   Run: pip install polygon-api-client")
        return False
    
    # Step 3: Create client
    print("\n3. Creating REST client...")
    try:
        client = RESTClient(api_key)
        print("   ✅ Client created")
    except Exception as e:
        print(f"   ❌ Failed to create client: {e}")
        return False
    
    # Step 4: Test basic stock data first (to verify API key works)
    print("\n4. Testing basic stock data access...")
    try:
        # Get SPY aggregates as basic test
        aggs = client.get_aggs("SPY", 1, "day", "2024-01-02", "2024-01-03")
        if aggs:
            print(f"   ✅ Basic data access works! Got {len(aggs)} bar(s)")
            if len(aggs) > 0:
                bar = aggs[0]
                print(f"   Sample bar: O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}")
        else:
            print("   ⚠️ No data returned for SPY test")
    except Exception as e:
        print(f"   ❌ Basic data test failed: {e}")
    
    # Step 5: Test options data
    print("\n5. Testing historical OPTIONS data access...")
    
    # Use a date range where we know UVXY options exist
    test_date = "2024-06-01"  # A known date with options data
    
    # Try UVXY options contracts
    test_tickers = [
        "O:UVXY240621C00010000",  # UVXY June 21 2024 $10 Call
        "O:UVXY240621C00015000",  # UVXY June 21 2024 $15 Call
        "O:SPY240621C00500000",   # SPY June 21 2024 $500 Call (more liquid)
    ]
    
    print(f"   Test date: {test_date}")
    
    success_count = 0
    for ticker in test_tickers:
        print(f"\n   Testing: {ticker}")
        try:
            # Use get_aggs with date strings (Polygon v1 API)
            aggs = client.get_aggs(
                ticker,
                1,  # multiplier
                "day",  # timespan
                test_date,  # from
                "2024-06-02"  # to
            )
            
            if aggs and len(aggs) > 0:
                bar = aggs[0]
                print(f"   ✅ Got data! Close=${bar.close:.2f}, Volume={bar.volume}")
                success_count += 1
            else:
                print("   ⚠️ No data returned - contract may not exist or no trades")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 6: Test list_aggs method (alternative API)
    print("\n6. Testing list_aggs method...")
    try:
        from_ts = int(datetime(2024, 6, 1).timestamp() * 1000)
        to_ts = int(datetime(2024, 6, 2).timestamp() * 1000)
        
        aggs = client.list_aggs(
            "O:SPY240621C00500000",
            1,
            "day",
            from_=from_ts,
            to=to_ts
        )
        
        # list_aggs returns a generator, convert to list
        aggs_list = list(aggs) if hasattr(aggs, '__iter__') else []
        
        if aggs_list:
            print(f"   ✅ list_aggs works! Got {len(aggs_list)} result(s)")
            bar = aggs_list[0]
            if hasattr(bar, 'close'):
                print(f"   Close: ${bar.close:.2f}")
            elif isinstance(bar, dict) and 'c' in bar:
                print(f"   Close: ${bar['c']:.2f}")
            success_count += 1
        else:
            print("   ⚠️ list_aggs returned no results")
            
    except Exception as e:
        print(f"   ❌ list_aggs error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    if success_count > 0:
        print(f"✅ SUCCESS: Retrieved {success_count} option prices from Polygon")
        print("Historical options data access confirmed!")
    else:
        print("❌ FAILED: No historical options prices retrieved")
        print("\nPossible issues:")
        print("1. API tier doesn't include Options data (requires paid plan)")
        print("2. Contract ticker format incorrect")
        print("3. Date range has no available data")
        print("\nPolygon.io Options data requires:")
        print("- Starter plan ($29/mo) for delayed options")
        print("- Developer plan ($79/mo) for real-time options")
        print("\nCheck: https://polygon.io/pricing")
    print("=" * 60)
    
    return success_count > 0


if __name__ == "__main__":
    test_connection()
