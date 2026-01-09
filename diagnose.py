#!/usr/bin/env python3
"""
Diagnostic script for VIX 5% Weekly Suite
Run from your project root: python diagnose.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("VIX 5% Weekly Suite - Diagnostic Report")
print("=" * 60)

# 1. Check for bad imports in all .py files
print("\n[1] Checking for invalid imports (01_vix_5w_suite.*)...")
bad_imports = []
for py_file in Path(".").glob("**/*.py"):
    if "__pycache__" in str(py_file):
        continue
    try:
        content = py_file.read_text()
        if "from 01_vix_5w_suite" in content or "import 01_vix_5w_suite" in content:
            bad_imports.append(str(py_file))
    except:
        pass

if bad_imports:
    print(f"   ❌ FOUND {len(bad_imports)} files with bad imports:")
    for f in bad_imports:
        print(f"      - {f}")
else:
    print("   ✅ No bad imports found")

# 2. Check required modules exist
print("\n[2] Checking required modules exist...")
required = [
    "enums.py",
    "regime_detector.py", 
    "variant_generator.py",
    "robustness_scorer.py",
    "trade_log.py",
    "exit_detector.py",
    "notification_engine.py",
]
missing = [f for f in required if not Path(f).exists()]
if missing:
    print(f"   ❌ MISSING {len(missing)} modules:")
    for f in missing:
        print(f"      - {f}")
else:
    print("   ✅ All required modules exist")

# 3. Try imports
print("\n[3] Testing imports...")
imports_ok = True

try:
    from enums import VolatilityRegime, VariantRole, TradeStatus
    print("   ✅ enums")
except Exception as e:
    print(f"   ❌ enums: {e}")
    imports_ok = False

try:
    from regime_detector import classify_regime, RegimeState, get_regime_color
    print("   ✅ regime_detector")
except Exception as e:
    print(f"   ❌ regime_detector: {e}")
    imports_ok = False

try:
    from variant_generator import generate_all_variants, SignalBatch, VariantParams
    print("   ✅ variant_generator")
except Exception as e:
    print(f"   ❌ variant_generator: {e}")
    imports_ok = False

try:
    from robustness_scorer import calculate_robustness, RobustnessResult
    print("   ✅ robustness_scorer")
except Exception as e:
    print(f"   ❌ robustness_scorer: {e}")
    imports_ok = False

try:
    from trade_log import TradeLog, get_trade_log, Position
    print("   ✅ trade_log")
except Exception as e:
    print(f"   ❌ trade_log: {e}")
    imports_ok = False

try:
    from exit_detector import detect_all_exits, ExitEvent, get_exit_store
    print("   ✅ exit_detector")
except Exception as e:
    print(f"   ❌ exit_detector: {e}")
    imports_ok = False

try:
    from notification_engine import get_notifier
    print("   ✅ notification_engine")
except Exception as e:
    print(f"   ❌ notification_engine: {e}")
    imports_ok = False

# 4. Test TradeLog.get_summary()
print("\n[4] Testing TradeLog.get_summary()...")
try:
    tl = get_trade_log()
    summary = tl.get_summary()
    print(f"   ✅ get_summary() works: {len(summary)} keys")
except Exception as e:
    print(f"   ❌ get_summary() failed: {e}")

# 5. Test signal generation
print("\n[5] Testing signal generation...")
try:
    import pandas as pd
    # Create dummy data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=52, freq='W')
    dummy_series = pd.Series([35.0 + i*0.1 for i in range(52)], index=dates)
    
    regime = classify_regime(dummy_series)
    print(f"   ✅ classify_regime(): {regime.regime.value}, VIX={regime.vix_level:.2f}, pct={regime.vix_percentile:.2%}")
    
    batch = generate_all_variants(dummy_series, regime)
    print(f"   ✅ generate_all_variants(): {len(batch.signals)} signals in batch")
    for sig in batch.signals:
        print(f"      - {sig.name}: {sig.position_type}, strike={sig.long_strike}")
except Exception as e:
    print(f"   ❌ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()

# 6. Check utils/regime_utils.py
print("\n[6] Checking utils/regime_utils.py...")
try:
    from utils.regime_utils import extract_current_regime
    print("   ✅ utils.regime_utils exists and imports")
except Exception as e:
    print(f"   ❌ utils.regime_utils: {e}")

print("\n" + "=" * 60)
if imports_ok and not bad_imports and not missing:
    print("✅ All checks passed - modules should work")
else:
    print("❌ Issues found - see above")
print("=" * 60)
