#!/usr/bin/env python3
"""
Diagnostic script to check variant generation.
Run this to see exactly how many variants are being created.
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("🔍 VARIANT GENERATION DIAGNOSTIC")
print("=" * 60)

# Step 1: Check imports
print("\n1️⃣ Checking imports...")
try:
    from enums import VolatilityRegime, VariantRole
    print("   ✅ enums imported")
except Exception as e:
    print(f"   ❌ enums import failed: {e}")
    sys.exit(1)

try:
    from regime_detector import classify_regime, RegimeState
    print("   ✅ regime_detector imported")
except Exception as e:
    print(f"   ❌ regime_detector import failed: {e}")
    sys.exit(1)

try:
    from variant_generator import generate_all_variants, get_variant_display_name
    print("   ✅ variant_generator imported")
except Exception as e:
    print(f"   ❌ variant_generator import failed: {e}")
    sys.exit(1)

# Step 2: Create a test regime
print("\n2️⃣ Creating test regime (CALM)...")
try:
    regime = RegimeState(
        regime=VolatilityRegime.CALM,
        vix_level=35.0,
        vix_percentile=0.02,
        confidence=0.7,
    )
    print(f"   ✅ Regime: {regime.regime.value.upper()}")
    print(f"   ✅ VIX Level: {regime.vix_level}")
    print(f"   ✅ Percentile: {regime.vix_percentile:.0%}")
except Exception as e:
    print(f"   ❌ RegimeState creation failed: {e}")
    sys.exit(1)

# Step 3: Generate variants
print("\n3️⃣ Generating variants...")
try:
    batch = generate_all_variants(regime)
    print(f"   ✅ Batch created: {batch.batch_id}")
    print(f"   ✅ Number of variants: {len(batch.variants)}")
except Exception as e:
    print(f"   ❌ generate_all_variants failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: List all variants
print("\n4️⃣ ALL GENERATED VARIANTS:")
print("-" * 60)
for i, v in enumerate(batch.variants, 1):
    is_active = regime.regime in v.active_in_regimes
    status = "🟢 RECOMMENDED" if is_active else "🔵 PAPER TEST"
    active_in = ", ".join([r.value.upper() for r in v.active_in_regimes])
    print(f"   {i}. {get_variant_display_name(v.role)}")
    print(f"      Status: {status}")
    print(f"      Active in: {active_in}")
    print(f"      Entry: ≤{v.entry_percentile:.0%} | OTM: +{v.long_strike_offset}pts | DTE: {v.long_dte_weeks}w")
    print()

# Step 5: Summary
print("=" * 60)
print("📊 SUMMARY")
print("=" * 60)
total = len(batch.variants)
recommended = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
paper_test = total - recommended

print(f"   Total variants generated: {total}")
print(f"   🟢 RECOMMENDED (would trade live): {recommended}")
print(f"   🔵 PAPER TEST (observe only): {paper_test}")

if total == 5:
    print("\n   ✅ CORRECT: All 5 variants generated!")
else:
    print(f"\n   ❌ BUG: Expected 5 variants, got {total}")
    print("      Check variant_generator.py for filtering logic")

print("\n" + "=" * 60)
