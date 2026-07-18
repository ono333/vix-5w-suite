"""
Patch: daily_signal.py — fix fresh signal long/short strikes
Root cause: offsets calibrated for vix_level not uvxy_price.
Fix: long always ITM (uvxy - offset), short always OTM (uvxy + offset)
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

VARIANT_PARAMS = """
        _v_strike_params = {
            "v1_income_harvester": {"long_itm": 5.0, "short_otm": 2.0},
            "v2_mean_reversion":   {"long_itm": 3.0, "short_otm": 3.0},
            "v3_shock_absorber":   {"long_itm": 0.0, "short_otm": 4.0},
            "v4_tail_hunter":      {"long_itm": 5.0, "short_otm": 5.0},
            "v5_regime_allocator": {"long_itm": 3.0, "short_otm": 2.5},
        }
        _v_role = getattr(variant.role, "value", str(variant.role)).lower()
        _vp = _v_strike_params.get(_v_role, {"long_itm": 3.0, "short_otm": 3.0})
        _fresh_long_k  = round(uvxy_price - _vp["long_itm"])
        _fresh_short_k = round(uvxy_price + _vp["short_otm"])"""

OLD1 = "        short_offset = getattr(variant, 'short_strike_offset', 2)\n        target = round(uvxy_price + short_offset, 0)"
NEW1 = "        short_offset = getattr(variant, 'short_strike_offset', 2)\n        target = round(uvxy_price + short_offset, 0)" + VARIANT_PARAMS

if OLD1 in src:
    src = src.replace(OLD1, NEW1)
    print("Fix 1 applied")
else:
    OLD1B = "        short_offset = getattr(variant, 'short_strike_offset', 2)\n        target = round(vix_level + short_offset, 0)"
    if OLD1B in src:
        src = src.replace(OLD1B, NEW1)
        print("Fix 1 applied (vix_level variant)")
    else:
        print("Fix 1 not found")

for old2, label in [
    ('          <td>${variant.long_strike:.0f} exp {long_exp_str} ({variant.long_dte_weeks}w)</td>\n          <td style="padding:3px 0;color:#555;">Short:</td>\n          <td>${target:.0f} exp {short_exp_str}</td>', "original"),
    ('          <td>${round(uvxy_price + getattr(variant, \'long_strike_offset\', 0)):.0f} exp {long_exp_str} ({variant.long_dte_weeks}w)</td>\n          <td style="padding:3px 0;color:#555;">Short:</td>\n          <td>${target:.0f} exp {short_exp_str}</td>', "partial patch"),
]:
    new2 = '          <td>${_fresh_long_k:.0f} exp {long_exp_str} ({variant.long_dte_weeks}w)</td>\n          <td style="padding:3px 0;color:#555;">Short:</td>\n          <td>${_fresh_short_k:.0f} exp {short_exp_str}</td>'
    if old2 in src:
        src = src.replace(old2, new2)
        print(f"Fix 2 applied ({label})")
        break
else:
    print("Fix 2 not found")

TARGET.write_text(src)

print("\nVerification at UVXY $51.17:")
params = [
    ("V1 Income Harvester",  5.0, 2.0),
    ("V2 Mean Reversion",    3.0, 3.0),
    ("V3 Shock Absorber",    0.0, 4.0),
    ("V4 Tail Hunter",       5.0, 5.0),
    ("V5 Regime Allocator",  3.0, 2.5),
]
uvxy = 51.17
for name, itm, otm in params:
    lk = round(uvxy - itm)
    sk = round(uvxy + otm)
    ok = "OK" if lk < sk else "INVERTED"
    print(f"  {name:28} Long ${lk} / Short ${sk}  {ok}")
print("Done")
