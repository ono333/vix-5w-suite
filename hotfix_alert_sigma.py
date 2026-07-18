"""
Hotfix: alert_engine.py — sigma used before assignment
Move sigma calculation above the OTM band selector.
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "alert_engine.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

OLD = """\
    # Long leg: ~15% ITM LEAP (convexity anchor, lower strike than short)
    # Short leg: OTM % derived from VIX regime band
    if   sigma < 0.17:  _s_otm = 0.10   # calm: 10% OTM"""

NEW = """\
    # Sigma must be defined before OTM band selector
    sigma     = min(2.60, max(0.80, (ms.get("vix", 25) / 100) * 4.5))

    # Long leg: ~15% ITM LEAP (convexity anchor, lower strike than short)
    # Short leg: OTM % derived from VIX regime band
    if   sigma < 0.17:  _s_otm = 0.10   # calm: 10% OTM"""

if OLD in src:
    src = src.replace(OLD, NEW)
    print("✅ Hotfix applied: sigma moved before OTM band selector")
else:
    print("⚠️  Pattern not found — manual fix needed")
    print("    Find this line in alert_engine.py:")
    print("    'if   sigma < 0.17:  _s_otm = 0.10'")
    print("    Add above it: sigma = min(2.60, max(0.80, (ms.get('vix', 25) / 100) * 4.5))")

# Also remove the duplicate sigma assignment that now comes after
OLD2 = """\
    long_dte  = 90
    short_dte = 21
    sigma     = min(2.60, max(0.80, (ms.get("vix", 25) / 100) * 4.5))"""

NEW2 = """\
    long_dte  = 90
    short_dte = 21"""

if OLD2 in src:
    src = src.replace(OLD2, NEW2)
    print("✅ Duplicate sigma assignment removed")
else:
    print("⚠️  Duplicate sigma line not found — may already be clean")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
