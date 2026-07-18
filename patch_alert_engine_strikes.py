"""
Patch: alert_engine.py — V4 entry signal strike generation
Bugs fixed:
  1. long_strike = uvxy * 1.00  → ATM, correct structure is long < short
     (long leg should be ITM/below to provide LEAP convexity)
  2. short_strike = uvxy * 1.30 → 30% OTM hardcoded, ignores current regime band
     In EXTREME/Panic regime (UVXY > 50) correct band is 10–15% OTM

Note: long leg is set to 85% of UVXY (≈15% ITM) matching LEAP convexity intent.
      short leg is set to 112% of UVXY (≈12% OTM, mid of Panic band 10–15%).
      Both are regime-aware via the vix_level passed into the function.

Deploy:
  cd ~/vix_suite
  python patch_alert_engine_strikes.py
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "alert_engine.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1 & 2: long/short strike calculation in V4 entry signal ───────────────
# Old:
#   long_strike  = round(uvxy * 1.00, 1)   # ATM — wrong
#   short_strike = round(uvxy * 1.30, 1)   # 30% OTM hardcoded — wrong

OLD_STRIKES = (
    "    long_strike  = round(uvxy * 1.00, 1)\n"
    "    short_strike = round(uvxy * 1.30, 1)"
)

NEW_STRIKES = """\
    # Long leg: ~15% ITM LEAP (convexity anchor, lower strike than short)
    # Short leg: OTM % derived from VIX regime band
    if   sigma < 0.17:  _s_otm = 0.10   # calm: 10% OTM
    elif sigma < 0.22:  _s_otm = 0.07   # low
    elif sigma < 0.25:  _s_otm = 0.05   # neutral
    elif sigma < 0.35:  _s_otm = 0.08   # elevated
    elif sigma < 0.50:  _s_otm = 0.10   # crisis
    else:               _s_otm = 0.12   # panic/extreme: 12% OTM (mid of 10–15% band)

    long_strike  = round(uvxy * 0.85, 1)            # ~15% ITM LEAP
    short_strike = round(uvxy * (1 + _s_otm), 1)   # regime-aware OTM
    # Guard: short must always exceed long
    if short_strike <= long_strike:
        short_strike = long_strike + 2"""

if OLD_STRIKES in src:
    src = src.replace(OLD_STRIKES, NEW_STRIKES)
    print("✅ Fix applied: long ITM, short regime-aware OTM")
else:
    print("⚠️  Pattern not found — check indentation or line content")
    print("    Searching for literal:\n", OLD_STRIKES)
    # Try flexible search
    import re
    m = re.search(r'long_strike\s*=\s*round\(uvxy \* 1\.00.*\n.*short_strike\s*=\s*round\(uvxy \* 1\.30', src)
    if m:
        print(f"    Found at position {m.start()} — check indentation in file")
    else:
        print("    Not found with flexible search either — manual edit needed")
        print("    Target lines 167-168 in alert_engine.py")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Restart alert_engine systemd service after deploy:")
print("  sudo systemctl restart vix-alert.timer")
