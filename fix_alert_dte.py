#!/usr/bin/env python3
"""Fix short_dte_val=999 fallback in alert_engine.py. Run from ~/vix_suite/"""
import sys, shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("alert_engine.py")
if not TARGET.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

OLD = "            short_dte_val = _dte(str(short_exp)) if short_exp else 999"
NEW = """            # Try to get DTE from short leg directly if current_short_expiration missing
            if short_exp:
                short_dte_val = _dte(str(short_exp))
            else:
                short = getattr(pos, "current_short_leg", None)
                short_exp2 = getattr(short, "expiration_date", None) if short else None
                short_dte_val = _dte(str(short_exp2)) if short_exp2 else 0"""

src = TARGET.read_text()
if OLD not in src:
    print("ERROR: pattern not found"); sys.exit(1)

backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

TARGET.write_text(src.replace(OLD, NEW, 1))
print("✅ Fixed short_dte_val=999 in alert_engine.py")
