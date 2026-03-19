#!/usr/bin/env python3
"""
fix_dte_999.py
Replaces all hardcoded 999 DTE fallbacks in real_trade_log.py with
a robust date parser that tries multiple formats before giving up.
Run from ~/vix_suite/
"""
import sys, shutil, re
from datetime import datetime, date
from pathlib import Path

TARGET = Path("real_trade_log.py")
if not TARGET.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

OLD = """        except Exception:
            long_dte  = 999"""

NEW = """        except Exception:
            # Fallback: try alternate date formats before giving up
            try:
                long_exp  = datetime.strptime(self.long_expiration, "%Y-%m-%d").date()
                long_dte  = (long_exp - today).days
            except Exception:
                try:
                    long_exp  = datetime.strptime(self.long_expiration, "%m/%d/%Y").date()
                    long_dte  = (long_exp - today).days
                except Exception:
                    long_dte  = 0   # unknown expiry — show 0d not 999d"""

src = TARGET.read_text()
count = src.count(OLD)
if count == 0:
    print("ERROR: pattern not found — may already be patched or indentation differs")
    # Show context around 999
    for i, line in enumerate(src.splitlines(), 1):
        if "999" in line and "long_dte" in line:
            print(f"  line {i}: {repr(line)}")
    sys.exit(1)

backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

patched = src.replace(OLD, NEW)
TARGET.write_text(patched)
print(f"✅ Fixed {count} occurrence(s) of long_dte=999 in real_trade_log.py")
print("   Restart Streamlit to apply.")
