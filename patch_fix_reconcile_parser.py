#!/usr/bin/env python3
"""
patch_fix_reconcile_parser.py
─────────────────────────────
Fixes parse_fidelity_csv() in reconcile.py to handle null/empty rows
and the Fidelity footer disclaimer rows.

Deploy:
    scp ~/Downloads/patch_fix_reconcile_parser.py shin@192.168.100.142:~/vix_suite/
    cd ~/vix_suite && source venv/bin/activate
    python3 patch_fix_reconcile_parser.py
"""

import sys
from pathlib import Path

RECONCILE_PATH = Path(__file__).parent / "reconcile.py"

OLD = '''def parse_fidelity_csv(csv_path: str) -> dict:
    """
    Returns dict of actual open positions keyed by (underlying, expiry, strike, cp).
    Only includes UVXY options.
    """
    import csv
    positions = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym_raw = row.get("Symbol", "").strip().lstrip("-").strip()
            m = TICKER_RE.match(sym_raw)
            if not m or m.group("sym") != "UVXY":
                continue

            yr     = 2000 + int(m.group("yr"))
            mo     = int(m.group("mo"))
            dy     = int(m.group("day"))
            expiry = date(yr, mo, dy).isoformat()
            strike = float(m.group("strike"))
            cp     = m.group("cp")
            qty    = int(_clean(row.get("Quantity", 0)))
            cost   = _clean(row.get("Average Cost Basis", 0))
            last   = _clean(row.get("Last Price", 0))
            pnl    = _clean(row.get("Total Gain/Loss Dollar", 0))

            key = (expiry, strike, cp)
            positions[key] = {
                "expiry":    expiry,
                "strike":    strike,
                "cp":        cp,
                "qty":       qty,          # negative = short, positive = long
                "avg_cost":  cost,
                "last":      last,
                "total_pnl": pnl,
                "is_short":  qty < 0,
                "contracts": abs(qty),
            }
    return positions'''

NEW = '''def parse_fidelity_csv(csv_path: str) -> dict:
    """
    Returns dict of actual open positions keyed by (underlying, expiry, strike, cp).
    Only includes UVXY options. Skips null/empty/footer rows.
    """
    import csv
    positions = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows where Symbol is None or empty (footer/disclaimer rows)
            sym_raw = row.get("Symbol")
            if not sym_raw or not isinstance(sym_raw, str):
                continue
            sym_raw = sym_raw.strip().lstrip("-").strip()
            if not sym_raw:
                continue

            m = TICKER_RE.match(sym_raw)
            if not m or m.group("sym") != "UVXY":
                continue

            # Skip rows with no meaningful quantity
            qty_raw = row.get("Quantity")
            if not qty_raw or str(qty_raw).strip() in ("", "--", "None"):
                continue

            yr     = 2000 + int(m.group("yr"))
            mo     = int(m.group("mo"))
            dy     = int(m.group("day"))
            expiry = date(yr, mo, dy).isoformat()
            strike = float(m.group("strike"))
            cp     = m.group("cp")
            qty    = int(_clean(qty_raw))
            cost   = _clean(row.get("Average Cost Basis", 0))
            last   = _clean(row.get("Last Price", 0))
            pnl    = _clean(row.get("Total Gain/Loss Dollar", 0))

            key = (expiry, strike, cp)
            positions[key] = {
                "expiry":    expiry,
                "strike":    strike,
                "cp":        cp,
                "qty":       qty,          # negative = short, positive = long
                "avg_cost":  cost,
                "last":      last,
                "total_pnl": pnl,
                "is_short":  qty < 0,
                "contracts": abs(qty),
            }
    return positions'''


def main():
    src = RECONCILE_PATH.read_text(encoding="utf-8")

    if OLD not in src:
        print("❌ Target function not found in reconcile.py — may already be patched.")
        sys.exit(1)

    patched = src.replace(OLD, NEW, 1)

    backup = RECONCILE_PATH.with_suffix(".py.bak_parser")
    RECONCILE_PATH.rename(backup)
    print(f"📦 Backed up to {backup}")

    RECONCILE_PATH.write_text(patched, encoding="utf-8")
    print("✅ reconcile.py parser fixed — null/empty rows now skipped.")
    print("🔄 No Streamlit restart needed (reconcile.py is imported at runtime).")


if __name__ == "__main__":
    main()
