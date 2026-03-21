#!/usr/bin/env python3
"""Fix add_short_leg missing entry_date param. Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

patch("trade_log.py",
    old='    def add_short_leg(self, strike: float, expiration: str, credit: float, contracts: Optional[int] = None) -> ShortLeg:',
    new='    def add_short_leg(self, strike: float, expiration: str, credit: float, contracts: Optional[int] = None, entry_date: str = "") -> ShortLeg:',
    description="Add missing entry_date param to add_short_leg",
)
print("Done. Run: python3 import_ib_trades.py 2>&1")
