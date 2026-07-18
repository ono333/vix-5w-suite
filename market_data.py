#!/usr/bin/env python3
"""
market_data.py — quote freshness/sanity guard + timestamped SQLite logger.

TWO responsibilities, one module (the chokepoint every quote passes through):

  validate_quote(quote, sandbox)  -> (ok: bool, reason: str)
      Guards against trading on bad data:
        - $0 bid            -> BLOCK  (both venues; unsellable)
        - wide spread       -> LOG-but-ALLOW for now (measure 2 wks, then enforce)
        - stale timestamp   -> BLOCK in LIVE only (sandbox bid_date is frozen ~15min)

  log_quote(...)                  -> None
      Appends one timestamped row per quote to ~/.vix_suite/market_data.db,
      recording bid/ask/mid/spread/age/context/decision — the self-refining
      dataset. context lets a 90-day prune keep decision-points, drop routine.

Design notes:
  - Timestamps: Tradier bid_date/ask_date are Unix MILLISECONDS.
  - Freshness is venue-aware (proven: sandbox bid_date frozen at ~901s while the
    bid value itself moves -> unusable as a freshness signal in sandbox).
  - Fails safe: a logging/DB error must NEVER break a trade (all wrapped).
"""
from __future__ import annotations
import os
import sqlite3
import datetime
from typing import Optional, Tuple

DB_PATH = os.path.expanduser("~/.vix_suite/market_data.db")

# ── Guard thresholds ────────────────────────────────────────────────────────
MIN_BID = 0.0                 # bid must be strictly greater than this to trade
WIDE_SPREAD_PCT = 0.60        # spread/mid above this = "wide" (LOGGED, not blocked yet)
STALE_AGE_SEC = 60            # live only: reject quotes older than this
ENFORCE_WIDE_SPREAD = False   # flip to True after ~2 weeks of data to start blocking


def _quote_age_sec(quote: dict) -> Optional[float]:
    """Age of the quote in seconds from bid_date (ms epoch). None if unavailable."""
    bd = quote.get("bid_date")
    if not bd:
        return None
    try:
        ts = int(bd) / 1000.0
        return (datetime.datetime.now() - datetime.datetime.fromtimestamp(ts)).total_seconds()
    except (ValueError, TypeError, OSError):
        return None


def _spread_pct(bid: float, ask: float) -> Optional[float]:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def validate_quote(quote: dict, sandbox: bool) -> Tuple[bool, str]:
    """Return (ok, reason). ok=False means DO NOT trade on this quote.
    reason is a short machine-loggable string (also used as the DB 'decision')."""
    try:
        bid = float(quote.get("bid", 0) or 0)
        ask = float(quote.get("ask", 0) or 0)
    except (ValueError, TypeError):
        return False, "rejected:unparseable_bid_ask"

    # 1. $0-bid — BLOCK, both venues (unsellable; can't sell into a vacuum)
    if bid <= MIN_BID:
        return False, "rejected:zero_bid"

    if ask <= 0:
        return False, "rejected:zero_ask"

    # 2. stale timestamp — BLOCK in LIVE only (sandbox bid_date is frozen/unusable)
    if not sandbox:
        age = _quote_age_sec(quote)
        if age is not None and age > STALE_AGE_SEC:
            return False, f"rejected:stale_{age:.0f}s"

    # 3. wide spread — LOG but ALLOW for now (measure, then enforce)
    sp = _spread_pct(bid, ask)
    if sp is not None and sp > WIDE_SPREAD_PCT:
        if ENFORCE_WIDE_SPREAD:
            return False, f"rejected:wide_spread_{sp:.0%}"
        return True, f"accepted:wide_spread_flagged_{sp:.0%}"

    return True, "accepted"


# ── SQLite logging ──────────────────────────────────────────────────────────

def _connect() -> Optional[sqlite3.Connection]:
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT,
                bid REAL, ask REAL, mid REAL,
                bid_date INTEGER, ask_date INTEGER,
                spread_pct REAL, age_sec REAL,
                context TEXT, decision TEXT,
                sandbox INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ctx ON quotes(context)")
        return conn
    except Exception:
        return None  # DB problems must never break trading


def log_quote(quote: dict, symbol: str, context: str, decision: str, sandbox: bool) -> None:
    """Append one row. Never raises — a logging failure must not stop a trade."""
    conn = _connect()
    if conn is None:
        return
    try:
        bid = float(quote.get("bid", 0) or 0)
        ask = float(quote.get("ask", 0) or 0)
        mid = round((bid + ask) / 2.0, 4) if (bid or ask) else None
        sp = _spread_pct(bid, ask)
        age = _quote_age_sec(quote)
        conn.execute(
            "INSERT INTO quotes (ts,symbol,bid,ask,mid,bid_date,ask_date,spread_pct,age_sec,context,decision,sandbox) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), symbol, bid, ask, mid,
             quote.get("bid_date"), quote.get("ask_date"),
             round(sp, 4) if sp is not None else None,
             round(age, 1) if age is not None else None,
             context, decision, 1 if sandbox else 0),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try: conn.close()
        except Exception: pass


def prune(keep_full_days: int = 90) -> int:
    """After keep_full_days, keep only decision-point rows (placement/reprice/fill);
    drop routine re-quotes. Returns rows deleted. Safe to call anytime."""
    conn = _connect()
    if conn is None:
        return 0
    try:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=keep_full_days)).isoformat()
        cur = conn.execute(
            "DELETE FROM quotes WHERE ts < ? AND context NOT IN ('placement','reprice','fill')",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount
    except Exception:
        return 0
    finally:
        try: conn.close()
        except Exception: pass


# ── convenience: validate + log in one call (the chokepoint) ────────────────

def check_and_log(quote: dict, symbol: str, context: str, sandbox: bool) -> Tuple[bool, str]:
    """Validate a quote AND log it with the decision. Returns (ok, reason).
    This is what the orchestrator calls at each get_quote site."""
    ok, reason = validate_quote(quote, sandbox)
    log_quote(quote, symbol, context, reason, sandbox)
    return ok, reason


if __name__ == "__main__":
    # Self-test against realistic quote shapes (incl. the real sandbox quote + the phantom).
    import json
    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    old_ms = now_ms - 901_000  # ~901s old, like the real sandbox quote

    cases = [
        ("real sandbox UVXY (901s old, tight)", {"bid":25.32,"ask":25.34,"bid_date":old_ms,"ask_date":old_ms}, True),
        ("same quote but LIVE (should reject stale)", {"bid":25.32,"ask":25.34,"bid_date":old_ms,"ask_date":old_ms}, False),
        ("fresh live quote", {"bid":0.65,"ask":0.70,"bid_date":now_ms,"ask_date":now_ms}, False),
        ("$0 bid (phantom-ish)", {"bid":0.0,"ask":0.60,"bid_date":now_ms,"ask_date":now_ms}, True),
        ("phantom $0.53 (bid0.30/ask0.75 = wide)", {"bid":0.30,"ask":0.75,"bid_date":now_ms,"ask_date":now_ms}, True),
    ]
    print("VALIDATE_QUOTE tests:")
    for label, q, sb in cases:
        ok, reason = validate_quote(q, sb)
        print(f"  [{'PASS' if ok else 'BLOCK'}] {label:45} -> {reason}")

    print("\nSQLITE logging test:")
    for label, q, sb in cases:
        check_and_log(q, "UVXY", "test", sb)
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    print(f"  rows logged: {n}")
    print("  sample row:", conn.execute("SELECT ts,symbol,bid,ask,spread_pct,age_sec,decision,sandbox FROM quotes ORDER BY id DESC LIMIT 1").fetchone())
    conn.close()
