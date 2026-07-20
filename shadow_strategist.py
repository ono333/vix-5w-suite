#!/usr/bin/env python3
"""shadow_strategist.py — forward-only shadow book for hypothetical UVXY strategies.

Runs right after the live orchestrator on trading days. NEVER places orders.
Records, from live Tradier quotes, what three hypothetical strategies would
have done, and marks/settles them over time so a real forward history accrues.

Strategies (all fills recorded at MID; bid stored as take-bid reference):
  S1 'vertical'   : short ~7DTE call (delta <= DELTA_CAP) + long wing ~+$3, same expiry
                    NO entry gates — this is the naive CONTROL arm.
  S2 'diagonal'   : same short call + long call in the ~25-46 DTE monthly at the
                    same-or-next strike. Long is kept; a new weekly short is
                    re-sold ("roll") after each short expiry while the long lives.
  S3 'long_spike' : BTO ~75-130 DTE call (delta ~ LONG_DELTA) when VIX percentile
                    < SPIKE_ENTRY_PCT. STC when mark >= TP_MULT x cost, or
                    percentile >= SPIKE_EXIT_PCT, or DTE < LONG_MIN_DTE.
  S4 'vertical_gated'    : same structure as S1, but entry requires
                    net credit >= GATE_CREDIT_PCT x wing width AND no
                    backwardation (risk_measures slope_ratio < GATE_SLOPE_MAX).
                    Gated-vs-S1 = does gating add value?
  S5 'vertical_defended' : same entries as S1, but at ~daily check, if the
                    short's chain delta >= DEFENSE_DELTA it is mechanically
                    rolled up-and-out (BTC + STO next weekly, higher strike).
                    Defended-vs-S1 = does roll-at-0.50 beat hold-through?
  S6 'ratio_wing' : same weekly short as S1, protected not by a same-expiry
                    wing but by PERSISTENT inventory of RATIO_LONGS far-OTM
                    ~100DTE longs (delta ~ RATIO_LONG_DELTA), topped up when
                    one is closed/expired; longs time-stopped at
                    RATIO_LONG_MIN_DTE. Cheap-long insurance question.

Tables created in ~/.vix_suite/market_data.db (existing tables untouched):
  vix_history, shadow_positions, shadow_trades, shadow_marks

Usage:
  python3 shadow_strategist.py                 # normal daily run (after orchestrator)
  python3 shadow_strategist.py --report        # P&L summary per strategy
  python3 shadow_strategist.py --seed-vix F    # seed VIX history from CSV (date,close)
  python3 shadow_strategist.py --force         # run even if market is closed
  python3 shadow_strategist.py --marks-only    # mark + defend only, NO entries (off-day cron)

Optional live take-bid hook — one line in tradier_orchestrator.py where a live
fill is confirmed (you have bid/ask from the quote used to place the order):
  from shadow_strategist import log_live_takebid
  log_live_takebid(occ_symbol, bid_at_entry, ask_at_entry, fill_price, variant="V2")
"""

import csv
import os
import sqlite3
import sys
import uuid
from datetime import date, datetime

import requests

# ---------------------------------------------------------------- config ----
DB_PATH = os.path.expanduser("~/.vix_suite/market_data.db")
LOG_PATH = os.path.expanduser("~/vix_suite/shadow.log")

UNDERLYING = "UVXY"
VIX_SYMBOL = "VIX"

DELTA_CAP = 0.30        # shadow short-call delta ceiling (live uses ~0.27 + relax)
DELTA_FLOOR = 0.12      # avoid junk strikes with no premium
MIN_BID = 0.10
WING_WIDTH = 3.0        # vertical wing target $ above short strike (min +2.0)
SHORT_DTE = (4, 10)     # weekly short expiry window
DIAG_LONG_DTE = (25, 46)
DIAG_LONG_TARGET = 35
LONG_DTE = (75, 130)    # S3 long-call expiry window
LONG_DTE_TARGET = 100
LONG_DELTA = 0.30       # S3 target delta
SPIKE_ENTRY_PCT = 25.0  # buy calm
SPIKE_EXIT_PCT = 75.0   # sell spike
TP_MULT = 2.5           # take profit at 2.5x cost
LONG_MIN_DTE = 21       # time-stop for S3 long
MIN_PCTILE_HISTORY = 60  # trading days of VIX closes needed before S3 acts

# --- S4 gates (mirror live engine's floors) ---
GATE_CREDIT_PCT = 0.03   # net credit must be >= 3% of wing width
GATE_SLOPE_MAX = 1.0     # risk_measures slope_ratio >= this => backwardation => skip

# --- S5 defense ---
DEFENSE_DELTA = 0.50     # roll the short when its chain delta reaches this

# --- S6 ratio wings ---
RATIO_LONGS = 2          # persistent far-OTM long inventory per short book
RATIO_LONG_DELTA = 0.10  # target delta for cheap longs
RATIO_LONG_MIN_DTE = 21  # time-stop: STC stub and let top-up replace it

SCHEMA = """
CREATE TABLE IF NOT EXISTS vix_history(
  date TEXT PRIMARY KEY, close REAL);
CREATE TABLE IF NOT EXISTS shadow_positions(
  id INTEGER PRIMARY KEY, spread_id TEXT, strategy TEXT, occ TEXT,
  expiry TEXT, strike REAL, right TEXT, qty INTEGER,
  open_price REAL, open_ts TEXT, status TEXT DEFAULT 'open',
  close_price REAL, close_ts TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS shadow_trades(
  id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, action TEXT, occ TEXT,
  expiry TEXT, strike REAL, right TEXT, qty INTEGER,
  price REAL, bid REAL, ask REAL, spot REAL, vix REAL, vix_pctile REAL, note TEXT);
CREATE TABLE IF NOT EXISTS shadow_marks(
  id INTEGER PRIMARY KEY, ts TEXT, position_id INTEGER, strategy TEXT, occ TEXT,
  bid REAL, ask REAL, mid REAL, spot REAL);
"""


# ---------------------------------------------------------------- helpers ---
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{date.today()} {line}\n")
    except OSError:
        pass


def get_client():
    """Reuse the live TradierClient; fall back to env vars if unavailable."""
    try:
        sys.path.insert(0, os.path.expanduser("~/vix_suite"))
        from tradier_orchestrator import TradierClient
        return TradierClient(sandbox=True)
    except Exception:
        class _Env:
            base = os.environ.get("TRADIER_BASE", "https://sandbox.tradier.com/v1")
            token = os.environ["TRADIER_TOKEN"]
        return _Env()


def api(client, path, **params):
    r = requests.get(
        f"{client.base}{path}", params=params,
        headers={"Authorization": f"Bearer {client.token}",
                 "Accept": "application/json"},
        timeout=20)
    r.raise_for_status()
    return r.json()


def as_list(x):
    if x is None or x == "null":
        return []
    return x if isinstance(x, list) else [x]


def mid(o):
    b, a = o.get("bid") or 0.0, o.get("ask") or 0.0
    if b > 0 and a > 0:
        return round((b + a) / 2, 2)
    return round(b or a, 2)


def dte(expiry_iso):
    return (date.fromisoformat(expiry_iso) - date.today()).days


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.executescript(SCHEMA)
    return conn


def vix_percentile(conn, today_iso, vix_close):
    """Percentile of today's close vs trailing <=252 stored closes (excl. today)."""
    if vix_close is None:
        return None
    rows = [r[0] for r in conn.execute(
        "SELECT close FROM vix_history WHERE date < ? ORDER BY date DESC LIMIT 252",
        (today_iso,))]
    if len(rows) < MIN_PCTILE_HISTORY:
        return None
    below = sum(1 for x in rows if x < vix_close)
    return round(100.0 * below / len(rows), 1)


def latest_slope(conn):
    """Most recent slope_ratio from risk_logger's risk_measures table.
    Returns None (gate treated as pass, logged) if table/column absent."""
    try:
        r = conn.execute(
            "SELECT slope_ratio FROM risk_measures "
            "ORDER BY date DESC LIMIT 1").fetchone()
        return r[0] if r else None
    except sqlite3.Error:
        return None


# ------------------------------------------------------------ selection -----
def pick_exp(exps, lo, hi, target=None):
    win = [e for e in exps if lo <= dte(e) <= hi]
    if not win:
        return None
    if target is None:
        return min(win, key=dte)
    return min(win, key=lambda e: abs(dte(e) - target))


def pick_short(chain_calls):
    cands = []
    for o in chain_calls:
        g = o.get("greeks") or {}
        d = g.get("delta")
        if d is None:
            continue
        if (o.get("bid") or 0) >= MIN_BID and (o.get("ask") or 0) > 0 \
                and DELTA_FLOOR <= d <= DELTA_CAP:
            cands.append(o)
    return min(cands, key=lambda o: o["strike"]) if cands else None


def pick_wing(chain_calls, short_strike):
    cands = [o for o in chain_calls
             if o["strike"] >= short_strike + 2.0 and (o.get("ask") or 0) > 0]
    if cands:
        return min(cands, key=lambda o: abs(o["strike"] - (short_strike + WING_WIDTH)))
    # sparse chain fallback: highest listed strike above the short
    above = [o for o in chain_calls
             if o["strike"] > short_strike and (o.get("ask") or 0) > 0]
    return max(above, key=lambda o: o["strike"]) if above else None


def pick_at_or_above(chain_calls, strike):
    cands = [o for o in chain_calls
             if o["strike"] >= strike and (o.get("ask") or 0) > 0]
    return min(cands, key=lambda o: o["strike"]) if cands else None


def pick_by_delta(chain_calls, target):
    cands = [o for o in chain_calls
             if (o.get("greeks") or {}).get("delta") is not None
             and (o.get("ask") or 0) > 0]
    if not cands:
        return None
    return min(cands, key=lambda o: abs(o["greeks"]["delta"] - target))


# ------------------------------------------------------------ bookkeeping ---
def open_leg(conn, spread_id, strategy, opt, qty, price, ts,
             spot, vix, pct, note=""):
    conn.execute(
        "INSERT INTO shadow_positions(spread_id,strategy,occ,expiry,strike,right,"
        "qty,open_price,open_ts,status,note) VALUES(?,?,?,?,?,?,?,?,?,'open',?)",
        (spread_id, strategy, opt["symbol"], opt["expiration_date"],
         opt["strike"], "call", qty, price, ts, note))
    conn.execute(
        "INSERT INTO shadow_trades(ts,strategy,action,occ,expiry,strike,right,qty,"
        "price,bid,ask,spot,vix,vix_pctile,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, strategy, "STO" if qty < 0 else "BTO", opt["symbol"],
         opt["expiration_date"], opt["strike"], "call", qty, price,
         opt.get("bid"), opt.get("ask"), spot, vix, pct, note))


def close_position(conn, pos_id, price, ts, action, note=""):
    row = conn.execute(
        "SELECT strategy,occ,expiry,strike,qty FROM shadow_positions WHERE id=?",
        (pos_id,)).fetchone()
    if not row:
        return
    strategy, occ, expiry, strike, qty = row
    conn.execute(
        "UPDATE shadow_positions SET status=?, close_price=?, close_ts=? WHERE id=?",
        (action, price, ts, pos_id))
    conn.execute(
        "INSERT INTO shadow_trades(ts,strategy,action,occ,expiry,strike,right,qty,"
        "price,note) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ts, strategy, action, occ, expiry, strike, "call", -qty, price, note))


def has_open_short(conn, strategy, expiry):
    return conn.execute(
        "SELECT 1 FROM shadow_positions WHERE strategy=? AND expiry=? "
        "AND qty<0 AND status='open' LIMIT 1", (strategy, expiry)).fetchone()


def has_live_short(conn, strategy, today_iso):
    """Any open short not expiring today — used by the defended arm so a
    rolled short (now in a later expiry) blocks a fresh weekly entry."""
    return conn.execute(
        "SELECT 1 FROM shadow_positions WHERE strategy=? AND expiry>? "
        "AND qty<0 AND status='open' LIMIT 1", (strategy, today_iso)).fetchone()


def open_rows(conn, strategy=None):
    q = ("SELECT id,spread_id,strategy,occ,expiry,strike,qty,open_price "
         "FROM shadow_positions WHERE status='open'")
    if strategy:
        return conn.execute(q + " AND strategy=?", (strategy,)).fetchall()
    return conn.execute(q).fetchall()


# ------------------------------------------------------------------ run -----
def run(force=False, marks_only=False):
    c = get_client()
    ts = datetime.now().isoformat(timespec="seconds")
    today = date.today().isoformat()

    clock = api(c, "/markets/clock").get("clock", {})
    if clock.get("state") != "open" and not force:
        log(f"shadow: market state={clock.get('state')!r}; skipping (--force to override)")
        return

    conn = connect()

    # --- quotes: spot + VIX -------------------------------------------------
    qs = as_list(api(c, "/markets/quotes",
                     symbols=f"{UNDERLYING},{VIX_SYMBOL}")
                 .get("quotes", {}).get("quote"))
    qmap = {x.get("symbol"): x for x in qs}
    spot = qmap.get(UNDERLYING, {}).get("last")
    vix = qmap.get(VIX_SYMBOL, {}).get("last")
    if spot is None:
        log("shadow: no UVXY quote — aborting")
        return
    if vix is not None:
        conn.execute("INSERT OR REPLACE INTO vix_history(date,close) VALUES(?,?)",
                     (today, vix))
    pct = vix_percentile(conn, today, vix)
    nhist = conn.execute("SELECT COUNT(*) FROM vix_history").fetchone()[0]
    log(f"shadow: spot={spot} vix={vix} pctile="
        f"{pct if pct is not None else f'n/a ({nhist}/{MIN_PCTILE_HISTORY} closes)'}")

    # --- settle expired positions (intrinsic at current spot as proxy) ------
    for pid, _sid, strat, occ, expiry, strike, qty, _op in open_rows(conn):
        if expiry < today:
            intrinsic = round(max(0.0, spot - strike), 2)
            close_position(conn, pid, intrinsic, ts, "expired",
                           f"settled@spot_proxy {spot}")
            log(f"shadow: {strat} {occ} expired -> settle {intrinsic:.2f}")

    # --- mark all remaining open positions (one batched quote call) ---------
    opens = open_rows(conn)
    marks = {}
    if opens:
        occs = ",".join(sorted({r[3] for r in opens}))
        oq = {x.get("symbol"): x for x in
              as_list(api(c, "/markets/quotes", symbols=occs, greeks="false")
                      .get("quotes", {}).get("quote"))}
        for pid, _sid, strat, occ, expiry, strike, qty, _op in opens:
            o = oq.get(occ)
            if not o:
                continue
            m = mid(o)
            marks[pid] = m
            conn.execute(
                "INSERT INTO shadow_marks(ts,position_id,strategy,occ,bid,ask,mid,spot)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (ts, pid, strat, occ, o.get("bid"), o.get("ask"), m, spot))

    # --- S3 exits ------------------------------------------------------------
    for pid, _sid, strat, occ, expiry, strike, qty, op in open_rows(conn, "long_spike"):
        m = marks.get(pid)
        if m is None:
            continue
        reason = None
        if m >= TP_MULT * op:
            reason = f"take_profit {m:.2f}>= {TP_MULT}x{op:.2f}"
        elif pct is not None and pct >= SPIKE_EXIT_PCT:
            reason = f"pctile_exit {pct}"
        elif dte(expiry) < LONG_MIN_DTE:
            reason = f"time_stop dte={dte(expiry)}"
        if reason:
            close_position(conn, pid, m, ts, "STC", reason)
            log(f"shadow: LONG_SPIKE STC {occ} @ {m:.2f} ({reason})")

    # --- expirations + chains -----------------------------------------------
    exps = as_list(api(c, "/markets/options/expirations", symbol=UNDERLYING,
                       includeAllRoots="true", strikes="false")
                   .get("expirations", {}).get("date"))
    chains = {}

    def calls(exp):
        if exp not in chains:
            chains[exp] = [o for o in
                           as_list(api(c, "/markets/options/chains",
                                       symbol=UNDERLYING, expiration=exp,
                                       greeks="true")
                                   .get("options", {}).get("option"))
                           if o.get("option_type") == "call"]
        return chains[exp]

    wexp = pick_exp(exps, *SHORT_DTE)

    # --- S5 defense: roll defended shorts whose chain delta >= DEFENSE_DELTA -
    for pid, sid, strat, occ, expiry, strike, qty, op in open_rows(conn, "vertical_defended"):
        if qty >= 0 or expiry < today:
            continue
        ch = calls(expiry)
        o = next((x for x in ch if x["strike"] == strike), None)
        d = ((o or {}).get("greeks") or {}).get("delta")
        if o is None or d is None:
            log(f"shadow: DEFENSE no chain/delta for {occ}; skip check")
            continue
        if d < DEFENSE_DELTA:
            continue
        btc = mid(o)
        close_position(conn, pid, btc, ts, "BTC", f"defense delta={d:.2f}")
        log(f"shadow: DEFENDED BTC {occ} @ {btc:.2f} (delta {d:.2f})")
        nexp = pick_exp(exps, dte(expiry) + 1, dte(expiry) + 9)
        roll = None
        if nexp:
            cands = [x for x in calls(nexp)
                     if (x.get("greeks") or {}).get("delta") is not None
                     and x["greeks"]["delta"] <= DELTA_CAP
                     and x["strike"] > strike
                     and (x.get("bid") or 0) >= MIN_BID]
            roll = min(cands, key=lambda x: x["strike"]) if cands else None
        if roll:
            rm = mid(roll)
            open_leg(conn, sid, "vertical_defended", roll, -1, rm, ts,
                     spot, vix, pct, f"defense roll from {strike:g}C {expiry}")
            log(f"shadow: DEFENDED roll STO {roll['strike']:g}C {nexp} @ {rm:.2f}")
        else:
            log("shadow: DEFENDED roll declined (no candidate) — flat after BTC")

    # --- marks-only pass ends here: settle/mark/S3-exit/S5-defense done, ------
    #     no new entries. Used by the daily off-day cron so the defended arm
    #     sees delta>=0.50 crossings same-day instead of only Mon/Fri.
    if marks_only:
        conn.commit()
        conn.close()
        log("shadow: marks-only run complete")
        return

    # --- S1/S4/S5 verticals (shared snapshot: same cached chain) --------------
    def enter_vertical(strat, s, w, note_extra=""):
        sid = uuid.uuid4().hex[:8]
        sm, wm = mid(s), mid(w)
        open_leg(conn, sid, strat, s, -1, sm, ts, spot, vix, pct, "short leg")
        open_leg(conn, sid, strat, w, +1, wm, ts, spot, vix, pct, "wing")
        log(f"shadow: {strat.upper()} STO {s['strike']:g}C {wexp} @ {sm:.2f} "
            f"(bid {s.get('bid')}) / BTO {w['strike']:g}C @ {wm:.2f} "
            f"-> net {sm - wm:.2f}{note_extra}")

    s_pick = pick_short(calls(wexp)) if wexp else None
    w_pick = pick_wing(calls(wexp), s_pick["strike"]) if s_pick else None
    pair_ok = s_pick and w_pick and w_pick["strike"] > s_pick["strike"]

    if wexp and not has_open_short(conn, "vertical", wexp):
        if pair_ok:
            enter_vertical("vertical", s_pick, w_pick)
        else:
            log(f"shadow: VERTICAL declined ({'no short candidate' if not s_pick else 'no wing'})")
    elif wexp:
        log(f"shadow: VERTICAL holds open short for {wexp}")

    # --- S4 gated vertical ----------------------------------------------------
    if wexp and not has_open_short(conn, "vertical_gated", wexp):
        if not pair_ok:
            log("shadow: GATED declined (no short/wing candidate)")
        else:
            net = mid(s_pick) - mid(w_pick)
            width = w_pick["strike"] - s_pick["strike"]
            slope = latest_slope(conn)
            if slope is not None and slope >= GATE_SLOPE_MAX:
                log(f"shadow: GATED stand-down (slope_ratio {slope:.3f} >= {GATE_SLOPE_MAX})")
            elif net < GATE_CREDIT_PCT * width:
                log(f"shadow: GATED declined (net {net:.2f} < "
                    f"{GATE_CREDIT_PCT:.0%} x width {width:g})")
            else:
                extra = " [slope n/a]" if slope is None else f" [slope {slope:.3f}]"
                enter_vertical("vertical_gated", s_pick, w_pick, extra)

    # --- S5 defended vertical (same entries as S1; defense handled above) -----
    # has_live_short (not per-expiry): a defense-rolled short in a later
    # expiry must block a fresh weekly entry, else the arm doubles up.
    if wexp and not has_live_short(conn, "vertical_defended", today):
        if pair_ok:
            enter_vertical("vertical_defended", s_pick, w_pick)
        else:
            log("shadow: DEFENDED declined (no short/wing candidate)")

    # --- S6 ratio_wing: weekly short + persistent far-OTM long inventory ------
    if wexp and not has_open_short(conn, "ratio_wing", wexp):
        if s_pick:
            sm = mid(s_pick)
            open_leg(conn, uuid.uuid4().hex[:8], "ratio_wing", s_pick, -1, sm,
                     ts, spot, vix, pct, "short leg")
            log(f"shadow: RATIO_WING STO {s_pick['strike']:g}C {wexp} @ {sm:.2f}")
        else:
            log("shadow: RATIO_WING short declined (no candidate)")
    rw_longs = [r for r in open_rows(conn, "ratio_wing") if r[6] > 0]
    for pid, _sid, _st, occ, expiry, strike, q, op in rw_longs:
        if dte(expiry) < RATIO_LONG_MIN_DTE:
            m = marks.get(pid)
            if m is not None:
                close_position(conn, pid, m, ts, "STC",
                               f"time_stop dte={dte(expiry)}")
                log(f"shadow: RATIO_WING STC stub {occ} @ {m:.2f}")
    rw_longs = [r for r in open_rows(conn, "ratio_wing") if r[6] > 0]
    need = RATIO_LONGS - sum(r[6] for r in rw_longs)
    if need > 0:
        lexp = pick_exp(exps, *LONG_DTE, target=LONG_DTE_TARGET)
        lg = pick_by_delta(calls(lexp), RATIO_LONG_DELTA) if lexp else None
        if lg:
            lm = mid(lg)
            open_leg(conn, uuid.uuid4().hex[:8], "ratio_wing", lg, +need, lm,
                     ts, spot, vix, pct, f"inventory top-up x{need}")
            log(f"shadow: RATIO_WING BTO {need}x {lg['strike']:g}C {lexp} @ {lm:.2f}")
        else:
            log("shadow: RATIO_WING top-up declined (no candidate)")

    # --- S2 diagonal -----------------------------------------------------------
    diag_open = open_rows(conn, "diagonal")
    diag_longs = [r for r in diag_open if r[6] > 0]
    diag_shorts = [r for r in diag_open if r[6] < 0]
    if not diag_longs:
        mexp = pick_exp(exps, *DIAG_LONG_DTE, target=DIAG_LONG_TARGET)
        s = pick_short(calls(wexp)) if wexp else None
        lg = pick_at_or_above(calls(mexp), s["strike"]) if (mexp and s) else None
        if s and lg:
            sid = uuid.uuid4().hex[:8]
            sm, lm = mid(s), mid(lg)
            open_leg(conn, sid, "diagonal", s, -1, sm, ts, spot, vix, pct, "short leg")
            open_leg(conn, sid, "diagonal", lg, +1, lm, ts, spot, vix, pct, "long leg")
            log(f"shadow: DIAGONAL STO {s['strike']}C {wexp} @ {sm:.2f} / "
                f"BTO {lg['strike']}C {mexp} @ {lm:.2f} -> net debit {lm - sm:.2f}")
        else:
            log("shadow: DIAGONAL declined (no short/long candidate)")
    elif not diag_shorts and wexp and dte(wexp) < dte(diag_longs[0][4]):
        # long survives, short expired -> roll a new weekly short against it
        s = pick_short(calls(wexp))
        if s and not has_open_short(conn, "diagonal", wexp):
            sm = mid(s)
            open_leg(conn, diag_longs[0][1], "diagonal", s, -1, sm, ts,
                     spot, vix, pct, "roll")
            log(f"shadow: DIAGONAL roll STO {s['strike']}C {wexp} @ {sm:.2f}")
        elif not s:
            log("shadow: DIAGONAL roll declined (no candidate)")
    else:
        log("shadow: DIAGONAL holds")

    # --- S3 long_spike entry -----------------------------------------------------
    if pct is None:
        log("shadow: LONG_SPIKE inactive — need VIX history "
            f"({nhist}/{MIN_PCTILE_HISTORY}; seed with --seed-vix)")
    elif open_rows(conn, "long_spike"):
        log("shadow: LONG_SPIKE holds")
    elif pct < SPIKE_ENTRY_PCT:
        lexp = pick_exp(exps, *LONG_DTE, target=LONG_DTE_TARGET)
        lg = pick_by_delta(calls(lexp), LONG_DELTA) if lexp else None
        if lg:
            lm = mid(lg)
            open_leg(conn, uuid.uuid4().hex[:8], "long_spike", lg, +1, lm, ts,
                     spot, vix, pct, f"entry pctile={pct}")
            log(f"shadow: LONG_SPIKE BTO {lg['strike']}C {lexp} @ {lm:.2f} "
                f"(pctile {pct})")
        else:
            log("shadow: LONG_SPIKE declined (no candidate)")
    else:
        log(f"shadow: LONG_SPIKE no entry (pctile {pct} >= {SPIKE_ENTRY_PCT})")

    conn.commit()
    conn.close()
    log("shadow: run complete")


# ---------------------------------------------------------------- report ----
def report():
    conn = connect()
    print(f"{'strategy':<19}{'open':>6}{'closed':>8}{'realized':>12}"
          f"{'unrealized':>12}{'total':>10}")
    for strat in ("vertical", "vertical_gated", "vertical_defended",
                  "ratio_wing", "diagonal", "long_spike"):
        closed = conn.execute(
            "SELECT qty,open_price,close_price FROM shadow_positions "
            "WHERE strategy=? AND status!='open'", (strat,)).fetchall()
        realized = sum((cp - op) * q * 100 for q, op, cp in closed
                       if cp is not None)
        opens = conn.execute(
            "SELECT id,qty,open_price FROM shadow_positions "
            "WHERE strategy=? AND status='open'", (strat,)).fetchall()
        unreal = 0.0
        for pid, q, op in opens:
            m = conn.execute(
                "SELECT mid FROM shadow_marks WHERE position_id=? "
                "ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
            if m and m[0] is not None:
                unreal += (m[0] - op) * q * 100
        print(f"{strat:<19}{len(opens):>6}{len(closed):>8}"
              f"{realized:>12.2f}{unreal:>12.2f}{realized + unreal:>10.2f}")
    tb = conn.execute(
        "SELECT COUNT(*), AVG(price - bid) FROM shadow_trades "
        "WHERE strategy='live_takebid_ref' AND bid IS NOT NULL").fetchone()
    if tb and tb[0]:
        print(f"\nlive take-bid refs: {tb[0]} entries, "
              f"avg credit saved vs take-bid: {tb[1]:.3f}/contract")
    conn.close()


# ------------------------------------------------------------- seeding ------
def seed_vix(path):
    conn = connect()
    n = 0
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            d, v = row[0].strip(), row[1].strip()
            try:
                if "/" in d:  # CBOE format MM/DD/YYYY
                    d = datetime.strptime(d, "%m/%d/%Y").date().isoformat()
                else:
                    d = date.fromisoformat(d).isoformat()
                conn.execute(
                    "INSERT OR REPLACE INTO vix_history(date,close) VALUES(?,?)",
                    (d, float(v)))
                n += 1
            except ValueError:
                continue  # header or malformed row
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM vix_history").fetchone()[0]
    conn.close()
    log(f"shadow: seeded {n} rows from {path} (vix_history total: {total})")


# ------------------------------------------------------ live take-bid hook --
def log_live_takebid(occ, bid, ask, fill_price, variant=""):
    """Call from the live orchestrator right after a fill is confirmed.
    Records fill vs bid so mid-working vs take-bid can be compared later."""
    try:
        conn = connect()
        conn.execute(
            "INSERT INTO shadow_trades(ts,strategy,action,occ,price,bid,ask,note)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"),
             "live_takebid_ref", "ref", occ, fill_price, bid, ask, variant))
        conn.commit()
        conn.close()
    except Exception as e:  # never break the live path
        print(f"log_live_takebid failed: {e}")


# ------------------------------------------------------------------ main ----
if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--seed-vix" in sys.argv:
        seed_vix(sys.argv[sys.argv.index("--seed-vix") + 1])
    else:
        run(force="--force" in sys.argv,
            marks_only="--marks-only" in sys.argv)
