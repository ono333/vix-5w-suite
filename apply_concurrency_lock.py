#!/usr/bin/env python3
"""
apply_concurrency_lock.py — adds a single-instance flock so the orchestrator
refuses to start if another instance is already running (prevents the
position-doubling bug: timer + manual run both placing the same variant's short).

Lock is acquired ONLY for order-placing modes (paper/live), just before run().
check/preview run without the lock (read-only, safe to overlap).
flock is atomic + auto-released on process death (no stale locks).
Lockfile: ~/vix_suite/tradier_orchestrator.lock. Emails on block.

Line-anchored edits (no pattern replacement). RUN ON SERVER.
"""
import shutil, datetime, py_compile, os, sys

TARGET = os.path.expanduser("~/vix_suite/tradier_orchestrator.py")
src = open(TARGET).read()
bak = TARGET + ".bak_lock_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(TARGET, bak)
print(f"\u2713 backup: {bak}")

# ── 1. add `import fcntl` after `import sys` ──
if "import fcntl" not in src:
    src = src.replace("import sys\n", "import sys\nimport fcntl\n", 1)
    print("\u2713 import fcntl added")
else:
    print("\u2022 import fcntl already present")

# ── 2. add lock helpers + alert just before `def main():` ──
if "acquire_single_instance_lock" not in src:
    anchor = "def main():"
    helpers = '''LOCK_PATH = os.path.expanduser("~/vix_suite/tradier_orchestrator.lock")


def acquire_single_instance_lock():
    """Refuse to start if another orchestrator instance holds the lock.
    flock is atomic and auto-released by the OS on process death (no stale locks).
    Returns: a file handle (lock held, proceed), or None (another instance holds it).
    On a lockfile I/O problem (missing dir, unwritable), returns the sentinel
    string 'NO_LOCKFILE' so the caller can proceed WITHOUT crashing \u2014 a lock
    problem must never take down the run."""
    try:
        lock_file = open(LOCK_PATH, "w")
    except OSError as e:
        print(f"\u26a0\ufe0f Could not open lockfile {LOCK_PATH}: {e} \u2014 proceeding without lock.")
        return "NO_LOCKFILE"
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return None
    try:
        lock_file.write(f"{os.getpid()}\\n")
        lock_file.flush()
    except OSError:
        pass
    return lock_file


def _send_lock_alert():
    """Email when a second instance was blocked. Reuses the SMTP env credentials."""
    try:
        user = os.environ.get("SMTP_USER", "")
        pw = os.environ.get("SMTP_PASS", "")
        if not user or not pw:
            return
        msg = MIMEText("A second tradier_orchestrator instance tried to start while one "
                       "was already running, and was blocked to prevent double-placement. "
                       "Likely an overlapping timer + manual run \\u2014 check before next run.")
        msg["Subject"] = "\\U0001f512 Orchestrator overlap blocked"
        msg["From"] = user
        msg["To"] = user
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.send_message(msg)
    except Exception:
        pass


'''
    src = src.replace(anchor, helpers + anchor, 1)
    print("\u2713 lock helpers + alert inserted before main()")
else:
    print("\u2022 lock helpers already present")

# ── 3. insert the lock acquire just before the run(...) call in main() ──
# The run( call in main() is the unique multi-line block:
run_anchor = '''    run(
        sandbox    = not args.live,
        preview    = args.preview or (not args.paper and not args.check),
        check_only = args.check,
    )'''
if run_anchor not in src:
    print("\u274c run() call block in main() not found as expected. File unchanged.")
    shutil.copy(bak, TARGET); sys.exit(1)

locked_run = '''    # ── single-instance lock (order-placing modes only) ──────────────────────
    _places_orders = args.paper or args.live
    _lock = None
    if _places_orders:
        _lock = acquire_single_instance_lock()
        if _lock is None:
            print("\\U0001f512 Another orchestrator instance is already running \\u2014 "
                  "exiting to prevent double-placement.")
            _send_lock_alert()
            return
    # _lock stays open (held) for the duration of run() below; released on exit.
    # ─────────────────────────────────────────────────────────────────────────

    run(
        sandbox    = not args.live,
        preview    = args.preview or (not args.paper and not args.check),
        check_only = args.check,
    )'''
src = src.replace(run_anchor, locked_run, 1)
print("\u2713 lock acquire inserted before run() (paper/live only)")

open(TARGET, "w").write(src)
try:
    py_compile.compile(TARGET, doraise=True)
    print("\u2713 py_compile OK")
except py_compile.PyCompileError as e:
    print(f"\u274c compile FAILED \u2014 restoring.\n{e}"); shutil.copy(bak, TARGET); sys.exit(1)
print("\n\u2705 DONE. Test: python3 tradier_orchestrator.py --preview")
print(f"   Revert: cp {bak} {TARGET}")
