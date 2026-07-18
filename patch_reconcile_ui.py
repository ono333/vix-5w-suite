#!/usr/bin/env python3
"""
patch_reconcile_ui.py
─────────────────────
Adds "Broker Reconciliation" section to render_system_health() in app.py.
Inserts AFTER the Data Paths block, BEFORE Backup Management.

Deploy:
    scp patch_reconcile_ui.py shin@192.168.100.142:~/vix_suite/
    cd ~/vix_suite && source venv/bin/activate
    python3 patch_reconcile_ui.py
"""

import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "app.py"

# ── The exact anchor line we insert AFTER ─────────────────────────────────────
ANCHOR = '''    # ═══════════════════════════════════════════════════════════════
    # BACKUP MANAGEMENT
    # ═══════════════════════════════════════════════════════════════'''

# ── Code block to insert ──────────────────────────────────────────────────────
INSERT = '''    # ═══════════════════════════════════════════════════════════════
    # BROKER RECONCILIATION
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔄 Broker Reconciliation")
    st.caption(
        "Upload your Fidelity CSV export to sync the trade log with actual positions. "
        "Always run a dry-run preview before applying."
    )

    uploaded_csv = st.file_uploader(
        "Upload Fidelity Portfolio CSV",
        type="csv",
        key="reconcile_csv_upload",
        help="Export from Fidelity: Accounts → Portfolio → Download CSV",
    )

    if uploaded_csv is not None:
        import tempfile, os
        from reconcile import parse_fidelity_csv, load_trade_log, diff, apply_mutations, backup, print_diff, TRADE_LOG_PATH
        import io, contextlib

        # Save upload to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_csv.read())
        tmp.close()

        try:
            fidelity = parse_fidelity_csv(tmp.name)
            log      = load_trade_log()
            mutations = diff(fidelity, log)

            total = sum(len(v) for v in mutations.values())

            # ── Diff summary table ────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Shorts to Close",  len(mutations["close_shorts"]),
                        delta=f"-{len(mutations['close_shorts'])}" if mutations["close_shorts"] else None,
                        delta_color="inverse")
            col2.metric("Shorts to Add",    len(mutations["add_shorts"]))
            col3.metric("Longs to Update",  len(mutations["update_longs"]))
            col4.metric("Corrupted to Fix", len(mutations["fix_corrupted"]),
                        delta=f"-{len(mutations['fix_corrupted'])}" if mutations["fix_corrupted"] else None,
                        delta_color="inverse")

            if total == 0:
                st.success("✅ Trade log is already in sync with Fidelity. Nothing to do.")
            else:
                # ── Detailed diff ─────────────────────────────────────────────
                with st.expander("📋 View full diff before applying", expanded=True):

                    if mutations["close_shorts"]:
                        st.markdown("**🔴 Shorts to CLOSE** (in log but not in Fidelity):")
                        for m in mutations["close_shorts"]:
                            st.markdown(
                                f"- `${m['strike']}` exp `{m['expiry']}` "
                                f"[{m['position_id']}] → _{m['exit_reason']}_"
                            )

                    if mutations["add_shorts"]:
                        st.markdown("**🟢 Shorts to ADD** (in Fidelity but missing from log):")
                        for m in mutations["add_shorts"]:
                            st.markdown(
                                f"- `${m['strike']}` exp `{m['expiry']}` "
                                f"× {m['contracts']} @ ${m['avg_cost']:.2f} credit"
                            )

                    if mutations["update_longs"]:
                        st.markdown("**🔵 Longs to UPDATE** (log long doesn't match Fidelity):")
                        for m in mutations["update_longs"]:
                            new = m.get("new_long")
                            new_str = f"→ `${new['strike']}` exp `{new['expiry']}`" if new else "→ ?"
                            st.markdown(
                                f"- [{m['variant']}] `${m['old_strike']}` "
                                f"exp `{m['old_expiry']}` {new_str}"
                            )

                    if mutations["fix_corrupted"]:
                        st.markdown("**⚠️ Corrupted legs to FIX:**")
                        for m in mutations["fix_corrupted"]:
                            st.markdown(
                                f"- [{m['position_id']}] `${m['strike']}` "
                                f"exp `{m['expiry']}` — {m['fix']}"
                            )

                # ── Apply button ──────────────────────────────────────────────
                st.warning(
                    f"⚡ {total} mutation(s) pending. "
                    "A backup will be created automatically before any changes."
                )
                if st.button("✅ Apply Reconciliation", type="primary",
                             key="reconcile_apply_btn"):
                    backup()
                    import json, copy
                    log_copy = copy.deepcopy(log)
                    apply_mutations(log_copy, mutations, fidelity)
                    with open(TRADE_LOG_PATH, "w") as f:
                        json.dump(log_copy, f, indent=2)
                    st.success(
                        f"✅ Trade log updated — {total} mutation(s) applied. "
                        "Refresh the page to see the changes."
                    )
                    st.balloons()

        except Exception as e:
            st.error(f"Reconciliation error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            os.unlink(tmp.name)

    st.markdown("---")
'''

# ── Patch logic ───────────────────────────────────────────────────────────────

def main():
    src = APP_PATH.read_text(encoding="utf-8")

    if "Broker Reconciliation" in src:
        print("⚠️  Reconciliation UI already present in app.py — skipping.")
        sys.exit(0)

    if ANCHOR not in src:
        print("❌ Anchor not found in app.py. Check that BACKUP MANAGEMENT header is intact.")
        sys.exit(1)

    patched = src.replace(ANCHOR, INSERT + ANCHOR, 1)

    # Backup original
    backup_path = APP_PATH.with_suffix(".py.bak_reconcile")
    APP_PATH.rename(backup_path)
    print(f"📦 Original backed up to {backup_path}")

    APP_PATH.write_text(patched, encoding="utf-8")
    print("✅ Patch applied: Broker Reconciliation section added to render_system_health()")
    print("🔄 Restart Streamlit to pick up the change.")


if __name__ == "__main__":
    main()
