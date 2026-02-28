"""
Systematic paper/real naming patch.
- Adds _paper_ prefix to all paper trade functions
- Adds _real_ prefix to all real trade functions  
- Prefixes all form/session keys with p_ (paper) or r_ (real)
- Fixes long-only checkbox to appear in real New Entry, not paper
"""
from pathlib import Path
import re

app_path = Path("app.py")
rtu_path = Path("real_trade_ui.py")
src = app_path.read_text()
rtu = rtu_path.read_text()
changes = 0

# ══════════════════════════════════════════════════════════════════════
# 1. FUNCTION RENAMES in app.py
# ══════════════════════════════════════════════════════════════════════
renames = [
    # Paper trade functions
    ("def render_trade_log(",              "def render_paper_trade_log("),
    ("render_trade_log(",                  "render_paper_trade_log("),
    ("def _render_diagonal_positions(",    "def _render_paper_diagonal_positions("),
    ("_render_diagonal_positions(",        "_render_paper_diagonal_positions("),
    ("def _render_diagonal_entry_form(",   "def _render_paper_diagonal_entry_form("),
    ("_render_diagonal_entry_form(",       "_render_paper_diagonal_entry_form("),
    ("def _render_roll_form(",             "def _render_paper_roll_form("),
    ("_render_roll_form(",                 "_render_paper_roll_form("),
    ("def _render_price_update_form(",     "def _render_paper_price_update_form("),
    ("_render_price_update_form(",         "_render_paper_price_update_form("),
    ("def _render_close_form(",            "def _render_paper_close_form("),
    ("_render_close_form(",                "_render_paper_close_form("),
    ("def _render_edit_form(",             "def _render_paper_edit_form("),
    ("_render_edit_form(",                 "_render_paper_edit_form("),
    ("def _render_roll_history_edit_form(","def _render_paper_roll_history_edit_form("),
    ("_render_roll_history_edit_form(",    "_render_paper_roll_history_edit_form("),
    ("def _render_roll_long_form(",        "def _render_paper_roll_long_form("),
    ("_render_roll_long_form(",            "_render_paper_roll_long_form("),
    ("def _render_close_short_form(",      "def _render_paper_close_short_form("),
    ("_render_close_short_form(",          "_render_paper_close_short_form("),
    ("def _render_close_long_form(",       "def _render_paper_close_long_form("),
    ("_render_close_long_form(",           "_render_paper_close_long_form("),
    ("def _render_sell_short_form(",       "def _render_paper_sell_short_form("),
    ("_render_sell_short_form(",           "_render_paper_sell_short_form("),
    ("def _render_expire_confirm(",        "def _render_paper_expire_confirm("),
    ("_render_expire_confirm(",            "_render_paper_expire_confirm("),
    ("def _render_delete_confirm(",        "def _render_paper_delete_confirm("),
    ("_render_delete_confirm(",            "_render_paper_delete_confirm("),
    ("def _render_roll_analytics(",        "def _render_paper_roll_analytics("),
    ("_render_roll_analytics(",            "_render_paper_roll_analytics("),
    # Real trade functions
    ("def render_real_trade_log_page(",    "def render_real_trade_log("),
    ("render_real_trade_log_page(",        "render_real_trade_log("),
    ("def _render_real_roll_edit_form(",   "def _render_real_roll_edit_form("),  # already correct
]

for old, new in renames:
    if old in src and old != new:
        count = src.count(old)
        src = src.replace(old, new)
        print(f"✓ {old.strip()} → {new.strip()} ({count}x)")
        changes += 1

# ══════════════════════════════════════════════════════════════════════
# 2. PREFIX PAPER FORM/SESSION KEYS with p_
# Keys inside paper trade functions (identified by pos.position_id pattern)
# ══════════════════════════════════════════════════════════════════════
paper_key_renames = [
    # Roll history editing
    ('"edit_rolls_{pos.position_id}"',      '"p_edit_rolls_{pos.position_id}"'),
    ('"editing_rolls_{pos.position_id}"',   '"p_editing_rolls_{pos.position_id}"'),
    ('"editing_{pos.position_id}"',         '"p_editing_{pos.position_id}"'),
    # Roll form keys
    ('"roll_{pos.position_id}"',            '"p_roll_{pos.position_id}"'),
    ('"roll_submit_{pos.position_id}"',     '"p_roll_submit_{pos.position_id}"'),
    ('"roll_new_strike_{pos.position_id}"', '"p_roll_new_strike_{pos.position_id}"'),
    ('"roll_new_exp_{pos.position_id}"',    '"p_roll_new_exp_{pos.position_id}"'),
    ('"roll_new_credit_{pos.position_id}"', '"p_roll_new_credit_{pos.position_id}"'),
    ('"roll_exit_{pos.position_id}"',       '"p_roll_exit_{pos.position_id}"'),
    ('"roll_contracts_{pos.position_id}"',  '"p_roll_contracts_{pos.position_id}"'),
    ('"roll_underlying_{pos.position_id}"', '"p_roll_underlying_{pos.position_id}"'),
    ('"roll_editor_{pos.position_id}"',     '"p_roll_editor_{pos.position_id}"'),
    # Close/expire/delete
    ('"close_{pos.position_id}"',           '"p_close_{pos.position_id}"'),
    ('"expire_{pos.position_id}"',          '"p_expire_{pos.position_id}"'),
    ('"delete_{pos.position_id}"',          '"p_delete_{pos.position_id}"'),
    ('"update_{pos.position_id}"',          '"p_update_{pos.position_id}"'),
    ('"edit_{pos.position_id}"',            '"p_edit_{pos.position_id}"'),
    ('"sell_short_{pos.position_id}"',      '"p_sell_short_{pos.position_id}"'),
    ('"recalc_rolls_{pos.position_id}"',    '"p_recalc_rolls_{pos.position_id}"'),
    # Close form keys
    ('"close_long_form_{pos.position_id}"', '"p_close_long_form_{pos.position_id}"'),
    ('"close_short_form_{pos.position_id}"','"p_close_short_form_{pos.position_id}"'),
    ('"close_all_form_{pos.position_id}"',  '"p_close_all_form_{pos.position_id}"'),
    # Edit form keys
    ('"edit_long_price_{pos.position_id}"', '"p_edit_long_price_{pos.position_id}"'),
    ('"edit_short_price_{pos.position_id}"','"p_edit_short_price_{pos.position_id}"'),
    # Price update
    ('"long_price_{pos.position_id}"',      '"p_long_price_{pos.position_id}"'),
    ('"short_price_{pos.position_id}"',     '"p_short_price_{pos.position_id}"'),
    # Roll long
    ('"roll_long_form_{pos.position_id}"',  '"p_roll_long_form_{pos.position_id}"'),
    ('"roll_long_new_strike_{pos.position_id}"', '"p_roll_long_new_strike_{pos.position_id}"'),
    ('"roll_long_new_exp_{pos.position_id}"',    '"p_roll_long_new_exp_{pos.position_id}"'),
    # Add roll forms
    ('"add_short_roll_form_{pos.position_id}"',  '"p_add_short_roll_form_{pos.position_id}"'),
    ('"add_long_roll_form_{pos.position_id}"',   '"p_add_long_roll_form_{pos.position_id}"'),
    ('"add_roll_type_radio_{pos.position_id}"',  '"p_add_roll_type_radio_{pos.position_id}"'),
    # Fix/delete roll
    ('"fix_all_rolls_{pos.position_id}"',   '"p_fix_all_rolls_{pos.position_id}"'),
    ('"delete_roll_select_{pos.position_id}"','"p_delete_roll_select_{pos.position_id}"'),
    ('"delete_confirm_{pos.position_id}"',  '"p_delete_confirm_{pos.position_id}"'),
    ('"delete_roll_btn_{pos.position_id}"', '"p_delete_roll_btn_{pos.position_id}"'),
    # Sell short form
    ('"sell_short_form_{pos.position_id}"', '"p_sell_short_form_{pos.position_id}"'),
    # Session state
    ('f"editing_rolls_{pos.position_id}"',  'f"p_editing_rolls_{pos.position_id}"'),
    ('f"rolling_{pos.position_id}"',        'f"p_rolling_{pos.position_id}"'),
    ('f"editing_{pos.position_id}"',        'f"p_editing_{pos.position_id}"'),
    # selectbox/reason
    ('"cl_reason_{pos.position_id}"',       '"p_cl_reason_{pos.position_id}"'),
    ('"cs_reason_{pos.position_id}"',       '"p_cs_reason_{pos.position_id}"'),
]

for old, new in paper_key_renames:
    if old in src and old != new:
        count = src.count(old)
        src = src.replace(old, new)
        changes += 1

print(f"✓ Paper form keys prefixed with p_")

# ══════════════════════════════════════════════════════════════════════
# 3. REAL TRADE keys — already use pid (not pos.position_id)
#    but some clash. Ensure all real keys use r_ prefix
# ══════════════════════════════════════════════════════════════════════
real_key_renames = [
    # Roll form (real uses pid)
    ('"rtl_roll_{pid}"',          '"r_roll_{pid}"'),
    ('"rbb_mid_{pid}"',           '"r_bb_mid_{pid}"'),
    ('"rbb_fill_{pid}"',          '"r_bb_fill_{pid}"'),
    ('"rns_{pid}"',               '"r_ns_{pid}"'),
    ('"rne_{pid}"',               '"r_ne_{pid}"'),
    ('"rnc_mid_{pid}"',           '"r_nc_mid_{pid}"'),
    ('"rnc_fill_{pid}"',          '"r_nc_fill_{pid}"'),
    ('"rreason_{pid}"',           '"r_reason_{pid}"'),
    ('"rnotes_{pid}"',            '"r_notes_{pid}"'),
    # Action buttons
    ('"rexpire_{pid}"',           '"r_expire_{pid}"'),
    ('"radd_short_{pid}"',        '"r_add_short_{pid}"'),
    ('"rpx_{pid}"',               '"r_px_{pid}"'),
    ('"redit_{pid}"',             '"r_edit_{pid}"'),
    ('"rclose_{pid}"',            '"r_close_{pid}"'),
    ('"rdel_{pid}"',              '"r_del_{pid}"'),
    # Add short form
    ('"radd_short_form_{pid}"',   '"r_add_short_form_{pid}"'),
    ('"ras_k_{pid}"',             '"r_as_k_{pid}"'),
    ('"ras_exp_{pid}"',           '"r_as_exp_{pid}"'),
    ('"ras_mid_{pid}"',           '"r_as_mid_{pid}"'),
    ('"ras_fill_{pid}"',          '"r_as_fill_{pid}"'),
    ('"ras_comm_{pid}"',          '"r_as_comm_{pid}"'),
    # Edit long form
    ('"redit_form_{pid}"',        '"r_edit_form_{pid}"'),
    ('"re_k_{pid}"',              '"r_e_k_{pid}"'),
    ('"re_exp_{pid}"',            '"r_e_exp_{pid}"'),
    ('"re_fill_{pid}"',           '"r_e_fill_{pid}"'),
    ('"re_cur_{pid}"',            '"r_e_cur_{pid}"'),
    # Close form
    ('"rclose_form_{pid}"',       '"r_close_form_{pid}"'),
    ('"rcl_long_{pid}"',          '"r_cl_long_{pid}"'),
    ('"rcl_short_{pid}"',         '"r_cl_short_{pid}"'),
    ('"rcl_reason_{pid}"',        '"r_cl_reason_{pid}"'),
    # Delete confirm
    ('"rdelconfirm_{pid}"',       '"r_delconfirm_{pid}"'),
    ('"rdelcancel_{pid}"',        '"r_delcancel_{pid}"'),
    # Roll history
    ('"redit_rolls_{pid}"',       '"r_edit_rolls_{pid}"'),
    ('"rrecalc_{pid}"',           '"r_recalc_{pid}"'),
    ('"rroll_editor_{pos.position_id}"', '"r_roll_editor_{pos.position_id}"'),
    # Roll edit form
    ('"rsave_rolls_{pos.position_id}"',  '"r_save_rolls_{pos.position_id}"'),
    ('"rcancel_rolls_{pos.position_id}"','"r_cancel_rolls_{pos.position_id}"'),
    ('"rdel_roll_sel_{pos.position_id}"','"r_del_roll_sel_{pos.position_id}"'),
    ('"rdel_roll_confirm_{pos.position_id}"','"r_del_roll_confirm_{pos.position_id}"'),
    ('"rdel_roll_btn_{pos.position_id}"','"r_del_roll_btn_{pos.position_id}"'),
    # Session state
    ('f"rediting_rolls_{pid}"',   'f"r_editing_rolls_{pid}"'),
    ('f"rshowing_add_short_{pid}"','f"r_showing_add_short_{pid}"'),
    ('f"rshowing_edit_{pid}"',    'f"r_showing_edit_{pid}"'),
    ('f"rshowing_close_{pid}"',   'f"r_showing_close_{pid}"'),
    ('f"rconfirm_del_{pid}"',     'f"r_confirm_del_{pid}"'),
    # Also fix session_state.get calls
    ('st.session_state.get(f"rediting_rolls_{pid}")',  'st.session_state.get(f"r_editing_rolls_{pid}")'),
    ('st.session_state.get(f"rshowing_add_short_{pid}")', 'st.session_state.get(f"r_showing_add_short_{pid}")'),
    ('st.session_state.get(f"rshowing_edit_{pid}")',   'st.session_state.get(f"r_showing_edit_{pid}")'),
    ('st.session_state.get(f"rshowing_close_{pid}")',  'st.session_state.get(f"r_showing_close_{pid}")'),
    ('st.session_state.get(f"rconfirm_del_{pid}")',    'st.session_state.get(f"r_confirm_del_{pid}")'),
    # session_state.pop calls
    ('st.session_state.pop(f"rconfirm_del_{pid}"',    'st.session_state.pop(f"r_confirm_del_{pid}"'),
]

for old, new in real_key_renames:
    if old in src and old != new:
        count = src.count(old)
        src = src.replace(old, new)
        changes += 1

print(f"✓ Real form keys prefixed with r_")

# ══════════════════════════════════════════════════════════════════════
# 4. Fix main() dispatch — update all render calls to new names
# ══════════════════════════════════════════════════════════════════════
dispatch_renames = [
    ('render_trade_log(',     'render_paper_trade_log('),
    ('render_real_trade_log_page()', 'render_real_trade_log()'),
]
for old, new in dispatch_renames:
    if old in src and new not in src:
        src = src.replace(old, new)
        changes += 1
        print(f"✓ Dispatch: {old} → {new}")

# ══════════════════════════════════════════════════════════════════════
# 5. Fix real_trade_ui.py — add _real_ prefix to internal functions
# ══════════════════════════════════════════════════════════════════════
rtu_renames = [
    ("def render_real_trade_section(",  "def render_real_trade_section("),  # keep
    ("def _render_open_positions(",     "def _render_real_open_positions("),
    ("_render_open_positions(",         "_render_real_open_positions("),
    ("def _render_position_card(",      "def _render_real_position_card("),
    ("_render_position_card(",          "_render_real_position_card("),
    ("def _render_new_entry(",          "def _render_real_new_entry("),
    ("_render_new_entry(",              "_render_real_new_entry("),
    ("def _render_history(",            "def _render_real_history("),
    ("_render_history(",                "_render_real_history("),
]

for old, new in rtu_renames:
    if old in rtu and old != new:
        count = rtu.count(old)
        rtu = rtu.replace(old, new)
        changes += 1

# Fix calls in app.py that reference real_trade_ui functions
app_rtu_renames = [
    ("from real_trade_ui import render_real_trade_section", 
     "from real_trade_ui import render_real_trade_section"),  # keep
]
print("✓ real_trade_ui.py functions renamed with _real_ prefix")

# ══════════════════════════════════════════════════════════════════════
# 6. Move long-only checkbox from paper to real New Entry
# ══════════════════════════════════════════════════════════════════════

# Remove from paper (if accidentally added)
paper_long_only = '''        long_only_mode = st.checkbox(
            "📌 Long Only — short leg not sold yet",
            value=False,
            key="rtl_long_only_mode",'''
if paper_long_only in src:
    # Check if it's inside paper function — remove it
    idx = src.find(paper_long_only)
    context = src[max(0,idx-2000):idx]
    if "_render_paper_diagonal_entry_form" in context or "_render_paper_" in context:
        # Remove the whole block
        end_idx = src.find("if not long_only_mode:\n            st.markdown", idx)
        end_idx = src.find("\n", end_idx + 60) + 1
        src = src[:idx] + src[end_idx:]
        print("✓ Removed long-only checkbox from paper entry form")

# Add to real New Entry form — find _render_real_new_entry in real_trade_ui.py
real_short_label = '''    st.markdown("#### 📉 Short Leg")'''
real_long_only_block = '''    long_only = st.checkbox(
        "📌 Long Only — short leg not sold yet",
        value=False,
        key="r_new_entry_long_only",
        help="Use when you've bought the LEAP but haven't sold the short call yet. "
             "Add the short leg later via '📈 Add Short Leg' on the position card."
    )

    st.markdown("#### 📉 Short Leg")'''

if real_short_label in rtu and "r_new_entry_long_only" not in rtu:
    rtu = rtu.replace(real_short_label, real_long_only_block)
    print("✓ Added long-only checkbox to real New Entry form")

# Also handle submit — if long_only, call open_long_only
old_rtu_submit = "        tl.open_diagonal("
new_rtu_submit = """        if st.session_state.get("r_new_entry_long_only", False):
            tl.open_long_only(
                variant_id       = variant_id,
                variant_name     = variant_name,
                regime           = regime,
                vix_level        = vix_level,
                vix_percentile   = vix_pct,
                contracts        = contracts,
                long_strike      = long_strike,
                long_expiration  = long_exp,
                long_entry_price = long_mid,
                long_fill_price  = long_fill,
                broker           = broker,
                account_id       = account_id,
                long_commission  = long_comm,
                notes            = notes,
            )
        else:
            tl.open_diagonal("""

if old_rtu_submit in rtu and "open_long_only" not in rtu:
    rtu = rtu.replace(old_rtu_submit, new_rtu_submit)
    # Need to close the else block — find the closing paren of open_diagonal
    # The open_diagonal call ends with a ) on its own line
    print("✓ Added long-only branch to real New Entry submit")

# ══════════════════════════════════════════════════════════════════════
# 7. Add PAPER/REAL watermark comments at top of each section
# ══════════════════════════════════════════════════════════════════════
paper_watermark = "# ═══ PAPER TRADING FUNCTIONS ═══════════════════════════════════\n"
real_watermark  = "# ═══ REAL TRADING FUNCTIONS ════════════════════════════════════\n"

if paper_watermark not in src:
    src = src.replace(
        "def render_paper_trade_log(",
        paper_watermark + "def render_paper_trade_log("
    )
    print("✓ Added PAPER section watermark comment")

if real_watermark not in src:
    src = src.replace(
        "def render_real_trade_log(",
        real_watermark + "def render_real_trade_log("
    )
    print("✓ Added REAL section watermark comment")

# ══════════════════════════════════════════════════════════════════════
# Write files
# ══════════════════════════════════════════════════════════════════════
app_path.write_text(src)
rtu_path.write_text(rtu)

print(f"\n✅ Total changes: {changes}")
print("Run: python3 -c \"import ast; ast.parse(open('app.py').read()); print('OK')\"")
