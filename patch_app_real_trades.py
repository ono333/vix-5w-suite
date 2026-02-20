"""
patch_app_real_trades.py
────────────────────────
Adds real trade log tab to app.py without touching paper trade logic.
Run once: python3 patch_app_real_trades.py
"""
from pathlib import Path
import re

app_path = Path("app.py")
src = app_path.read_text()

# ── 1. Add import at top
import_line = "from real_trade_ui import render_real_trade_section"
if "real_trade_ui" not in src:
    lines = src.splitlines(keepends=True)
    last_import = 0
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            last_import = i
    lines.insert(last_import + 1, import_line + "\n")
    src = "".join(lines)
    print("✓ Added import")
else:
    print("  Import already present")

# ── 2. Find the main tab structure and add Real Trades tab
# Look for st.tabs pattern
tab_patterns = [
    # Common pattern: tab1, tab2 = st.tabs(["...", "..."])
    (r'(tab\w+,\s*tab\w+)\s*=\s*st\.tabs\(\[([^\]]+)\]\)',
     'multi_tab'),
    # Or: tabs = st.tabs([...])
    (r'(tabs)\s*=\s*st\.tabs\(\[([^\]]+)\]\)',
     'tabs_var'),
]

found = False
for pattern, style in tab_patterns:
    m = re.search(pattern, src)
    if m:
        old_tabs_line = m.group(0)
        existing_tabs = m.group(2)

        if "Real Trades" not in existing_tabs:
            new_tabs_content = existing_tabs.rstrip() + ', "💵 Real Trades"'

            if style == 'multi_tab':
                old_vars = m.group(1)
                new_vars = old_vars.rstrip() + ", tab_real"
                new_tabs_line = old_tabs_line.replace(
                    old_vars, new_vars).replace(
                    existing_tabs, new_tabs_content)
            else:
                new_tabs_line = old_tabs_line.replace(
                    existing_tabs, new_tabs_content)

            src = src.replace(old_tabs_line, new_tabs_line)
            print(f"✓ Added 'Real Trades' tab to st.tabs")
            found = True
        else:
            print("  Real Trades tab already in st.tabs")
            found = True
        break

if not found:
    print("✗ Could not find st.tabs — will append section at end of main()")
    # Append before final if __name__ block
    append_code = """
# ── Real Trades section (injected by patch_app_real_trades.py)
st.markdown("---")
render_real_trade_section()
"""
    if "if __name__" in src:
        src = src.replace("if __name__", append_code + "\nif __name__")
    else:
        src += append_code
    print("✓ Appended render_real_trade_section() call")

# ── 3. If we have tab_real var, add the with block
if "tab_real" in src and "with tab_real:" not in src:
    # Find a good insertion point — after the last "with tab" block
    # Find end of last tab block
    last_with = src.rfind("\nwith tab")
    if last_with > 0:
        # Find the end of that block by looking for next same-level with/def/class
        after = src[last_with + 1:]
        next_block = re.search(r'\n(with |def |class |\Z)', after)
        if next_block:
            insert_at = last_with + 1 + next_block.start()
        else:
            insert_at = len(src)

        real_tab_block = """
with tab_real:
    render_real_trade_section()
"""
        src = src[:insert_at] + real_tab_block + src[insert_at:]
        print("✓ Added 'with tab_real:' block")

app_path.write_text(src)
print("\n✅ app.py patched successfully")
print("   Restart Streamlit to see changes:")
print("   sudo systemctl restart vix_app.service")
