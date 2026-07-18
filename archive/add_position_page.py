#!/usr/bin/env python3
"""
Patch script to add Position Manager page to app.py

This adds:
1. Import for position_ui module
2. "Position Manager" to page list
3. Page routing for Position Manager

Run: python3 add_position_page.py
"""

import re
from pathlib import Path
from datetime import datetime


def patch_app():
    app_path = Path("app.py")
    
    if not app_path.exists():
        print("❌ app.py not found in current directory")
        return False
    
    # Read current content
    content = app_path.read_text()
    
    # Create backup
    backup_name = f"app.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(backup_name).write_text(content)
    print(f"✅ Backup created: {backup_name}")
    
    modified = False
    
    # ================================================================
    # PATCH 1: Add import for position_ui
    # ================================================================
    if "from position_ui import" not in content:
        # Find a good place to add import (after other local imports)
        import_line = """
# Position management UI
try:
    from position_ui import (
        render_position_management_page,
        get_entry_suppressed_variants,
        should_suppress_entry,
    )
    POSITION_UI_AVAILABLE = True
except ImportError:
    POSITION_UI_AVAILABLE = False
    print("⚠️ position_ui.py not found - Position Manager disabled")
"""
        
        # Try to add after variant_generator import
        if "from variant_generator import" in content:
            content = content.replace(
                "from variant_generator import",
                f"from variant_generator import",
            )
            # Add after the last import block
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_idx = i + 1
            
            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
            print("   ✅ Added position_ui import")
            modified = True
        else:
            # Add at top after other imports
            if "import streamlit as st" in content:
                content = content.replace(
                    "import streamlit as st",
                    f"import streamlit as st\n{import_line}",
                )
                print("   ✅ Added position_ui import (fallback location)")
                modified = True
    else:
        print("   ⏭️ position_ui import already exists")
    
    # ================================================================
    # PATCH 2: Add Position Manager to page list
    # ================================================================
    page_patterns = [
        (r'page_options\s*=\s*\[([^\]]+)\]', 'page_options'),
        (r'pages\s*=\s*\[([^\]]+)\]', 'pages'),
        (r'"Dashboard",\s*"Backtester"', 'inline_pages'),
    ]
    
    position_manager_added = False
    
    for pattern, name in page_patterns:
        match = re.search(pattern, content)
        if match and '"Position Manager"' not in content:
            if name == 'inline_pages':
                # Add after existing pages
                old = match.group(0)
                new = old.rstrip('"') + '", "Position Manager"'
                content = content.replace(old, new)
                position_manager_added = True
                print("   ✅ Added Position Manager to page list")
                modified = True
                break
            else:
                # Add to list
                old_list = match.group(1)
                if '"Position Manager"' not in old_list:
                    new_list = old_list.rstrip().rstrip(',') + ', "Position Manager"'
                    content = content.replace(old_list, new_list)
                    position_manager_added = True
                    print("   ✅ Added Position Manager to page list")
                    modified = True
                    break
    
    if not position_manager_added and '"Position Manager"' not in content:
        print("   ⚠️ Could not find page list to modify - add manually")
    
    # ================================================================
    # PATCH 3: Add page routing for Position Manager
    # ================================================================
    routing_code = '''
    # ================================================================
    # PAGE: Position Manager
    # ================================================================
    if page == "Position Manager":
        if POSITION_UI_AVAILABLE:
            # Get current market state from regime
            current_regime = "CALM"
            current_vix = 20.0
            current_percentile = 0.5
            
            try:
                if 'regime_state' in dir() and regime_state:
                    current_regime = regime_state.regime.value.upper()
                    current_vix = regime_state.vix_level
                    current_percentile = regime_state.vix_percentile
            except:
                pass
            
            render_position_management_page(
                current_regime=current_regime,
                current_vix=current_vix,
                current_percentile=current_percentile,
            )
        else:
            st.error("Position Manager not available. Install position_ui.py first.")
        return
'''
    
    if 'page == "Position Manager"' not in content:
        # Find a good place to add (before or after Trade Explorer)
        if 'page == "Trade Explorer"' in content:
            # Add before Trade Explorer
            content = content.replace(
                'if page == "Trade Explorer"',
                f'{routing_code}\n    if page == "Trade Explorer"',
            )
            print("   ✅ Added Position Manager routing")
            modified = True
        elif 'if page == "Backtester"' in content:
            # Add after Backtester block (find the return)
            # This is trickier - let's add at end of main function
            if 'def main()' in content:
                # Find last return in main
                main_start = content.find('def main()')
                if main_start != -1:
                    # Add before the final return or at end of function
                    lines = content.split('\n')
                    insert_idx = len(lines) - 1
                    
                    for i in range(len(lines) - 1, main_start // 50, -1):
                        if 'return' in lines[i] and 'if page ==' not in lines[i-1]:
                            insert_idx = i
                            break
                    
                    # Insert the routing code
                    routing_lines = routing_code.split('\n')
                    for j, rline in enumerate(routing_lines):
                        lines.insert(insert_idx + j, rline)
                    
                    content = '\n'.join(lines)
                    print("   ✅ Added Position Manager routing (end of main)")
                    modified = True
    else:
        print("   ⏭️ Position Manager routing already exists")
    
    # ================================================================
    # Write modified content
    # ================================================================
    if modified:
        app_path.write_text(content)
        print(f"\n✅ app.py patched successfully!")
        print(f"   Backup at: {backup_name}")
    else:
        print("\n⚠️ No changes made - patches may already exist")
    
    return modified


def main():
    print("=" * 60)
    print("🔧 Adding Position Manager to app.py")
    print("=" * 60)
    print()
    
    patch_app()
    
    print()
    print("Next steps:")
    print("  1. Restart Streamlit: streamlit run app.py")
    print("  2. Look for 'Position Manager' in the sidebar")
    print("  3. Record trades, track positions, view performance")
    print()


if __name__ == "__main__":
    main()
