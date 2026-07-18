"""
App Integration Patch for Google Drive Backup
==============================================

This file shows the code changes needed to integrate Google Drive backup
into your existing VIX 5W Suite app.py.

Copy the relevant sections into your app.py.

Author: VIX 5W Suite
"""

# ============================================================================
# SECTION 1: Add these imports at the top of app.py
# ============================================================================

# Add after other imports:
try:
    from gdrive_backup import GDriveBackupManager
    from backup_ui import (
        render_backup_page, 
        render_backup_status_card,
        render_backup_hook_on_trade,
        BACKUP_AVAILABLE
    )
except ImportError:
    BACKUP_AVAILABLE = False
    print("⚠️ Google Drive backup module not found. Install with: pip install google-auth google-auth-oauthlib google-api-python-client")


# ============================================================================
# SECTION 2: Add backup page to your page selector
# ============================================================================

# Find your page selector in the sidebar and add "☁️ Backup Manager"
# Example:

def get_pages():
    """Get available pages based on installed modules"""
    pages = [
        "📊 Dashboard",
        "📝 Position Manager", 
        "🔄 Signal Generator",
        "🔬 Backtester",
        "📈 Performance"
    ]
    
    # Add backup page if module is available
    if BACKUP_AVAILABLE:
        pages.append("☁️ Backup Manager")
    
    return pages


# ============================================================================
# SECTION 3: Add page routing in your main function
# ============================================================================

# Add this to your page routing logic:

def route_to_page(page: str):
    """Route to the selected page"""
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "📝 Position Manager":
        render_position_manager()
    elif page == "🔄 Signal Generator":
        render_signal_generator()
    elif page == "🔬 Backtester":
        render_backtester()
    elif page == "📈 Performance":
        render_performance()
    elif page == "☁️ Backup Manager" and BACKUP_AVAILABLE:
        render_backup_page()
    else:
        st.error(f"Unknown page: {page}")


# ============================================================================
# SECTION 4: Add sidebar backup status widget
# ============================================================================

# Add this to your sidebar section (after page selector):

def render_sidebar():
    """Render sidebar with all controls"""
    st.sidebar.title("VIX 5% Weekly Suite")
    
    # Page selector
    pages = get_pages()
    page = st.sidebar.selectbox("Navigation", pages)
    
    st.sidebar.divider()
    
    # Add backup status widget
    if BACKUP_AVAILABLE:
        render_backup_status_card()
    
    # ... rest of your sidebar code
    
    return page


# ============================================================================
# SECTION 5: Add backup hook after trade recording
# ============================================================================

# In your trade recording function (e.g., in position_ui.py or trade_log.py):

def record_trade(trade_data: dict):
    """Record a new trade"""
    # ... your existing trade recording logic ...
    
    # Add this at the end to trigger auto-backup
    if BACKUP_AVAILABLE:
        try:
            render_backup_hook_on_trade()
        except Exception as e:
            print(f"Auto-backup check failed: {e}")
    
    return True


# ============================================================================
# SECTION 6: Alternative - Minimal integration (sidebar button only)
# ============================================================================

# If you want a simpler integration, just add this to your sidebar:

def add_simple_backup_button():
    """Add a simple backup button to sidebar"""
    if not BACKUP_AVAILABLE:
        return
    
    st.sidebar.divider()
    st.sidebar.subheader("☁️ Backup")
    
    if st.sidebar.button("Backup Now", key="quick_backup"):
        with st.spinner("Backing up to Google Drive..."):
            try:
                manager = GDriveBackupManager()
                success, msg = manager.backup()
                if success:
                    st.sidebar.success("✅ Backup complete!")
                else:
                    st.sidebar.error(f"❌ {msg}")
            except Exception as e:
                st.sidebar.error(f"❌ Backup failed: {e}")


# ============================================================================
# SECTION 7: Complete app.py example structure
# ============================================================================

"""
Here's how your app.py should be structured:

import streamlit as st
import os
from datetime import datetime

# Your existing imports...
from trade_log import TradeLog
from regime_detector import RegimeDetector
from position_ui import render_position_manager
# ... etc

# NEW: Add backup imports
try:
    from gdrive_backup import GDriveBackupManager
    from backup_ui import (
        render_backup_page, 
        render_backup_status_card,
        render_backup_hook_on_trade,
        BACKUP_AVAILABLE
    )
except ImportError:
    BACKUP_AVAILABLE = False

st.set_page_config(
    page_title="VIX 5% Weekly Suite",
    page_icon="📈",
    layout="wide"
)

def main():
    # Sidebar
    st.sidebar.title("VIX 5% Weekly Suite")
    
    pages = [
        "📊 Dashboard",
        "📝 Position Manager", 
        "🔄 Signal Generator",
        "🔬 Backtester",
    ]
    if BACKUP_AVAILABLE:
        pages.append("☁️ Backup Manager")
    
    page = st.sidebar.selectbox("Navigation", pages)
    
    # NEW: Add backup status
    if BACKUP_AVAILABLE:
        render_backup_status_card()
    
    st.sidebar.divider()
    
    # Page routing
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "📝 Position Manager":
        render_position_manager()
    elif page == "☁️ Backup Manager" and BACKUP_AVAILABLE:
        render_backup_page()
    # ... other pages

if __name__ == "__main__":
    main()
"""
