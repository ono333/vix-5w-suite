"""
Streamlit UI Components for Google Drive Backup
================================================

Integrates with the VIX 5W Suite Streamlit app to provide:
- Backup status dashboard
- Manual backup triggers
- Restore functionality
- Configuration management

Usage in app.py:
    from backup_ui import render_backup_page
    
    # In your sidebar or page selector:
    if page == "🔄 Backup Manager":
        render_backup_page()

Author: VIX 5W Suite
"""

import streamlit as st
import os
from datetime import datetime
from typing import Optional
import json

# Import the backup manager
try:
    from gdrive_backup import GDriveBackupManager, BackupConfig, BackupState
    BACKUP_AVAILABLE = True
except ImportError:
    BACKUP_AVAILABLE = False


def get_backup_manager() -> Optional[GDriveBackupManager]:
    """Get or create cached backup manager instance"""
    if not BACKUP_AVAILABLE:
        return None
    
    if 'backup_manager' not in st.session_state:
        try:
            st.session_state.backup_manager = GDriveBackupManager()
        except Exception as e:
            st.error(f"Failed to initialize backup manager: {e}")
            return None
    
    return st.session_state.backup_manager


def render_backup_status_card():
    """Render a compact backup status card for the sidebar"""
    manager = get_backup_manager()
    
    if not manager:
        st.sidebar.warning("⚠️ Backup not configured")
        return
    
    status = manager.get_backup_status()
    
    with st.sidebar.expander("☁️ Backup Status", expanded=False):
        if status['last_backup_time']:
            last_backup = datetime.fromisoformat(status['last_backup_time'])
            time_ago = datetime.now() - last_backup
            
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600}h ago"
            else:
                time_str = f"{time_ago.seconds // 60}m ago"
            
            st.metric("Last Backup", time_str)
        else:
            st.metric("Last Backup", "Never")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", status['backup_count'])
        with col2:
            st.metric("Local", status['local_backups'])
        
        if st.button("🔄 Backup Now", key="sidebar_backup", use_container_width=True):
            with st.spinner("Backing up..."):
                success, msg = manager.backup(force=True)
                if success:
                    st.success("✅ Done!")
                else:
                    st.error(msg)


def render_backup_page():
    """Render the full backup management page"""
    st.title("☁️ Google Drive Backup Manager")
    
    if not BACKUP_AVAILABLE:
        st.error("""
        **Google Drive backup module not found.**
        
        Please ensure `gdrive_backup.py` is in your project directory.
        
        Required packages:
        ```bash
        pip install google-auth google-auth-oauthlib google-api-python-client
        ```
        """)
        return
    
    manager = get_backup_manager()
    if not manager:
        st.error("Failed to initialize backup manager")
        return
    
    # Status overview
    status = manager.get_backup_status()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status['last_backup_time']:
            last_backup = datetime.fromisoformat(status['last_backup_time'])
            st.metric("Last Backup", last_backup.strftime("%Y-%m-%d %H:%M"))
        else:
            st.metric("Last Backup", "Never")
    
    with col2:
        st.metric("Total Backups", status['backup_count'])
    
    with col3:
        st.metric("Local Backups", status['local_backups'])
    
    with col4:
        conn_status = "🟢 Connected" if status['initialized'] else "🟡 Ready"
        if not status['credentials_exists']:
            conn_status = "🔴 No Credentials"
        st.metric("Status", conn_status)
    
    st.divider()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Backup & Restore", 
        "📋 Backup History", 
        "⚙️ Configuration",
        "📖 Setup Guide"
    ])
    
    # Tab 1: Backup & Restore
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Create Backup")
            
            force_backup = st.checkbox("Force backup (even if no changes)", value=False)
            
            if st.button("🔄 Backup Now", type="primary", use_container_width=True):
                with st.spinner("Creating backup..."):
                    try:
                        success, msg = manager.backup(force=force_backup)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"❌ Backup failed: {e}")
            
            st.caption("Backups are stored in Google Drive with automatic versioning.")
        
        with col2:
            st.subheader("Restore Backup")
            
            # Fetch available backups
            backups = []
            try:
                if status['credentials_exists']:
                    with st.spinner("Fetching backups..."):
                        backups = manager.list_backups(limit=20)
            except Exception as e:
                st.warning(f"Could not fetch backup list: {e}")
            
            if backups:
                backup_options = {
                    f"{b['name']} ({b['createdTime'][:10]})": b['id'] 
                    for b in backups
                }
                selected_backup = st.selectbox(
                    "Select backup to restore",
                    options=list(backup_options.keys()),
                    key="restore_select"
                )
                
                if selected_backup:
                    st.warning("⚠️ Restoring will overwrite current trade log!")
                    
                    if st.button("🔙 Restore Selected", type="secondary", use_container_width=True):
                        file_id = backup_options[selected_backup]
                        with st.spinner("Restoring..."):
                            success, msg = manager.restore(file_id=file_id)
                            if success:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
            else:
                st.info("No backups available. Create your first backup above.")
    
    # Tab 2: Backup History
    with tab2:
        st.subheader("Backup History")
        
        # Show local backup state history
        state = manager.state
        if state.backup_history:
            history_data = []
            for entry in reversed(state.backup_history[-20:]):
                history_data.append({
                    "Time": entry.get('time', 'Unknown')[:19],
                    "Filename": entry.get('filename', 'Unknown'),
                    "Hash": entry.get('hash', 'N/A')[:8] + "...",
                    "File ID": entry.get('file_id', 'N/A')[:15] + "..."
                })
            
            st.dataframe(history_data, use_container_width=True, hide_index=True)
        else:
            st.info("No backup history yet.")
        
        # Show cloud backups
        st.subheader("Cloud Backups (Google Drive)")
        
        if st.button("🔄 Refresh List"):
            st.rerun()
        
        try:
            backups = manager.list_backups(limit=30)
            if backups:
                cloud_data = []
                for b in backups:
                    cloud_data.append({
                        "Filename": b.get('name', 'Unknown'),
                        "Created": b.get('createdTime', 'Unknown')[:19],
                        "Size": f"{int(b.get('size', 0)) / 1024:.1f} KB"
                    })
                st.dataframe(cloud_data, use_container_width=True, hide_index=True)
            else:
                st.info("No backups found in Google Drive.")
        except Exception as e:
            st.warning(f"Could not fetch cloud backups: {e}")
    
    # Tab 3: Configuration
    with tab3:
        st.subheader("Backup Configuration")
        
        config = manager.config
        
        with st.form("backup_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_versions = st.number_input(
                    "Max backup versions",
                    min_value=5,
                    max_value=100,
                    value=config.max_versions,
                    help="Number of backup versions to keep in Google Drive"
                )
                
                auto_interval = st.number_input(
                    "Auto backup interval (hours)",
                    min_value=1,
                    max_value=168,
                    value=config.auto_backup_interval_hours,
                    help="Hours between automatic backups"
                )
            
            with col2:
                backup_on_change = st.checkbox(
                    "Only backup when changed",
                    value=config.backup_on_change_only,
                    help="Skip backup if file hasn't changed"
                )
                
                folder_name = st.text_input(
                    "Google Drive folder name",
                    value=config.folder_name,
                    help="Name of the backup folder in Google Drive"
                )
            
            submitted = st.form_submit_button("💾 Save Configuration", use_container_width=True)
            
            if submitted:
                config.max_versions = max_versions
                config.auto_backup_interval_hours = auto_interval
                config.backup_on_change_only = backup_on_change
                config.folder_name = folder_name
                config.save()
                st.success("✅ Configuration saved!")
        
        st.divider()
        
        st.subheader("Paths")
        st.code(f"""
Trade Log:      {config.trade_log_path}
Credentials:    {config.credentials_path}
Local Backups:  {config.local_backup_dir}
Backup State:   {config.backup_state_path}
        """)
    
    # Tab 4: Setup Guide
    with tab4:
        st.subheader("📖 Google Drive API Setup Guide")
        
        st.markdown("""
        ### Step 1: Create Google Cloud Project
        
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Name it something like "VIX Suite Backup"
        
        ### Step 2: Enable Google Drive API
        
        1. In the Cloud Console, go to **APIs & Services > Library**
        2. Search for "Google Drive API"
        3. Click **Enable**
        
        ### Step 3: Create Service Account (Recommended for Automation)
        
        1. Go to **IAM & Admin > Service Accounts**
        2. Click **Create Service Account**
        3. Name: `vix-suite-backup`
        4. Click **Create and Continue**
        5. Skip role assignment (click **Continue**)
        6. Click **Done**
        7. Click on the new service account
        8. Go to **Keys** tab
        9. Click **Add Key > Create new key > JSON**
        10. Download the JSON file
        11. Save it as:
        """)
        
        st.code("~/.vix_suite/gdrive_credentials.json")
        
        st.markdown("""
        ### Step 4: Test Connection
        
        Run in terminal:
        ```bash
        python gdrive_backup.py --setup
        ```
        
        ### Step 5: Set Up Automation (Optional)
        
        #### On Mac (launchd):
        ```bash
        # Create LaunchAgent plist
        python setup_automation.py --platform mac
        
        # Load it
        launchctl load ~/Library/LaunchAgents/com.vix5w.gdrive_backup.plist
        ```
        
        #### On Ubuntu (cron):
        ```bash
        # Add to crontab
        python setup_automation.py --platform ubuntu
        
        # Or manually:
        crontab -e
        # Add: 0 */6 * * * cd /path/to/vix-5w-suite && python gdrive_backup.py --auto
        ```
        
        ### Troubleshooting
        
        - **"Credentials file not found"**: Make sure the JSON file is saved to `~/.vix_suite/gdrive_credentials.json`
        - **"Failed to initialize"**: Check that Google Drive API is enabled
        - **"Permission denied"**: Service account needs access - share the backup folder with the service account email
        """)


def render_backup_hook_on_trade():
    """
    Hook to call after recording a trade to trigger backup.
    Call this in your trade recording flow.
    """
    manager = get_backup_manager()
    if manager and manager.should_backup():
        success, msg = manager.auto_backup()
        if success and "successful" in msg.lower():
            st.toast("☁️ Auto-backup completed", icon="✅")


# Convenience function to add backup page to existing app
def add_backup_page_to_app():
    """
    Returns code snippet to add backup page to app.py
    """
    return '''
# Add to your imports at top of app.py:
from backup_ui import render_backup_page, render_backup_status_card, render_backup_hook_on_trade

# Add to your sidebar page selector:
pages = ["📊 Dashboard", "📝 Position Manager", "🔬 Backtester", "☁️ Backup Manager"]

# Add to your page routing:
elif page == "☁️ Backup Manager":
    render_backup_page()

# Add sidebar status widget (in sidebar section):
render_backup_status_card()

# After recording trades, call:
render_backup_hook_on_trade()
'''


if __name__ == "__main__":
    # For testing the UI independently
    st.set_page_config(
        page_title="VIX 5W Suite - Backup Manager",
        page_icon="☁️",
        layout="wide"
    )
    render_backup_page()
