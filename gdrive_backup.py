"""
Google Drive Backup Automation for VIX 5% Weekly Suite Trade Logs
=================================================================

Features:
- Automatic backup of trade_log.json to Google Drive
- Incremental backups with versioning (keeps last N versions)
- Cross-platform support (Mac & Ubuntu)
- Service account authentication (headless) or OAuth (interactive)
- Backup status tracking and notifications
- Integration with Streamlit UI

Setup:
1. Create Google Cloud project & enable Drive API
2. Create service account or OAuth credentials
3. Store credentials in ~/.vix_suite/gdrive_credentials.json
4. Run: python gdrive_backup.py --setup

Author: VIX 5W Suite
"""

import os
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gdrive_backup')


class AuthType(Enum):
    SERVICE_ACCOUNT = "service_account"
    OAUTH = "oauth"


@dataclass
class BackupConfig:
    """Configuration for Google Drive backups"""
    # Paths
    trade_log_path: str = "~/.vix_suite/trade_log.json"
    credentials_path: str = "~/.vix_suite/gdrive_credentials.json"
    token_path: str = "~/.vix_suite/gdrive_token.json"
    backup_state_path: str = "~/.vix_suite/backup_state.json"
    local_backup_dir: str = "~/.vix_suite/backups"
    
    # Google Drive settings
    folder_name: str = "VIX_5W_Suite_Backups"
    folder_id: Optional[str] = None  # Will be set after folder creation
    
    # Backup settings
    max_versions: int = 30  # Keep last 30 backups
    auto_backup_interval_hours: int = 6
    backup_on_change_only: bool = True
    
    # Auth type
    auth_type: str = "service_account"
    
    def expand_paths(self) -> 'BackupConfig':
        """Expand ~ in all paths"""
        return BackupConfig(
            trade_log_path=os.path.expanduser(self.trade_log_path),
            credentials_path=os.path.expanduser(self.credentials_path),
            token_path=os.path.expanduser(self.token_path),
            backup_state_path=os.path.expanduser(self.backup_state_path),
            local_backup_dir=os.path.expanduser(self.local_backup_dir),
            folder_name=self.folder_name,
            folder_id=self.folder_id,
            max_versions=self.max_versions,
            auto_backup_interval_hours=self.auto_backup_interval_hours,
            backup_on_change_only=self.backup_on_change_only,
            auth_type=self.auth_type
        )
    
    def save(self, path: Optional[str] = None):
        """Save config to JSON file"""
        save_path = path or os.path.expanduser("~/.vix_suite/backup_config.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Config saved to {save_path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'BackupConfig':
        """Load config from JSON file"""
        load_path = path or os.path.expanduser("~/.vix_suite/backup_config.json")
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()


@dataclass
class BackupState:
    """Tracks backup state for incremental backups"""
    last_backup_time: Optional[str] = None
    last_backup_hash: Optional[str] = None
    last_backup_file_id: Optional[str] = None
    backup_count: int = 0
    backup_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.backup_history is None:
            self.backup_history = []
    
    def save(self, path: str):
        """Save state to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BackupState':
        """Load state from JSON file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()


class GDriveBackupManager:
    """
    Manages Google Drive backups for VIX 5W Suite trade logs.
    
    Supports both service account (for automation) and OAuth (for interactive use).
    """
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = (config or BackupConfig.load()).expand_paths()
        self.state = BackupState.load(self.config.backup_state_path)
        self._service = None
        self._initialized = False
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _check_dependencies(self) -> Tuple[bool, str]:
        """Check if required packages are installed"""
        missing = []
        try:
            from google.oauth2 import service_account
        except ImportError:
            missing.append('google-auth')
        
        try:
            from googleapiclient.discovery import build
        except ImportError:
            missing.append('google-api-python-client')
        
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            missing.append('google-auth-oauthlib')
        
        if missing:
            return False, f"Missing packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
        return True, "All dependencies satisfied"
    
    def initialize(self) -> bool:
        """Initialize Google Drive API service"""
        deps_ok, deps_msg = self._check_dependencies()
        if not deps_ok:
            logger.error(deps_msg)
            return False
        
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        
        try:
            creds = None
            
            if self.config.auth_type == AuthType.SERVICE_ACCOUNT.value:
                # Service account authentication (headless/automated)
                if not os.path.exists(self.config.credentials_path):
                    logger.error(f"Credentials file not found: {self.config.credentials_path}")
                    return False
                
                creds = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path,
                    scopes=self.SCOPES
                )
                
            else:
                # OAuth authentication (interactive)
                if os.path.exists(self.config.token_path):
                    creds = Credentials.from_authorized_user_file(
                        self.config.token_path, self.SCOPES
                    )
                
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        if not os.path.exists(self.config.credentials_path):
                            logger.error(f"Credentials file not found: {self.config.credentials_path}")
                            return False
                        
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.config.credentials_path, self.SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                    
                    # Save the credentials for future runs
                    with open(self.config.token_path, 'w') as token:
                        token.write(creds.to_json())
            
            self._service = build('drive', 'v3', credentials=creds)
            self._initialized = True
            logger.info("Google Drive API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive API: {e}")
            return False
    
    def _ensure_folder(self) -> Optional[str]:
        """Ensure backup folder exists in Google Drive, create if needed"""
        if not self._initialized:
            if not self.initialize():
                return None
        
        # Check if we have a cached folder ID
        if self.config.folder_id:
            try:
                # Verify folder still exists
                self._service.files().get(fileId=self.config.folder_id).execute()
                return self.config.folder_id
            except Exception:
                logger.warning("Cached folder ID invalid, searching for folder...")
                self.config.folder_id = None
        
        # Search for existing folder
        query = f"name='{self.config.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self._service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = results.get('files', [])
        
        if folders:
            folder_id = folders[0]['id']
            logger.info(f"Found existing backup folder: {folder_id}")
        else:
            # Create new folder
            file_metadata = {
                'name': self.config.folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self._service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logger.info(f"Created new backup folder: {folder_id}")
        
        # Cache folder ID
        self.config.folder_id = folder_id
        self.config.save()
        
        return folder_id
    
    def backup(self, force: bool = False) -> Tuple[bool, str]:
        """
        Backup trade log to Google Drive.
        
        Args:
            force: If True, backup even if file hasn't changed
            
        Returns:
            Tuple of (success, message)
        """
        from googleapiclient.http import MediaFileUpload
        
        # Check if trade log exists
        if not os.path.exists(self.config.trade_log_path):
            return False, f"Trade log not found: {self.config.trade_log_path}"
        
        # Check if file has changed (skip if no changes and not forced)
        current_hash = self._get_file_hash(self.config.trade_log_path)
        if (not force and 
            self.config.backup_on_change_only and 
            current_hash == self.state.last_backup_hash):
            return True, "No changes detected, backup skipped"
        
        # Initialize API if needed
        if not self._initialized:
            if not self.initialize():
                return False, "Failed to initialize Google Drive API"
        
        # Ensure backup folder exists
        folder_id = self._ensure_folder()
        if not folder_id:
            return False, "Failed to create/find backup folder"
        
        # Create local backup first
        self._create_local_backup()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_log_{timestamp}.json"
        
        try:
            # Upload to Google Drive
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(
                self.config.trade_log_path,
                mimetype='application/json',
                resumable=True
            )
            
            file = self._service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            
            file_id = file.get('id')
            web_link = file.get('webViewLink', 'N/A')
            
            # Update state
            self.state.last_backup_time = datetime.now().isoformat()
            self.state.last_backup_hash = current_hash
            self.state.last_backup_file_id = file_id
            self.state.backup_count += 1
            self.state.backup_history.append({
                'time': self.state.last_backup_time,
                'file_id': file_id,
                'filename': filename,
                'hash': current_hash
            })
            
            # Keep only last N history entries
            if len(self.state.backup_history) > self.config.max_versions:
                self.state.backup_history = self.state.backup_history[-self.config.max_versions:]
            
            self.state.save(self.config.backup_state_path)
            
            # Cleanup old backups in Drive
            self._cleanup_old_backups(folder_id)
            
            msg = f"Backup successful: {filename} (ID: {file_id})"
            logger.info(msg)
            return True, msg
            
        except Exception as e:
            msg = f"Backup failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def _create_local_backup(self):
        """Create a local backup copy"""
        os.makedirs(self.config.local_backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            self.config.local_backup_dir,
            f"trade_log_{timestamp}.json"
        )
        
        shutil.copy2(self.config.trade_log_path, backup_path)
        logger.info(f"Local backup created: {backup_path}")
        
        # Cleanup old local backups
        self._cleanup_local_backups()
    
    def _cleanup_local_backups(self):
        """Remove old local backups, keeping only max_versions"""
        backup_dir = Path(self.config.local_backup_dir)
        if not backup_dir.exists():
            return
        
        backups = sorted(backup_dir.glob("trade_log_*.json"), reverse=True)
        for old_backup in backups[self.config.max_versions:]:
            old_backup.unlink()
            logger.info(f"Deleted old local backup: {old_backup}")
    
    def _cleanup_old_backups(self, folder_id: str):
        """Remove old backups from Google Drive, keeping only max_versions"""
        query = f"'{folder_id}' in parents and name contains 'trade_log_' and trashed=false"
        results = self._service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, createdTime)',
            orderBy='createdTime desc'
        ).execute()
        
        files = results.get('files', [])
        
        # Delete files beyond max_versions
        for old_file in files[self.config.max_versions:]:
            try:
                self._service.files().delete(fileId=old_file['id']).execute()
                logger.info(f"Deleted old Drive backup: {old_file['name']}")
            except Exception as e:
                logger.warning(f"Failed to delete old backup {old_file['name']}: {e}")
    
    def restore(self, file_id: Optional[str] = None, timestamp: Optional[str] = None) -> Tuple[bool, str]:
        """
        Restore trade log from Google Drive backup.
        
        Args:
            file_id: Specific file ID to restore
            timestamp: Timestamp to search for (format: YYYYMMDD_HHMMSS)
            
        Returns:
            Tuple of (success, message)
        """
        from googleapiclient.http import MediaIoBaseDownload
        import io
        
        if not self._initialized:
            if not self.initialize():
                return False, "Failed to initialize Google Drive API"
        
        # Find file to restore
        if file_id:
            target_file_id = file_id
        elif timestamp:
            folder_id = self._ensure_folder()
            if not folder_id:
                return False, "Failed to access backup folder"
            
            query = f"'{folder_id}' in parents and name contains 'trade_log_{timestamp}' and trashed=false"
            results = self._service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            files = results.get('files', [])
            
            if not files:
                return False, f"No backup found for timestamp: {timestamp}"
            target_file_id = files[0]['id']
        else:
            # Restore latest
            if not self.state.last_backup_file_id:
                return False, "No previous backup found"
            target_file_id = self.state.last_backup_file_id
        
        try:
            # Create backup of current file before restore
            if os.path.exists(self.config.trade_log_path):
                pre_restore_backup = self.config.trade_log_path + ".pre_restore"
                shutil.copy2(self.config.trade_log_path, pre_restore_backup)
                logger.info(f"Pre-restore backup created: {pre_restore_backup}")
            
            # Download and restore
            request = self._service.files().get_media(fileId=target_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Write to trade log path
            os.makedirs(os.path.dirname(self.config.trade_log_path), exist_ok=True)
            with open(self.config.trade_log_path, 'wb') as f:
                f.write(fh.getvalue())
            
            msg = f"Successfully restored backup from file ID: {target_file_id}"
            logger.info(msg)
            return True, msg
            
        except Exception as e:
            msg = f"Restore failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def list_backups(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List available backups from Google Drive"""
        if not self._initialized:
            if not self.initialize():
                return []
        
        folder_id = self._ensure_folder()
        if not folder_id:
            return []
        
        query = f"'{folder_id}' in parents and name contains 'trade_log_' and trashed=false"
        results = self._service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, createdTime, size, webViewLink)',
            orderBy='createdTime desc',
            pageSize=limit
        ).execute()
        
        return results.get('files', [])
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup status"""
        return {
            'initialized': self._initialized,
            'last_backup_time': self.state.last_backup_time,
            'last_backup_hash': self.state.last_backup_hash,
            'backup_count': self.state.backup_count,
            'folder_id': self.config.folder_id,
            'trade_log_exists': os.path.exists(self.config.trade_log_path),
            'credentials_exists': os.path.exists(self.config.credentials_path),
            'local_backups': len(list(Path(self.config.local_backup_dir).glob("trade_log_*.json"))) 
                            if os.path.exists(self.config.local_backup_dir) else 0
        }
    
    def should_backup(self) -> bool:
        """Check if a backup is due based on interval"""
        if not self.state.last_backup_time:
            return True
        
        last_backup = datetime.fromisoformat(self.state.last_backup_time)
        interval = timedelta(hours=self.config.auto_backup_interval_hours)
        
        return datetime.now() - last_backup > interval
    
    def auto_backup(self) -> Tuple[bool, str]:
        """Perform automatic backup if due"""
        if self.should_backup():
            return self.backup()
        return True, "Backup not due yet"


def setup_gdrive_backup():
    """Interactive setup for Google Drive backup"""
    print("\n" + "="*60)
    print("VIX 5W Suite - Google Drive Backup Setup")
    print("="*60 + "\n")
    
    config = BackupConfig.load()
    
    # Check dependencies
    manager = GDriveBackupManager(config)
    deps_ok, deps_msg = manager._check_dependencies()
    
    if not deps_ok:
        print(f"⚠️  {deps_msg}")
        print("\nInstall with:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        return
    
    print("✅ All dependencies installed\n")
    
    # Configuration
    print("Current Configuration:")
    print(f"  Trade log path: {config.trade_log_path}")
    print(f"  Credentials path: {config.credentials_path}")
    print(f"  Backup folder: {config.folder_name}")
    print(f"  Max versions: {config.max_versions}")
    print(f"  Auto backup interval: {config.auto_backup_interval_hours} hours")
    print(f"  Auth type: {config.auth_type}")
    
    # Check if credentials exist
    creds_path = os.path.expanduser(config.credentials_path)
    if not os.path.exists(creds_path):
        print(f"\n⚠️  Credentials file not found at: {creds_path}")
        print("\nTo set up credentials:")
        print("1. Go to Google Cloud Console (console.cloud.google.com)")
        print("2. Create a new project or select existing")
        print("3. Enable the Google Drive API")
        print("4. For service account (recommended for automation):")
        print("   - Create service account under 'IAM & Admin'")
        print("   - Download JSON key file")
        print("   - Save as: ~/.vix_suite/gdrive_credentials.json")
        print("5. For OAuth (interactive):")
        print("   - Create OAuth 2.0 credentials")
        print("   - Download client secrets JSON")
        print("   - Save as: ~/.vix_suite/gdrive_credentials.json")
        return
    
    print(f"\n✅ Credentials found at: {creds_path}")
    
    # Test connection
    print("\nTesting Google Drive connection...")
    if manager.initialize():
        print("✅ Successfully connected to Google Drive!")
        
        # Create/find backup folder
        folder_id = manager._ensure_folder()
        if folder_id:
            print(f"✅ Backup folder ready: {config.folder_name} (ID: {folder_id})")
        
        # Perform initial backup if trade log exists
        trade_log_path = os.path.expanduser(config.trade_log_path)
        if os.path.exists(trade_log_path):
            print("\nPerforming initial backup...")
            success, msg = manager.backup(force=True)
            if success:
                print(f"✅ {msg}")
            else:
                print(f"❌ {msg}")
        else:
            print(f"\n⚠️  Trade log not found at: {trade_log_path}")
            print("   Backup will run when trade log is created.")
    else:
        print("❌ Failed to connect to Google Drive")
        print("   Check your credentials and try again")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VIX 5W Suite Google Drive Backup")
    parser.add_argument('--setup', action='store_true', help='Run interactive setup')
    parser.add_argument('--backup', action='store_true', help='Perform backup')
    parser.add_argument('--force', action='store_true', help='Force backup even if no changes')
    parser.add_argument('--restore', type=str, help='Restore backup by file ID or "latest"')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--status', action='store_true', help='Show backup status')
    parser.add_argument('--auto', action='store_true', help='Run auto backup if due')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_gdrive_backup()
    elif args.backup:
        manager = GDriveBackupManager()
        success, msg = manager.backup(force=args.force)
        print(msg)
    elif args.restore:
        manager = GDriveBackupManager()
        file_id = None if args.restore == 'latest' else args.restore
        success, msg = manager.restore(file_id=file_id)
        print(msg)
    elif args.list:
        manager = GDriveBackupManager()
        backups = manager.list_backups()
        print("\nAvailable Backups:")
        print("-" * 60)
        for b in backups:
            created = b.get('createdTime', 'Unknown')
            print(f"  {b['name']:40} {created[:19]}")
        print("-" * 60)
    elif args.status:
        manager = GDriveBackupManager()
        status = manager.get_backup_status()
        print("\nBackup Status:")
        print("-" * 40)
        for key, value in status.items():
            print(f"  {key:25} {value}")
        print("-" * 40)
    elif args.auto:
        manager = GDriveBackupManager()
        success, msg = manager.auto_backup()
        print(msg)
    else:
        parser.print_help()
