"""
Backup Manager for VIX 5% Weekly Suite

Provides automatic backup functionality for trade data:
1. Timestamped local backups before each save
2. Cloud sync support (Dropbox/Google Drive folders)
3. Export to CSV for human-readable records
4. Restore from any backup point

Usage:
    from backup_manager import BackupManager
    backup_mgr = BackupManager()
    backup_mgr.backup_now()  # Manual backup
    backup_mgr.restore_latest()  # Restore most recent
"""

import json
import shutil
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import os


class BackupManager:
    """Manages backups for trade_log.json and other critical data."""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_local_backups: int = 50,
        cloud_sync_paths: Optional[List[str]] = None,
    ):
        """
        Initialize backup manager.
        
        Args:
            data_dir: Base directory for data (default: ~/.vix_suite)
            max_local_backups: Maximum number of local backups to keep
            cloud_sync_paths: List of cloud sync folders (Dropbox, Google Drive, etc.)
        """
        self.data_dir = Path(data_dir or os.path.expanduser("~/.vix_suite"))
        self.backup_dir = self.data_dir / "backups"
        self.cloud_sync_paths = cloud_sync_paths or []
        self.max_local_backups = max_local_backups
        
        # Auto-detect cloud sync folders
        self._detect_cloud_folders()
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to backup
        self.critical_files = [
            "trade_log.json",
            "current_signal_batch.json",
            "regime_history.json",
        ]
    
    def _detect_cloud_folders(self):
        """Auto-detect common cloud sync folders."""
        home = Path.home()
        cloud_candidates = [
            home / "Dropbox" / "VIX_Suite_Backup",
            home / "Google Drive" / "VIX_Suite_Backup",
            home / "OneDrive" / "VIX_Suite_Backup",
            home / "iCloud Drive" / "VIX_Suite_Backup",  # macOS
            Path("/mnt/gdrive/VIX_Suite_Backup"),  # Linux mounted
        ]
        
        for path in cloud_candidates:
            parent = path.parent
            if parent.exists() and str(path) not in self.cloud_sync_paths:
                # Parent exists (e.g., ~/Dropbox exists)
                self.cloud_sync_paths.append(str(path))
    
    def backup_now(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Create immediate backup of all critical files.
        
        Args:
            reason: Reason for backup (manual, pre_save, scheduled, etc.)
            
        Returns:
            Dict with backup details
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = self.backup_dir / f"backup_{timestamp}_{reason}"
        backup_folder.mkdir(parents=True, exist_ok=True)
        
        backed_up = []
        errors = []
        
        for filename in self.critical_files:
            source = self.data_dir / filename
            if source.exists():
                try:
                    dest = backup_folder / filename
                    shutil.copy2(source, dest)
                    backed_up.append(filename)
                except Exception as e:
                    errors.append(f"{filename}: {e}")
        
        # Create backup manifest
        manifest = {
            "timestamp": timestamp,
            "reason": reason,
            "files": backed_up,
            "errors": errors,
            "created_at": datetime.now().isoformat(),
        }
        
        manifest_path = backup_folder / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Sync to cloud if available
        cloud_synced = self._sync_to_cloud(backup_folder)
        manifest["cloud_synced"] = cloud_synced
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        return manifest
    
    def backup_before_save(self) -> str:
        """
        Call this BEFORE saving trade_log.json.
        Creates a pre-save backup automatically.
        
        Returns:
            Backup folder path
        """
        result = self.backup_now(reason="pre_save")
        return str(self.backup_dir / f"backup_{result['timestamp']}_pre_save")
    
    def _sync_to_cloud(self, backup_folder: Path) -> List[str]:
        """Sync backup to cloud folders."""
        synced = []
        
        for cloud_path in self.cloud_sync_paths:
            cloud_dir = Path(cloud_path)
            try:
                cloud_dir.mkdir(parents=True, exist_ok=True)
                dest = cloud_dir / backup_folder.name
                
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(backup_folder, dest)
                synced.append(str(cloud_dir))
            except Exception as e:
                print(f"Cloud sync failed for {cloud_path}: {e}")
        
        # Also copy latest to a fixed "latest" folder for easy access
        for cloud_path in self.cloud_sync_paths:
            try:
                latest_dir = Path(cloud_path) / "latest"
                if latest_dir.exists():
                    shutil.rmtree(latest_dir)
                shutil.copytree(backup_folder, latest_dir)
            except Exception:
                pass
        
        return synced
    
    def _cleanup_old_backups(self):
        """Remove old local backups, keeping only max_local_backups most recent."""
        backups = sorted(self.backup_dir.glob("backup_*"), key=lambda p: p.stat().st_mtime)
        
        while len(backups) > self.max_local_backups:
            oldest = backups.pop(0)
            try:
                shutil.rmtree(oldest)
            except Exception as e:
                print(f"Failed to remove old backup {oldest}: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with details."""
        backups = []
        
        for backup_folder in sorted(self.backup_dir.glob("backup_*"), reverse=True):
            manifest_path = backup_folder / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    manifest["path"] = str(backup_folder)
                    backups.append(manifest)
                except Exception:
                    # Fallback for backups without manifest
                    backups.append({
                        "path": str(backup_folder),
                        "timestamp": backup_folder.name.split("_")[1],
                        "files": [f.name for f in backup_folder.glob("*.json")],
                    })
            else:
                backups.append({
                    "path": str(backup_folder),
                    "timestamp": backup_folder.name.split("_")[1] if "_" in backup_folder.name else "unknown",
                    "files": [f.name for f in backup_folder.glob("*.json")],
                })
        
        return backups
    
    def restore_backup(self, backup_path: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Restore from a specific backup.
        
        Args:
            backup_path: Path to backup folder
            confirm: Must be True to actually restore
            
        Returns:
            Dict with restore details
        """
        backup_folder = Path(backup_path)
        
        if not backup_folder.exists():
            return {"success": False, "error": "Backup folder not found"}
        
        if not confirm:
            return {
                "success": False,
                "error": "Set confirm=True to actually restore",
                "preview": [f.name for f in backup_folder.glob("*.json") if f.name != "manifest.json"]
            }
        
        # Create a backup of current state before restoring
        self.backup_now(reason="pre_restore")
        
        restored = []
        errors = []
        
        for json_file in backup_folder.glob("*.json"):
            if json_file.name == "manifest.json":
                continue
            
            try:
                dest = self.data_dir / json_file.name
                shutil.copy2(json_file, dest)
                restored.append(json_file.name)
            except Exception as e:
                errors.append(f"{json_file.name}: {e}")
        
        return {
            "success": len(errors) == 0,
            "restored": restored,
            "errors": errors,
        }
    
    def restore_latest(self, confirm: bool = False) -> Dict[str, Any]:
        """Restore from the most recent backup."""
        backups = self.list_backups()
        
        if not backups:
            return {"success": False, "error": "No backups found"}
        
        return self.restore_backup(backups[0]["path"], confirm=confirm)
    
    def export_trades_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export trade log to CSV for human-readable backup.
        
        Returns:
            Path to exported CSV
        """
        trade_log_path = self.data_dir / "trade_log.json"
        
        if not trade_log_path.exists():
            raise FileNotFoundError("trade_log.json not found")
        
        with open(trade_log_path, 'r') as f:
            data = json.load(f)
        
        output_path = output_path or str(
            self.data_dir / f"trade_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        # Export diagonal positions
        diagonals = data.get("diagonal_positions", {})
        
        if diagonals:
            with open(output_path, 'w', newline='') as f:
                # Flatten the data for CSV
                fieldnames = [
                    "position_id", "variant_id", "variant_name", "status",
                    "entry_date", "entry_regime", "entry_vix_level", "contracts",
                    "long_strike", "long_expiration", "long_entry_price",
                    "current_short_strike", "current_short_expiration", "current_short_credit",
                    "total_rolls", "total_credits", "total_commissions", "notes"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for pos_id, pos in diagonals.items():
                    # Get current short leg info
                    short_legs = pos.get("short_legs", [])
                    current_short = None
                    for leg in short_legs:
                        if leg.get("status") == "open":
                            current_short = leg
                            break
                    
                    row = {
                        "position_id": pos.get("position_id"),
                        "variant_id": pos.get("variant_id"),
                        "variant_name": pos.get("variant_name"),
                        "status": pos.get("status"),
                        "entry_date": pos.get("entry_date"),
                        "entry_regime": pos.get("entry_regime"),
                        "entry_vix_level": pos.get("entry_vix_level"),
                        "contracts": pos.get("contracts"),
                        "long_strike": pos.get("long_strike"),
                        "long_expiration": pos.get("long_expiration"),
                        "long_entry_price": pos.get("long_entry_price"),
                        "current_short_strike": current_short.get("strike") if current_short else "",
                        "current_short_expiration": current_short.get("expiration_date") if current_short else "",
                        "current_short_credit": current_short.get("entry_credit") if current_short else "",
                        "total_rolls": pos.get("total_rolls", 0),
                        "total_credits": pos.get("total_short_credits", 0) + pos.get("total_roll_credits", 0),
                        "total_commissions": pos.get("total_commissions", 0),
                        "notes": pos.get("notes", ""),
                    }
                    writer.writerow(row)
        
        return output_path
    
    def get_status(self) -> Dict[str, Any]:
        """Get backup system status."""
        backups = self.list_backups()
        
        return {
            "data_dir": str(self.data_dir),
            "backup_dir": str(self.backup_dir),
            "total_backups": len(backups),
            "latest_backup": backups[0] if backups else None,
            "cloud_sync_paths": self.cloud_sync_paths,
            "cloud_sync_enabled": len(self.cloud_sync_paths) > 0,
            "max_local_backups": self.max_local_backups,
        }


# Singleton instance
_backup_manager: Optional[BackupManager] = None

def get_backup_manager() -> BackupManager:
    """Get global backup manager instance."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


# Convenience functions
def backup_now(reason: str = "manual") -> Dict[str, Any]:
    """Create immediate backup."""
    return get_backup_manager().backup_now(reason)

def restore_latest(confirm: bool = False) -> Dict[str, Any]:
    """Restore from latest backup."""
    return get_backup_manager().restore_latest(confirm)

def list_backups() -> List[Dict[str, Any]]:
    """List all backups."""
    return get_backup_manager().list_backups()
