#!/usr/bin/env python3
"""
Automation Setup for Google Drive Backup
=========================================

Sets up scheduled backups for both Mac (launchd) and Ubuntu (cron/systemd).

Usage:
    python setup_automation.py --platform mac
    python setup_automation.py --platform ubuntu
    python setup_automation.py --platform ubuntu --systemd  # Use systemd timer instead of cron
    
Author: VIX 5W Suite
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class AutomationSetup:
    """Cross-platform automation setup for Google Drive backups"""
    
    def __init__(self, project_dir: str = None, interval_hours: int = 6):
        self.project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))
        self.interval_hours = interval_hours
        self.python_path = sys.executable
        self.backup_script = os.path.join(self.project_dir, "gdrive_backup.py")
        
    def setup_mac_launchd(self) -> tuple:
        """
        Set up Mac launchd LaunchAgent for scheduled backups.
        
        Returns:
            Tuple of (success, message)
        """
        plist_name = "com.vix5w.gdrive_backup.plist"
        plist_dir = os.path.expanduser("~/Library/LaunchAgents")
        plist_path = os.path.join(plist_dir, plist_name)
        
        # Log file location
        log_dir = os.path.expanduser("~/.vix_suite/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Calculate interval in seconds
        interval_seconds = self.interval_hours * 3600
        
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vix5w.gdrive_backup</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{self.python_path}</string>
        <string>{self.backup_script}</string>
        <string>--auto</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{self.project_dir}</string>
    
    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/gdrive_backup.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/gdrive_backup_error.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
'''
        try:
            # Create LaunchAgents directory if needed
            os.makedirs(plist_dir, exist_ok=True)
            
            # Unload existing if present
            if os.path.exists(plist_path):
                subprocess.run(
                    ["launchctl", "unload", plist_path],
                    capture_output=True
                )
            
            # Write plist file
            with open(plist_path, 'w') as f:
                f.write(plist_content)
            
            # Load the LaunchAgent
            result = subprocess.run(
                ["launchctl", "load", plist_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Failed to load LaunchAgent: {result.stderr}"
            
            return True, f"""
✅ Mac launchd automation configured!

Plist Location: {plist_path}
Interval: Every {self.interval_hours} hours
Logs: {log_dir}/gdrive_backup.log

Commands:
  launchctl list | grep vix5w     # Check status
  launchctl unload {plist_path}   # Stop automation
  launchctl load {plist_path}     # Start automation
"""
        except Exception as e:
            return False, f"Failed to set up Mac automation: {e}"
    
    def setup_ubuntu_cron(self) -> tuple:
        """
        Set up Ubuntu cron job for scheduled backups.
        
        Returns:
            Tuple of (success, message)
        """
        # Log file location
        log_dir = os.path.expanduser("~/.vix_suite/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Cron expression for every N hours
        # 0 */6 * * * = every 6 hours at minute 0
        cron_line = f"0 */{self.interval_hours} * * * cd {self.project_dir} && {self.python_path} {self.backup_script} --auto >> {log_dir}/gdrive_backup.log 2>&1"
        
        marker_start = "# VIX 5W Suite Google Drive Backup - START"
        marker_end = "# VIX 5W Suite Google Drive Backup - END"
        
        try:
            # Get current crontab
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                current_crontab = result.stdout
            else:
                current_crontab = ""
            
            # Remove existing VIX backup entries
            lines = current_crontab.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if marker_start in line:
                    skip = True
                    continue
                if marker_end in line:
                    skip = False
                    continue
                if not skip and line.strip():
                    new_lines.append(line)
            
            # Add new entry
            new_lines.append("")
            new_lines.append(marker_start)
            new_lines.append(cron_line)
            new_lines.append(marker_end)
            new_lines.append("")
            
            new_crontab = '\n'.join(new_lines)
            
            # Install new crontab
            process = subprocess.Popen(
                ["crontab", "-"],
                stdin=subprocess.PIPE,
                text=True
            )
            process.communicate(input=new_crontab)
            
            if process.returncode != 0:
                return False, "Failed to install crontab"
            
            return True, f"""
✅ Ubuntu cron automation configured!

Cron Schedule: Every {self.interval_hours} hours (at minute 0)
Logs: {log_dir}/gdrive_backup.log

Commands:
  crontab -l                    # View current crontab
  crontab -e                    # Edit crontab manually

Entry added:
  {cron_line}
"""
        except Exception as e:
            return False, f"Failed to set up cron: {e}"
    
    def setup_ubuntu_systemd(self) -> tuple:
        """
        Set up Ubuntu systemd timer for scheduled backups.
        Alternative to cron for more robust scheduling.
        
        Returns:
            Tuple of (success, message)
        """
        user_systemd_dir = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(user_systemd_dir, exist_ok=True)
        
        # Log file location
        log_dir = os.path.expanduser("~/.vix_suite/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Service file
        service_content = f'''[Unit]
Description=VIX 5W Suite Google Drive Backup
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={self.project_dir}
ExecStart={self.python_path} {self.backup_script} --auto
StandardOutput=append:{log_dir}/gdrive_backup.log
StandardError=append:{log_dir}/gdrive_backup_error.log

[Install]
WantedBy=default.target
'''
        
        # Timer file
        timer_content = f'''[Unit]
Description=VIX 5W Suite Google Drive Backup Timer

[Timer]
OnBootSec=5min
OnUnitActiveSec={self.interval_hours}h
Persistent=true

[Install]
WantedBy=timers.target
'''
        
        service_path = os.path.join(user_systemd_dir, "vix5w-gdrive-backup.service")
        timer_path = os.path.join(user_systemd_dir, "vix5w-gdrive-backup.timer")
        
        try:
            # Write service file
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            # Write timer file
            with open(timer_path, 'w') as f:
                f.write(timer_content)
            
            # Reload systemd
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            
            # Enable and start timer
            subprocess.run(["systemctl", "--user", "enable", "vix5w-gdrive-backup.timer"], check=True)
            subprocess.run(["systemctl", "--user", "start", "vix5w-gdrive-backup.timer"], check=True)
            
            return True, f"""
✅ Ubuntu systemd timer automation configured!

Service: {service_path}
Timer: {timer_path}
Interval: Every {self.interval_hours} hours
Logs: {log_dir}/gdrive_backup.log

Commands:
  systemctl --user status vix5w-gdrive-backup.timer   # Check timer status
  systemctl --user list-timers                         # List all timers
  systemctl --user stop vix5w-gdrive-backup.timer     # Stop automation
  systemctl --user start vix5w-gdrive-backup.timer    # Start automation
  journalctl --user -u vix5w-gdrive-backup            # View logs
"""
        except Exception as e:
            return False, f"Failed to set up systemd timer: {e}"
    
    def create_wrapper_script(self) -> str:
        """Create a standalone wrapper script for manual/cron use"""
        wrapper_path = os.path.join(self.project_dir, "run_backup.sh")
        
        wrapper_content = f'''#!/bin/bash
# VIX 5W Suite Google Drive Backup Wrapper
# Generated: {datetime.now().isoformat()}

cd "{self.project_dir}"

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "vix_env/bin/activate" ]; then
    source vix_env/bin/activate
fi

# Run backup
{self.python_path} "{self.backup_script}" --auto

# Exit with backup script's exit code
exit $?
'''
        
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        os.chmod(wrapper_path, 0o755)
        
        return wrapper_path


def main():
    parser = argparse.ArgumentParser(
        description="Set up automated Google Drive backups for VIX 5W Suite"
    )
    parser.add_argument(
        '--platform',
        choices=['mac', 'ubuntu', 'auto'],
        default='auto',
        help="Target platform (default: auto-detect)"
    )
    parser.add_argument(
        '--systemd',
        action='store_true',
        help="Use systemd timer instead of cron (Ubuntu only)"
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=6,
        help="Backup interval in hours (default: 6)"
    )
    parser.add_argument(
        '--project-dir',
        type=str,
        default=None,
        help="Project directory path (default: current directory)"
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help="Remove automation instead of setting up"
    )
    
    args = parser.parse_args()
    
    # Auto-detect platform
    platform = args.platform
    if platform == 'auto':
        import platform as plat
        system = plat.system().lower()
        if system == 'darwin':
            platform = 'mac'
        elif system == 'linux':
            platform = 'ubuntu'
        else:
            print(f"❌ Unsupported platform: {system}")
            sys.exit(1)
    
    setup = AutomationSetup(
        project_dir=args.project_dir,
        interval_hours=args.interval
    )
    
    # Create wrapper script
    wrapper_path = setup.create_wrapper_script()
    print(f"📜 Created wrapper script: {wrapper_path}")
    
    if args.remove:
        print("\n⚠️  Remove automation manually:")
        if platform == 'mac':
            print(f"  launchctl unload ~/Library/LaunchAgents/com.vix5w.gdrive_backup.plist")
            print(f"  rm ~/Library/LaunchAgents/com.vix5w.gdrive_backup.plist")
        else:
            print("  crontab -e  # Remove VIX 5W Suite entries")
            print("  systemctl --user disable vix5w-gdrive-backup.timer")
        sys.exit(0)
    
    # Set up automation
    if platform == 'mac':
        success, message = setup.setup_mac_launchd()
    elif args.systemd:
        success, message = setup.setup_ubuntu_systemd()
    else:
        success, message = setup.setup_ubuntu_cron()
    
    print(message)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
