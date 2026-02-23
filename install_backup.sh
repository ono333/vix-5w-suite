#!/bin/bash
# VIX 5W Suite - Google Drive Backup Installation Script
# Usage: bash install_backup.sh

set -e

echo "=============================================="
echo "VIX 5W Suite - Google Drive Backup Installer"
echo "=============================================="
echo ""

# Detect platform
PLATFORM="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="mac"
    echo "🍎 Detected: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="ubuntu"
    echo "🐧 Detected: Linux/Ubuntu"
else
    echo "⚠️  Unknown platform: $OSTYPE"
fi
echo ""

# Check Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Please install Python 3.8+"
    exit 1
fi

echo "✅ Using: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install dependencies
echo "📦 Installing Google API dependencies..."
$PYTHON_CMD -m pip install --quiet google-auth google-auth-oauthlib google-api-python-client

echo "✅ Dependencies installed"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p ~/.vix_suite/backups
mkdir -p ~/.vix_suite/logs

echo "✅ Directories created:"
echo "   ~/.vix_suite/"
echo "   ~/.vix_suite/backups/"
echo "   ~/.vix_suite/logs/"
echo ""

# Check for credentials
CREDS_PATH="$HOME/.vix_suite/gdrive_credentials.json"
if [ -f "$CREDS_PATH" ]; then
    echo "✅ Credentials found at: $CREDS_PATH"
else
    echo "⚠️  No credentials found at: $CREDS_PATH"
    echo ""
    echo "To set up credentials:"
    echo "1. Go to: https://console.cloud.google.com/"
    echo "2. Create project & enable Google Drive API"
    echo "3. Create Service Account → Download JSON key"
    echo "4. Save as: ~/.vix_suite/gdrive_credentials.json"
fi
echo ""

# Test import
echo "🧪 Testing backup module..."
cd "$SCRIPT_DIR"
$PYTHON_CMD -c "from gdrive_backup import GDriveBackupManager; print('✅ Backup module OK')" 2>/dev/null || {
    echo "❌ Failed to import backup module"
    exit 1
}

# Run setup if credentials exist
if [ -f "$CREDS_PATH" ]; then
    echo ""
    echo "Running setup wizard..."
    $PYTHON_CMD gdrive_backup.py --setup
fi

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
if [ ! -f "$CREDS_PATH" ]; then
    echo "1. Set up Google Cloud credentials (see above)"
    echo "2. Run: python gdrive_backup.py --setup"
else
    echo "1. Test backup: python gdrive_backup.py --backup"
    echo "2. Check status: python gdrive_backup.py --status"
fi
echo ""
echo "To set up automated backups:"
if [ "$PLATFORM" == "mac" ]; then
    echo "   python setup_automation.py --platform mac"
else
    echo "   python setup_automation.py --platform ubuntu"
fi
echo ""
echo "To integrate with Streamlit app:"
echo "   See app_integration.py for code examples"
echo ""
