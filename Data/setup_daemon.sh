#!/bin/bash
# Setup script for CosmicWatch systemd daemon
# Run this script with sudo on the target machine

set -e

echo "=========================================="
echo "CosmicWatch Daemon Setup Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Check if uv is installed
if ! which uv &> /dev/null; then
    echo "Error: uv package manager not found"
    echo "Install it from: https://github.com/astral-sh/uv"
    exit 1
fi

echo "Step 1: Creating system user 'cosmicwatch_logger'"
if id "cosmicwatch_logger" &>/dev/null; then
    echo "  User already exists, skipping..."
else
    useradd -r -s /usr/sbin/nologin -c "CosmicWatch Data Logger" cosmicwatch_logger
    echo "  User created"
fi

echo ""
echo "Step 2: Adding user to dialout group for serial port access"
usermod -a -G dialout cosmicwatch_logger
echo "  Done"

echo ""
echo "Step 3: Creating directories"
mkdir -p /opt/cosmicwatch
mkdir -p /var/data
mkdir -p /var/log/cosmicwatch
mkdir -p /home/cosmicwatch_logger
mkdir -p /home/cosmicwatch_logger/.cache

echo "  Setting ownership..."
chown cosmicwatch_logger:cosmicwatch_logger /opt/cosmicwatch
chown cosmicwatch_logger:cosmicwatch_logger /var/data
chown cosmicwatch_logger:cosmicwatch_logger /var/log/cosmicwatch
chown cosmicwatch_logger:cosmicwatch_logger /home/cosmicwatch_logger
chown cosmicwatch_logger:cosmicwatch_logger /home/cosmicwatch_logger/.cache
echo "  Done"

echo ""
echo "Step 4: Creating Python virtual environment with uv"
cd /opt/cosmicwatch
sudo -u cosmicwatch_logger uv venv venv
echo "  Virtual environment created"

echo ""
echo "Step 5: Installing Python dependencies"
sudo -u cosmicwatch_logger /opt/cosmicwatch/venv/bin/pip install pyserial
echo "  Dependencies installed"

echo ""
echo "Step 6: Copying daemon script"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/import_data_daemon.py" /opt/cosmicwatch/
chmod +x /opt/cosmicwatch/import_data_daemon.py
chown cosmicwatch_logger:cosmicwatch_logger /opt/cosmicwatch/import_data_daemon.py
echo "  Script copied to /opt/cosmicwatch/"

echo ""
echo "Step 7: Installing systemd service"
cp "$SCRIPT_DIR/cosmicwatch.service" /etc/systemd/system/
chmod 644 /etc/systemd/system/cosmicwatch.service
echo "  Service file installed"

echo ""
echo "Step 8: Reloading systemd"
systemctl daemon-reload
echo "  Done"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Important: Verify that /dev/tty_cosmicwatch exists and points to your device"
echo ""
echo "Next steps:"
echo "  1. Check device symlink:"
echo "     ls -l /dev/tty_cosmicwatch"
echo ""
echo "  2. Enable service to start on boot:"
echo "     sudo systemctl enable cosmicwatch.service"
echo ""
echo "  3. Start the service:"
echo "     sudo systemctl start cosmicwatch.service"
echo ""
echo "  4. Check service status:"
echo "     sudo systemctl status cosmicwatch.service"
echo ""
echo "  5. View logs:"
echo "     sudo journalctl -u cosmicwatch.service -f"
echo "     tail -f /var/log/cosmicwatch/cosmicwatch.log"
echo ""
echo "  6. Monitor data files:"
echo "     ls -lh /var/data/"
echo ""
