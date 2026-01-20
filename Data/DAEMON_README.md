# CosmicWatch Systemd Daemon Setup

This directory contains files for running CosmicWatch as a systemd service for continuous, unattended data logging.

## Files

- **`import_data_daemon.py`** - Modified version of `import_data.py` for daemon operation
- **`cosmicwatch.service`** - Systemd service configuration file
- **`setup_daemon.sh`** - Automated setup script for installation
- **`DAEMON_README.md`** - This file

## Prerequisites

1. **Linux system with systemd** (tested on Ubuntu/Debian/RHEL/CentOS)
2. **uv package manager** installed (https://github.com/astral-sh/uv)
3. **Serial device** accessible at `/dev/tty_cosmicwatch`
4. **Root access** for installation

## Quick Start

### 1. Verify Serial Device

Make sure your CosmicWatch detector appears as `/dev/tty_cosmicwatch`. If not, create a udev rule:

Create `/etc/udev/rules.d/99-cosmicwatch.rules`:
```udev
# Replace SERIAL_NUMBER with your device's serial number
# Find it with: udevadm info -a -n /dev/ttyACM0 | grep serial
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", ATTRS{serial}=="YOUR_SERIAL_HERE", SYMLINK+="tty_cosmicwatch", MODE="0660", GROUP="dialout"
```

Or for any Raspberry Pi Pico device:
```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", SYMLINK+="tty_cosmicwatch", MODE="0660", GROUP="dialout"
```

Reload udev rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2. Run Setup Script

```bash
cd /path/to/cosmicwatch/Data
chmod +x setup_daemon.sh
sudo ./setup_daemon.sh
```

The setup script will:
- Create `cosmicwatch_logger` user
- Set up directories (`/var/data`, `/var/log/cosmicwatch`, `/opt/cosmicwatch`)
- Create Python virtual environment using `uv`
- Install dependencies (pyserial)
- Install systemd service

### 3. Enable and Start Service

```bash
# Enable service to start on boot
sudo systemctl enable cosmicwatch.service

# Start the service now
sudo systemctl start cosmicwatch.service

# Check status
sudo systemctl status cosmicwatch.service
```

## Operation

### How It Works

1. Service starts when the system boots (or when manually started)
2. Script waits for serial port `/dev/tty_cosmicwatch` to be available
3. Reads first data line to extract device name from column 11
4. Creates timestamped data file: `/var/data/CW_{DeviceName}_{timestamp}.txt`
5. Continuously logs data to the file
6. If service crashes or port disconnects, systemd automatically restarts it (creating a new file)

### Monitoring

**View live logs (systemd journal):**
```bash
sudo journalctl -u cosmicwatch.service -f
```

**View operational log file:**
```bash
tail -f /var/log/cosmicwatch/cosmicwatch.log
```

**Check data files:**
```bash
ls -lh /var/data/
```

**Monitor in real-time:**
```bash
tail -f /var/data/CW_*.txt
```

### Service Management

**Stop the service:**
```bash
sudo systemctl stop cosmicwatch.service
```

**Restart the service:**
```bash
sudo systemctl restart cosmicwatch.service
```

**Disable auto-start on boot:**
```bash
sudo systemctl disable cosmicwatch.service
```

**View recent logs:**
```bash
sudo journalctl -u cosmicwatch.service -n 100
```

## Updating the Daemon Script

If you modify `import_data_daemon.py` and need to update the running service:

### 1. Copy Updated Script

From your development directory (where you modified the script):
```bash
sudo cp import_data_daemon.py /opt/cosmicwatch/
sudo chown cosmicwatch_logger:cosmicwatch_logger /opt/cosmicwatch/import_data_daemon.py
sudo chmod +x /opt/cosmicwatch/import_data_daemon.py
```

### 2. Restart the Service

```bash
sudo systemctl restart cosmicwatch.service
```

### 3. Verify the Update

Check that the service started successfully:
```bash
sudo systemctl status cosmicwatch.service
```

View the logs to confirm new behavior:
```bash
sudo journalctl -u cosmicwatch.service -f
```

**Note:** Restarting the service will close the current data file and create a new one with a fresh timestamp.

## File Locations

| Item | Location |
|------|----------|
| Data files | `/var/data/CW_{DeviceName}_{timestamp}.txt` |
| Operational logs | `/var/log/cosmicwatch/cosmicwatch.log` |
| Python script | `/opt/cosmicwatch/import_data_daemon.py` |
| Virtual environment | `/opt/cosmicwatch/venv/` |
| Service file | `/etc/systemd/system/cosmicwatch.service` |
| System user | `cosmicwatch_logger` (no login shell) |

## Troubleshooting

### Service won't start

Check if the serial device exists:
```bash
ls -l /dev/tty_cosmicwatch
```

Check systemd status:
```bash
sudo systemctl status cosmicwatch.service
```

View detailed logs:
```bash
sudo journalctl -u cosmicwatch.service -n 50 --no-pager
```

### Permission issues

Verify user is in dialout group:
```bash
groups cosmicwatch_logger
```

Check directory permissions:
```bash
ls -ld /var/data /var/log/cosmicwatch
```

### Device not detected

List USB devices:
```bash
lsusb
```

List serial ports:
```bash
ls -l /dev/tty*
```

Check udev rules:
```bash
udevadm info -a -n /dev/ttyACM0 | grep -E "serial|idVendor|idProduct"
```

### Service keeps restarting

Check for errors in the log:
```bash
sudo journalctl -u cosmicwatch.service -e
cat /var/log/cosmicwatch/cosmicwatch.log
```

Common issues:
- Serial port permissions
- Device not connected
- Incorrect device path
- Python dependencies missing

## Advanced Configuration

### Change flush interval

Edit `/opt/cosmicwatch/import_data_daemon.py`:
```python
flush_interval = 1.0  # Change to desired seconds
```

### Change restart delay

Edit `/etc/systemd/system/cosmicwatch.service`:
```ini
RestartSec=5s  # Change to desired delay
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart cosmicwatch.service
```

### Manually manage virtual environment

```bash
# Activate environment
sudo -u cosmicwatch_logger -s
cd /opt/cosmicwatch
source venv/bin/activate

# Install additional packages
pip install some-package

# Deactivate
deactivate
exit
```

## Data File Format

Each restart creates a new file with format:
```
CW_{DeviceName}_{YYYY-MM-DD_HH-MM-SS}.txt
```

Examples:
- `CW_AxLab_2026-01-19_14-30-00.txt`
- `CW_AxLab_C_005_2026-01-19_15-45-22.txt`

Files include header and tab-separated data:
```
Event  Timestamp[s]  Flag  ADC[12b]  SiPM[mV]  Deadtime[s]  Temp[C]  Press[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date
```

## Uninstallation

```bash
# Stop and disable service
sudo systemctl stop cosmicwatch.service
sudo systemctl disable cosmicwatch.service

# Remove service file
sudo rm /etc/systemd/system/cosmicwatch.service
sudo systemctl daemon-reload

# Remove files (backup data first!)
sudo rm -rf /opt/cosmicwatch
# Optional: remove data and logs
# sudo rm -rf /var/data/CW_*.txt
# sudo rm -rf /var/log/cosmicwatch

# Remove user
sudo userdel cosmicwatch_logger
```

## Notes

- Each service restart creates a new data file (by design)
- Old data files are never deleted automatically
- Log rotation is not configured by default (consider adding logrotate config)
- The service will wait up to 60 seconds for device name detection
- Virtual environment allows for isolated Python dependencies
- systemd handles automatic restarts on failure or device reconnection
