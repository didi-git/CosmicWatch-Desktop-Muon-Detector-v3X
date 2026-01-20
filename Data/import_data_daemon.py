#!/usr/bin/env python3
"""
CosmicWatch Data Logger Daemon
Designed to run as a systemd service for continuous data acquisition
"""

from __future__ import print_function
import serial
import time
import sys
import os
import signal
import logging
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/tty_cosmicwatch'
BAUDRATE = 115200
DATA_DIR = '/var/data'
LOG_DIR = '/var/log/cosmicwatch'
MAX_STARTUP_WAIT = 60  # seconds to wait for first valid data line

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'cosmicwatch.log')),
        logging.StreamHandler(sys.stdout)  # Also log to systemd journal
    ]
)
logger = logging.getLogger(__name__)


class CosmicWatchDaemon:
    def __init__(self):
        self.serial_port = None
        self.data_file = None
        self.device_name = None
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f'Received signal {signum}, shutting down...')
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Close serial port and file handles"""
        if self.data_file:
            try:
                self.data_file.flush()
                self.data_file.close()
                logger.info('Data file closed')
            except Exception as e:
                logger.error(f'Error closing data file: {e}')
        
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
                logger.info('Serial port closed')
            except Exception as e:
                logger.error(f'Error closing serial port: {e}')
    
    def open_serial_port(self):
        """Open the serial port"""
        logger.info(f'Opening serial port: {SERIAL_PORT}')
        try:
            self.serial_port = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
            time.sleep(2)  # Give the device time to initialize
            logger.info('Serial port opened successfully')
            return True
        except serial.SerialException as e:
            logger.error(f'Failed to open serial port: {e}')
            raise
    
    def get_device_name(self):
        """Read the first valid data line to extract device name"""
        logger.info('Waiting for device name from data stream...')
        start_time = time.time()
        
        while time.time() - start_time < MAX_STARTUP_WAIT:
            if not self.running:
                return None
            
            try:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Skip empty lines and comment lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try to parse the data line
                    data = line.split('\t')
                    
                    # Expected format has at least 11 columns, Name is at index 10
                    if len(data) >= 11:
                        device_name = data[10].strip()
                        if device_name:  # Make sure it's not empty
                            logger.info(f'Device name detected: {device_name}')
                            return device_name
                
                time.sleep(0.1)
            
            except Exception as e:
                logger.warning(f'Error reading device name: {e}')
                time.sleep(0.5)
        
        logger.error(f'Failed to detect device name within {MAX_STARTUP_WAIT} seconds')
        raise TimeoutError('Could not detect device name from data stream')
    
    def create_data_file(self):
        """Create a new data file with timestamp and device name"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'CW_{self.device_name}_{timestamp}.txt'
        filepath = os.path.join(DATA_DIR, filename)
        
        logger.info(f'Creating data file: {filepath}')
        
        try:
            self.data_file = open(filepath, 'w')
            
            # Write header
            self.data_file.write("###########################################################################################################################################################\n")
            self.data_file.write("#                                                          CosmicWatch: The Desktop Muon Detector v3X\n")
            self.data_file.write("#                                                                   Questions? saxani@udel.edu\n")
            self.data_file.write("# Event  Timestamp[s]  Flag  ADC[12b]  SiPM[mV]  Deadtime[s]  Temp[C]  Press[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date\n")
            self.data_file.write("###########################################################################################################################################################\n")
            self.data_file.flush()
            
            logger.info('Data file created successfully')
            return True
        
        except Exception as e:
            logger.error(f'Failed to create data file: {e}')
            raise
    
    def record_data(self):
        """Main data recording loop"""
        logger.info('Starting data acquisition...')
        event_counter = 0
        last_flush_time = time.time()
        flush_interval = 10.0  # Flush every 10 seconds to minimize I/O overhead
        
        while self.running:
            try:
                # Blocking read - returns immediately when line arrives
                # Timeout prevents indefinite blocking
                line = self.serial_port.readline()
                
                # Timestamp IMMEDIATELY after data arrives
                timestamp = datetime.now()
                
                # Skip empty lines (timeout returns empty bytes)
                if not line:
                    continue
                
                # Decode after timestamping to minimize latency
                line = line.decode('utf-8', errors='ignore').replace('\r\n', '')
                
                if not line:
                    continue
                
                # Parse and append timestamp
                data = line.split('\t')
                ti = str(timestamp).split(" ")
                comp_time = ti[-1]
                data.append(comp_time)
                
                comp_date = ti[0].split('-')
                data.append(comp_date[2] + '/' + comp_date[1] + '/' + comp_date[0])
                
                # Write to file (buffered, fast)
                self.data_file.write('\t'.join(data) + '\n')
                
                event_counter += 1
                
                # Periodic flush to disk (less frequent to reduce I/O overhead)
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval:
                    self.data_file.flush()
                    last_flush_time = current_time
                    
                    # Log progress every 100 events
                    if event_counter % 100 == 0:
                        logger.debug(f'Events recorded: {event_counter}')
            
            except serial.SerialException as e:
                logger.error(f'Serial port error: {e}')
                raise  # Let systemd restart the service
            
            except Exception as e:
                logger.error(f'Error recording data: {e}', exc_info=True)
                time.sleep(1)  # Brief pause before continuing
    
    def run(self):
        """Main entry point for the daemon"""
        try:
            logger.info('========================================================')
            logger.info('  CosmicWatch Data Logger Daemon Starting')
            logger.info(f'  Serial Port: {SERIAL_PORT}')
            logger.info(f'  Data Directory: {DATA_DIR}')
            logger.info('========================================================')
            
            # Open serial port
            self.open_serial_port()
            
            # Get device name from data stream
            self.device_name = self.get_device_name()
            if not self.device_name:
                logger.error('Failed to get device name, exiting')
                return 1
            
            # Create data file
            self.create_data_file()
            
            # Start recording
            self.record_data()
            
            logger.info('Data acquisition stopped normally')
            return 0
        
        except Exception as e:
            logger.error(f'Fatal error: {e}', exc_info=True)
            return 1
        
        finally:
            self.cleanup()


if __name__ == '__main__':
    daemon = CosmicWatchDaemon()
    sys.exit(daemon.run())
