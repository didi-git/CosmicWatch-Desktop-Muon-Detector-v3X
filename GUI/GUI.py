#imports
import sys
#from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QGridLayout, QTextEdit,
    QFileDialog, QComboBox, QRadioButton, QToolButton, QLineEdit
)
from PyQt6.QtCore import (
    Qt, QSize, pyqtSignal
)
from PyQt6.QtGui import (
    QPixmap, QIcon
)
import sys
import time
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import serial.tools.list_ports
import glob
import serial 
import glob
import os
import os.path
import signal
from datetime import datetime
from multiprocessing import Process
import math
import random
import platform
import threading
import math
import pyqtgraph.opengl as gl
from stl import mesh
import sys, os, time, warnings, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="libpng")

input_images = ['refresh_white.png', 'upload_white.png', 'save_white.png', 'play_white.png', 'pause_white.png','logo_white.png']
for input_image in input_images:
    out_image = input_image.replace('white','black')
    if os.path.exists(out_image):
        continue
    # Open the image (preserves alpha channel)
    img = Image.open(input_image).convert("RGBA")
    # Split into RGB and alpha
    r, g, b, a = img.split()
    # Invert only the RGB channels
    rgb_image = Image.merge("RGB", (r, g, b))
    inverted_rgb = ImageOps.invert(rgb_image)
    # Combine back with alpha
    inverted_img = Image.merge("RGBA", (*inverted_rgb.split(), a))
    # Save
    inverted_img.save(input_image.replace('white','black'))

class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None, get_filename_func=None, plot_type="plot"):
        super().__init__(canvas, parent)
        self.get_filename_func = get_filename_func
        self.plot_type = plot_type

    def save_figure(self, *args):
        # --- 1. Get input file if available ---
        try:
            input_file = self.get_filename_func() if self.get_filename_func else "plot"
        except Exception:
            input_file = "plot"

        base = os.path.splitext(os.path.basename(input_file))[0]

        # --- 2. Determine save directory ---
        if input_file and input_file != "plot":
            base_dir = os.path.dirname(input_file)
        else:
            base_dir = os.getcwd()

        figures_dir = os.path.join(base_dir, "Figures")
        os.makedirs(figures_dir, exist_ok=True)   # ✅ auto-create if missing

        suggested = os.path.join(figures_dir, f"{base}_{self.plot_type}.pdf")

        # --- 3. Open save dialog ---
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", suggested, 
            "PDF (*.pdf);;PNG (*.png);;All Files (*)"
        )

        if path:
            self.canvas.figure.savefig(path, format=os.path.splitext(path)[1][1:])

font = {
    "family": "serif",
    "serif": "Computer Modern Roman",
    "weight": 200,
    "size": 15
}
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Define your own color palette
mycolors = ['#c70039','#ff5733','#ff8d1a','#ffc300','#eddd53','#add45c','#57c785',
               '#00baad','#2a7b9b','#3d3d6b','#511849','#900c3f','#900c3f']

mpl.rcParams['savefig.format'] = 'pdf'


class CWClass():
    def __init__(self,file_name, bin_size, feed_box):
        self.bin_size = bin_size
        self.feed_box = feed_box
        self.name = file_name.split('/')[-1]
        fileHandle = open(file_name,"r" )
        lineList = fileHandle.readlines()
        fileHandle.close()
        header_lines = 0
        last_line_of_header = 0
        for i in range(min(len(lineList),1000)):
            if "#" in lineList[i]:
                last_line_of_header = i+1
                #Determine number of columns by looking at the second last line in the file.
        # Strip whitespace and filter empty strings to handle trailing tabs
        number_of_columns = len([col for col in lineList[len(lineList)-2].strip().split("\t") if col])
        column_array = range(0,number_of_columns)


        file_from_computer = False
        file_from_sdcard   = False
        if number_of_columns == 13:
            file_from_computer = True
            self.feed_box.append('File from Computer')
            data = np.genfromtxt(file_name, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            event_number = data[:,0].astype(float) #first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            coincident = data[:,2].astype(int).astype(bool)  # Convert to int first, then to bool
            adc = data[:,3].astype(int)
            sipm = data[:,4].astype(float)
            deadtime = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            pressure = data[:,7].astype(float)
            accelerometer = data[:,8].astype(str)
            accel_x = []
            accel_y = []
            accel_z = []
            Gyro = data [:,9].astype(str)
            gyro_x = []
            gyro_y = []
            gyro_z = []
            # Parse accelerometer data
            for i in range(len(accelerometer)):
                accel = accelerometer[i].split(':')
                accel_x.append(accel[0])
                accel_y.append(accel[1])
                accel_z.append(accel[2])
            accel_x = np.asarray(accel_x).astype(float)
            accel_y = np.asarray(accel_y).astype(float)
            accel_z = np.asarray(accel_z).astype(float)
            # Parse gyro data
            for i in range(len(Gyro)):
                gyro = Gyro[i].split(':')
                gyro_x.append(gyro[0])
                gyro_y.append(gyro[1])
                gyro_z.append(gyro[2])
            gyro_x = np.asarray(gyro_x).astype(float)
            gyro_y = np.asarray(gyro_y).astype(float)
            gyro_z = np.asarray(gyro_z).astype(float)
            detName = data[:,10]
            comp_time = data[:,11]
            comp_date = data[:,12]
        
        elif number_of_columns == 10:
            file_from_sdcard = True 
            self.feed_box.append('File from MicroSD Card')
            data = np.genfromtxt(file_name, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            event_number = data[:,0].astype(float)#first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            coincident = data[:,2].astype(int).astype(bool)  # Convert to int first, then to bool
            adc = data[:,3].astype(int)
            sipm = data[:,4].astype(float)
            deadtime = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            pressure = data[:,7].astype(float)
            accelerometer = data[:,8].astype(str)
            accel_x = []
            accel_y = []
            accel_z = []
            Gyro = data [:,9].astype(str)
            gyro_x = []
            gyro_y = []
            gyro_z = []
            for i in range(len(accelerometer)):
                accel = accelerometer[i].split(':')
                accel_x.append(accel[0])
                accel_y.append(accel[1])
                accel_z.append(accel[2])
            accel_x = np.asarray(accel_x).astype(float)
            accel_y = np.asarray(accel_y).astype(float)
            accel_z = np.asarray(accel_z).astype(float)
            for i in range(len(Gyro)):
                gyro = Gyro[i].split(':')
                gyro_x.append(gyro[0])
                gyro_y.append(gyro[1])
                gyro_z.append(gyro[2])
            gyro_x = np.asarray(gyro_x).astype(float)
            gyro_y = np.asarray(gyro_y).astype(float)
            gyro_z = np.asarray(gyro_z).astype(float)
        else: 
            error_msg = f"Incorrect number of columns in file: {number_of_columns}. Expected 10 or 13 columns."
            self.feed_box.append(error_msg)
            raise ValueError(error_msg)
            
        if file_from_computer:
            time_stamp = []
            for i in range(len(comp_date)):
                day  = int(comp_date[i].split('/')[0])
                month = int(comp_date[i].split('/')[1])
                year   = int(comp_date[i].split('/')[2])
                hour  = int(comp_time[i].split(':')[0])
                mins  = int(comp_time[i].split(':')[1])
                sec   = int(np.floor(float(comp_time[i].split(':')[2])))
                try:  
                    decimal = float('0.'+str(comp_time[i].split('.')[-1]))
                except:
                    decimal = 0.0
                time_stamp.append(float(time.mktime((year, month, day, hour, mins, sec, 0, 0, 0))) + decimal) 
            self.time_stamp_s     = np.asarray(time_stamp) -  min(np.asarray(time_stamp))       # The absolute time of an event in seconds
            self.time_stamp_ms    = np.asarray(time_stamp -  min(np.asarray(time_stamp)))*1000  # The absolute time of an event in miliseconds   
            self.total_time_s     = max(time_stamp) -  min(time_stamp)     # The absolute time of an event in seconds
            self.detector_name    = detName                                
            self.n_detector       = len(set(detName))
        event_deadtime_s = np.diff(np.append([0],deadtime))
        self.PICO_timestamp_s       = PICO_timestamp_s
        self.PICO_total_time_s = max(self.PICO_timestamp_s) - min(self.PICO_timestamp_s)
        self.PICO_total_time_ms= self.PICO_total_time_s * 1000.
        self.event_number     = np.asarray(event_number)  # an arrray of the event numbers
        self.total_counts     = max(event_number.astype(int)) - min(event_number.astype(int))
        self.select_coincident        = coincident         # an arrray of the measured event ADC value

        self.adc              = adc         # an arrray of the measured event ADC value
        self.sipm             = sipm        # an arrray of the measured event SiPM value
        
        self.temperature      = temperature         # an arrray of the measured event ADC value
        self.pressure        = pressure         # an arrray of the measured event ADC value

        self.accel_x        = accel_x         # an arrray event acceleration x data
        self.accel_y        = accel_y        # an arrray event acceleration x data
        self.accel_z        = accel_z        # an arrray event acceleration x data

        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        

        self.event_deadtime_s   = event_deadtime_s     # an array of the measured event deadtime in seconds
        self.event_deadtime_ms  = event_deadtime_s*1000            # an array of the measured event deadtime in miliseconds
        self.total_deadtime_s   = max(deadtime) - min(deadtime)       # an array of the measured event deadtime in miliseconds
        self.total_deadtime_ms  = self.total_deadtime_s*1000. # The total deadtime in seconds
        self.PICO_event_livetime_s = np.diff(np.append([0],self.PICO_timestamp_s)) - self.event_deadtime_s
        def round(x, err):
            """Round x and err based on the first significant digit of err."""
            if err == 0:
                return x, err  # Avoid division by zero
            # Find order of magnitude of error
            order_of_magnitude = int(np.floor(np.log10(err)))
            # Find the first significant digit of err
            first_digit = int(err / (10 ** order_of_magnitude))
            # Round both values to the first significant digit of err
            rounded_x = np.round(x, -order_of_magnitude+1)
            rounded_err = np.round(err, -order_of_magnitude+1)#first_digit * (10 ** (order_of_magnitude)) 
            return rounded_x, rounded_err

        if file_from_computer:
            self.live_time        = (self.total_time_s - self.total_deadtime_s)
            self.live_time_s      = self.live_time  # Alias for consistency with sdcard files
            self.weights          = np.ones(len(event_number)) / self.live_time
            self.count_rate       = self.total_counts/self.live_time 
            self.count_rate_err   = np.sqrt(self.total_counts)/self.live_time 

            bins = range(int(min(self.time_stamp_s)), int(max(self.time_stamp_s)), self.bin_size)
            counts, binEdges       = np.histogram(self.time_stamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.time_stamp_s, bins = bins, weights = self.PICO_event_livetime_s)
        
            # Bin the pressure by taking the average pressure in each bin
            sum_pressure, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_x, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.accel_x)
            count_accel_x, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_accel_x = sum_accel_x / np.maximum(count_accel_x, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_y, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.accel_y)
            count_accel_y, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_accel_y = sum_accel_y / np.maximum(count_accel_y, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_z, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.accel_z)
            count_accel_z, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_accel_z = sum_accel_z / np.maximum(count_accel_z, 1)  # Avoid division by Bin the temperature by taking the average temperature in each bin

            sum_gyro_x, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.gyro_x)
            count_gyro_x, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_gyro_x = sum_gyro_x / np.maximum(count_gyro_x, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_gyro_y, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.gyro_y)
            count_gyro_y, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_gyro_y = sum_gyro_y / np.maximum(count_gyro_y, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_gyro_z, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.gyro_z)
            count_gyro_z, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_gyro_z = sum_gyro_z / np.maximum(count_gyro_z, 1)  # Avoid division by
            
            # Calculate binned count rates
            self.binned_counts = counts
            self.binned_counts_err = np.sqrt(counts)
            self.binned_count_rate = counts / bin_livetime
            self.binned_count_rate_err = np.sqrt(counts) / bin_livetime
            
            # Calculate binned coincident count rates
            counts_coincident, _ = np.histogram(self.time_stamp_s[self.select_coincident], bins=bins)
            bin_livetime_coincident, _ = np.histogram(self.time_stamp_s[self.select_coincident], bins=bins, weights=self.PICO_event_livetime_s[self.select_coincident])
            self.total_coincident = len(self.time_stamp_s[self.select_coincident])
            self.binned_counts_coincident = counts_coincident
            self.binned_counts_err_coincident = np.sqrt(counts_coincident)
            self.binned_count_rate_coincident = counts_coincident / np.maximum(bin_livetime, 1)
            self.binned_count_rate_err_coincident = np.sqrt(counts_coincident) / np.maximum(bin_livetime, 1)
            
            # Calculate coincident rate statistics
            self.count_rate_coincident = self.total_coincident / self.live_time
            self.count_rate_err_coincident = np.sqrt(self.total_coincident) / self.live_time
            
            # Calculate binned non-coincident count rates
            counts_non_coincident, _ = np.histogram(self.time_stamp_s[~self.select_coincident], bins=bins)
            self.total_non_coincident = len(self.time_stamp_s[~self.select_coincident])
            self.binned_counts_non_coincident = counts_non_coincident
            self.binned_counts_err_non_coincident = np.sqrt(counts_non_coincident)
            self.binned_count_rate_non_coincident = counts_non_coincident / np.maximum(bin_livetime, 1)
            self.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident) / np.maximum(bin_livetime, 1)
            
            # Calculate non-coincident rate statistics
            self.count_rate_non_coincident = self.total_non_coincident / self.live_time
            self.count_rate_err_non_coincident = np.sqrt(self.total_non_coincident) / self.live_time
            
            # Calculate binned deadtime percentage
            bin_deadtime, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.event_deadtime_s)
            self.binned_deadtime_percentage = bin_deadtime / self.bin_size * 100
        
        elif file_from_sdcard:
            self.live_time_s        = (self.PICO_total_time_s - self.total_deadtime_s)
            self.live_time_ms        = (self.PICO_total_time_ms - self.total_deadtime_ms)/1000.

            self.weights          = np.ones(len(event_number)) / self.live_time_s

            n = 4
            self.feed_box.append(
                f"    -- Total Count Rate: {np.round(self.total_counts/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_counts)/self.live_time_s, n)} Hz"
            )

            self.count_rate, self.count_rate_err = round(
                    self.total_counts/self.live_time_s, 
                    np.sqrt(self.total_counts)/self.live_time_s)
            

            bins = range(int(min(self.PICO_timestamp_s)),int(max(self.PICO_timestamp_s)),self.bin_size)
            counts, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.PICO_event_livetime_s)

            self.bin_size          = bin_size
            self.binned_counts     = counts
            self.binned_counts_err = np.sqrt(counts)
            self.binned_count_rate = counts/bin_livetime
            self.binned_count_rate_err = np.sqrt(counts)/bin_livetime

            counts_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[self.select_coincident], bins = bins)
            bin_deadtime, binEdges      = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.event_deadtime_s)

            self.total_coincident = len(self.PICO_timestamp_s[self.select_coincident])
            
            self.feed_box.append(
                f"    -- Count Rate Coincident (coincident): {np.round(self.total_coincident/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_coincident)/self.live_time_s, n)} Hz"
            )

            self.count_rate_coincident, self.count_rate_err_coincident = round(
                    self.total_coincident/self.live_time_s, 
                    np.sqrt(self.total_coincident)/self.live_time_s)
            
            
            
            
            self.binned_counts_coincident     = counts_coincident
            self.binned_counts_err_coincident = np.sqrt(counts_coincident)
            self.binned_count_rate_coincident = counts_coincident/(bin_size-bin_deadtime)
            self.binned_count_rate_err_coincident = np.sqrt(counts_coincident)/(bin_size-bin_deadtime)



            counts_non_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[~self.select_coincident], bins = bins)
            bin_deadtime, binEdges      = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.event_deadtime_s)
            self.total_non_coincident = len(self.PICO_timestamp_s[~self.select_coincident])
            self.binned_counts_non_coincident     = counts_non_coincident
            self.binned_counts_err_non_coincident = np.sqrt(counts_non_coincident)
            self.binned_count_rate_non_coincident = counts_non_coincident/(bin_size-bin_deadtime)
            self.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident)/(bin_size-bin_deadtime)
            self.feed_box.append(
                f"    -- Count Rate Non-Coincident: {np.round(self.total_non_coincident/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_non_coincident)/self.live_time_s, n)} Hz"
            )

            self.count_rate_non_coincident, self.count_rate_err_non_coincident = round(
                    self.total_non_coincident/self.live_time_s, 
                    np.sqrt(self.total_non_coincident)/self.live_time_s)

            # Calculate binned deadtime percentage
            self.binned_deadtime_percentage = bin_deadtime / self.bin_size * 100

            sum_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.accel_x)
            count_accel_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_accel_x = sum_accel_x / np.maximum(count_accel_x, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_y, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.accel_y)
            count_accel_y, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_accel_y = sum_accel_y / np.maximum(count_accel_y, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_z, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.accel_z)
            count_accel_z, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_accel_z = sum_accel_z / np.maximum(count_accel_z, 1)  # Avoid division by zero

            # Bin gyro_x by taking the average in each bin
            sum_gyro_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.gyro_x)
            count_gyro_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_gyro_x = sum_gyro_x / np.maximum(count_gyro_x, 1)  # Avoid division by zero

            # Bin gyro_y by taking the average in each bin
            sum_gyro_y, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.gyro_y)
            count_gyro_y, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_gyro_y = sum_gyro_y / np.maximum(count_gyro_y, 1)

            # Bin gyro_z by taking the average in each bin
            sum_gyro_z, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.gyro_z)
            count_gyro_z, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_gyro_z = sum_gyro_z / np.maximum(count_gyro_z, 1)

            # Coincident binned data
        bincenters = 0.5*(binEdges[1:]+ binEdges[:-1])
        self.binned_time_s     = bincenters
        self.binned_time_m     = bincenters/60.
        self.weights           = np.ones(len(event_number)) / self.live_time_s  


        
def serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
        sys.exit(0)
    result = []
    for port in ports:
        try: 
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def plusSTD(n,array):
    xh = np.add(n,np.sqrt(np.abs(array)))
    return xh

def subSTD(n,array):
    xl = np.subtract(n,np.sqrt(np.abs(array)))
    return xl
def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, lw=2, **kwargs):
    # If no Axes object given, grab the current one:
    if ax is None:
        ax = plt.gca()
    
    # First, duplicate the x values
    xx = np.ravel(np.column_stack((x, x)))[1:]
    
    # Now: calculate the average x bin width
    xstep = np.ravel(np.column_stack((x[1:] - x[:-1], x[1:] - x[:-1])))
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    
    # Now: add one step at the end of the row
    xx = np.append(xx, xx.max() + xstep[-1])

    # Adjust step alignment
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = np.ravel(np.column_stack((y1, y1)))
    if isinstance(y2, np.ndarray):
        y2 = np.ravel(np.column_stack((y2, y2)))

    # Plotting
    ax.fill_between(xx, y1, y2=y2, lw=lw, **kwargs)
    return ax







#GUI Display
class FuturisticDashboard(QWidget):
    data_ready = pyqtSignal(str)
    data_ready2 = pyqtSignal(str)
    data_ready3 = pyqtSignal(str)
    data_ready4 = pyqtSignal(str)
        
    def handle_port_selected(self, selected_port):
        time.sleep(0.1)
        if selected_port in ("Select Port", "No ports found", ""):
            return
        baudrate = 115200
        self.serial_connection = serial.Serial(selected_port, baudrate)
        time.sleep(0.1)
        if self.serial_connection.is_open:
            self.feed_box.append(f"Connected to {selected_port}.")
        else:
            self.feed_box.append(f"Failed to connect to {selected_port}.")
        
        return
    def refresh_ports(self):
        if hasattr(self, 'serial_connection') and self.serial_connection and self.serial_connection.is_open:
            self.feed_box.append(f"Disconnected from serial port {self.serial_connection.port}.")
            self.serial_connection.close()
        self.port_dropdown.clear()
        self.port_dropdown.addItem("Select Port") 
        time.sleep(0.1)
        t1 = time.time()
        ports = serial_ports()
        if (time.time()-t1)>6:
            self.feed_box.append('Listing ports is taking unusually long...')
        for port in ports:
            self.port_dropdown.addItem(port)
        self.feed_box.append(f"Serial port refreshed.")
        if not ports:
            self.port_dropdown.addItem("No ports found")
        return
        
    
    #creating window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CosmicWatch Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #0e1a2b; color: white;")
        self.init_ui()
        self.apply_theme()
        self.read_serial_active = False
        self.data_ready.connect(self.feed_box.append)
        self.data_ready2.connect(self.avg_stat1.setText)
        # self.data_ready3.connect(self.avg_stat_2.setText)
        # self.data_ready4.connect(self.avg_stat_3.setText)

        
        

    #creating layout/funcitons
    def init_ui(self):
        self.detector_name = "AxLab"
        self.time_stamp = "   "
        self.rate = "Unknown"
        self.rate_error = " "
        self.coincidence_rate = "Unknown"
        self.coincidence_rate_error = " "
        self.binning_selected = False

        self.themes = {
            "dark": {
                "bg": "#0e1a2b",
                "fg": "white",
                "accent": "#0e1a2b",
                "panel": "#1c2b3a",
                "button_background": "#122c3d",   # ← dark button
                "button_hover": "#0e1a2b",  # dark button hover
                "button_border": "white",      # ← light butto
                "button_text": "white"  # dark button hover
            },
            "light": {
                "bg": "white",
                "fg": "#202020",
                "accent": "#0066cc",
                "panel": "#f0f0f0",
                "button_background": "#122c3d",      # ← light button
                "button_border": "black",      # ← light butto
                "button_hover": "#0e1a2b",  # dark button hover
                "button_text": "white"  # dark button hover
            }
        }
        self.current_theme = "dark"
        
        #setting side lengths of right and left panel
        main_layout = QGridLayout()
        main_layout.setColumnStretch(0, 10)  # Left side = 75%
        main_layout.setColumnStretch(1, 3)  # Right side = 25%
        
        #left Panel/top buttons
        left_panel = QVBoxLayout()
        control_btns = QHBoxLayout()
        button_style = """
            QPushButton, QToolButton {
                background-color: #122c3d;
                color: #00ffcc;              /* keep cyan text/icons */
                border: 1px solid white;     /* <-- white border */
                border-radius: 8px;
                padding: 6px;
            }

            QPushButton:hover, QToolButton:hover {
                background-color: #0e1a2b;
                border: 1px solid white;     /* <-- white border */
                color: #00ffff;
            }

            QPushButton:pressed, QToolButton:pressed {
                background-color: #0b1a29;
                border: 1px solid white;     /* <-- white border */
                color: #00cccc;
            }
            """
        

        #load button
        load_btn = QToolButton()
        load_btn.setIcon(QIcon("upload_white.png"))
        load_btn.setIconSize(QSize(20, 20))
        load_btn.setStyleSheet(button_style)
        load_btn.clicked.connect(self.load_file)
        load_btn.setToolTip("Load Data File")
        control_btns.addWidget(load_btn)
        #save
        save_btn = QToolButton()
        save_btn.setIcon(QIcon("save_white.png"))
        save_btn.setIconSize(QSize(20, 20))
        save_btn.setStyleSheet(button_style)
        save_btn.setToolTip("Save Current Plot")
        save_btn.clicked.connect(self.save_plot)
        control_btns.addWidget(save_btn)
        left_panel.addLayout(control_btns)
        # STOP button
        stop_btn = QToolButton()
        stop_btn.setIcon(QIcon("pause_white.png"))
        stop_btn.setIconSize(QSize(20, 20))
        stop_btn.setStyleSheet(button_style)
        stop_btn.clicked.connect(self.stop_file)
        stop_btn.setToolTip("Stop Data Acquisition")
        control_btns.addWidget(stop_btn)
        #record
        record_btn = QToolButton()
        record_btn.setIcon(QIcon("play_white.png"))
        record_btn.setIconSize(QSize(20, 20))
        record_btn.setStyleSheet(button_style)
        record_btn.clicked.connect(self.record_file)
        record_btn.setToolTip("Start Data Acquisition")
        control_btns.addWidget(record_btn)
        
        #portselection
        self.port_dropdown = QComboBox()
        self.port_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #122c3d;
                color: #eee;
                border: 1px solid white;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #0e1a2b;
                color: #00ffcc;
                selection-background-color: #00ffcc;
            }
        """)
                

        #bin time input
        binning_layout = QHBoxLayout()

        self.binning_label = QLabel("Rate Time Interval:")
        self.binning_label.setStyleSheet("""
            font-family: 'Times New Roman', Times, serif;
            color: #eee;
            padding-right: 5px;
            font-size: 18px;
            font-weight: bold;
        """)
        binning_layout.addWidget(self.binning_label)
        
        # Set default bin time
        self.selected_bin_time = 30
        self.binning_selected = True
        
        # Add custom bin time input
        self.custom_bin_input = QLineEdit()
        self.custom_bin_input.setText("30")  # Set default value
        self.custom_bin_input.setFixedWidth(80)
        self.custom_bin_input.setStyleSheet("""
            QLineEdit {
                background-color: #122c3d;
                color: #eee;
                border: 1px solid white;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 2px solid #00ffcc;
            }
        """)
        self.custom_bin_input.returnPressed.connect(self.select_custom_bin)
        binning_layout.addWidget(self.custom_bin_input)
        
        custom_unit_label = QLabel("s")
        custom_unit_label.setStyleSheet("""
            color: #eee;
            font-size: 14px;
            padding-left: 3px;
        """)
        binning_layout.addWidget(custom_unit_label)
        
        # Add stretch to push everything to the left
        binning_layout.addStretch()
        
        left_panel.addLayout(binning_layout)

        # graph
        self.static_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.static_canvas.setFixedHeight(450)
        self.static_canvas.setStyleSheet("border: 1px solid #1f2f46; padding: 10px;")


        left_panel.addWidget(self.static_canvas)
     
        # Toolbar removed to maximize plotting area
        # Create a minimal toolbar object for compatibility with existing code
        class DummyToolbar:
            plot_type = "plot"
        self.toolbar = DummyToolbar()

        # --- Add custom coordinate display ---
        self.coord_label = QLabel("x: -, y: -")
        self.coord_label.setStyleSheet("color: white;")   # will adapt to theme later
        left_panel.addWidget(self.coord_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # --- Connect mpl event for mouse motion ---
        def update_coords(event):
            if event.inaxes:  # inside the plot
                x, y = event.xdata, event.ydata
                self.coord_label.setText(f"x: {x:.2f}, y: {y:.2f}")
            else:  # outside axes
                self.coord_label.setText("x: -, y: -")

        self.static_canvas.mpl_connect("motion_notify_event", update_coords)

        
        self.static_ax = self.static_canvas.figure.subplots()
        
        for spine in self.static_ax.spines.values():
            spine.set_edgecolor("white")

        self.static_ax.set_facecolor("#0e1a2b")  # dark dashboard color
        self.static_canvas.figure.set_facecolor("#0e1a2b")  # whole figure background
        # 4️⃣ Make all text white
        self.static_ax.title.set_color('white')
        self.static_ax.xaxis.label.set_color('white')
        self.static_ax.yaxis.label.set_color('white')
        self.static_ax.tick_params(axis='x', colors='white')
        self.static_ax.tick_params(axis='y', colors='white')
        # 5️⃣ Add grid with white lines
        self.static_ax.grid(True, color='white', linestyle='--', alpha=0.3)

        #Bottom buttons
        scan_btns = QHBoxLayout()

        # Rate button
        self.rate_btn = QPushButton("Rate")
        self.rate_btn.clicked.connect(self.run_rate)
        scan_btns.addWidget(self.rate_btn)

        # Deadtime button
        self.deadtime_btn = QPushButton("Deadtime")
        self.deadtime_btn.clicked.connect(self.run_deadtime)
        scan_btns.addWidget(self.deadtime_btn)

        # ADC button
        self.adc_btn = QPushButton("ADC")
        self.adc_btn.clicked.connect(self.run_adc)
        scan_btns.addWidget(self.adc_btn)

        # Pressure button
        self.SiPM_btn = QPushButton("SiPM")
        #self.pressure_btn.setStyleSheet("background-color: #142d4c; color: #eee;")
        self.SiPM_btn.clicked.connect(self.run_voltage)
        scan_btns.addWidget(self.SiPM_btn)

    
        # Acceleration button
        self.pressure_btn = QPushButton("Pressure")
        #self.accel_btn.setStyleSheet("background-color: #142d4c; color: #eee;")
        self.pressure_btn.clicked.connect(self.run_pressure)
        scan_btns.addWidget(self.pressure_btn)

        # Angular Velocity button
        self.temperature_btn = QPushButton("Temperature")
        #angv_btn.setStyleSheet("background-color: #142d4c; color: #eee;")
        self.temperature_btn.clicked.connect(self.run_temperature)
        scan_btns.addWidget(self.temperature_btn)

        # Linear Acceleration
        self.acc_btn = QPushButton("Linear Acceleration")
        #acc_btn.setStyleSheet("background-color: #142d4c; color: #eee;")
        self.acc_btn.clicked.connect(self.run_acc)
        scan_btns.addWidget(self.acc_btn)

        self.gyro_btn = QPushButton("Angular velocity")
        self.gyro_btn.clicked.connect(self.run_gyro)
        scan_btns.addWidget(self.gyro_btn)

        # Count Distribution button
        self.count_dist_btn = QPushButton("Count Distribution")
        self.count_dist_btn.clicked.connect(self.run_count_distribution)
        scan_btns.addWidget(self.count_dist_btn)

        # Add to layout
        left_panel.addLayout(scan_btns)
        main_layout.addLayout(left_panel, 0, 0)


        # right panel text display
        right_panel = QGridLayout()
        right_panel.setColumnStretch(0, 1)
        right_panel.setColumnStretch(0, 1)

        self.stats_container = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_container)
        self.stats_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_layout.setSpacing(12)  # Slightly more breathing room between boxes

        # New improved style for the QTextEdit "content box"
        stats_style = """
            background-color: #1c2b3a;
            color: #00ffcc;          /* Neon green text */
            border-radius: 8px;
            font-family: 'Consolas', 'Courier New', monospace;  /* Monospaced font for techy look */
            font-size: 14px;
            font-weight: bold;
            background-image: url('refresh_white.png'); 
            background-repeat: no-repeat; 
            background-position: center;
            background-color: #1c2b3a; 
            color: white;        /* <--- text color */
            border-radius: 12px;

            letter-spacing: 1px;    /* space between letters */
        """

        # A separate style just for the LABELS
        #font-family: 'Times New Roman', Times, serif;
        label_style = """
            color: #FFFFFF;
            font-size: 13px;
            font-weight: bold;
            padding-left: 2px;
        """

        def make_labeled_box(label_text, content_text):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(6, 4, 6, 4)
            vbox.setSpacing(4)

            label = QLabel(label_text)
            
            

            textedit = QTextEdit()
            textedit.setReadOnly(True)
            textedit.setFixedHeight(250)
            textedit.setText(content_text)
            textedit.setStyleSheet("""
                border: none;               /* ✅ no border */
                background: transparent;    /* optional, blends with parent */
            """)

            vbox.addWidget(label)
            vbox.addWidget(textedit)

            return container, label, textedit

        # Create all 3 labeled boxes
        

        box1, self.status_label, self.avg_stat1 = make_labeled_box("Status:", "")
        #self.stats_layout.addWidget(box1)
        self.stats_layout.addWidget(box1)
        right_panel.addWidget(self.stats_container, 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.data_ready2.connect(self.avg_stat1.setPlainText) 


        

        self.avg_stat1 = QTextEdit()
        self.avg_stat1.setReadOnly(True)
        self.avg_stat1.setText(f"Run Time: {self.time_stamp}")
        self.avg_stat1.setStyleSheet(stats_style)
        self.avg_stat1.setFixedHeight(70)

        # CAD viewer container
        # --- CAD viewer container
        image_container = QWidget()
        image_container.setFixedWidth(250)
        image_container.setFixedHeight(330)  # 80 (logo) + 250 (GL)   ⬅️ was 250

        image_container.setStyleSheet("background: transparent; border: none;")

        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)
        #image_container.setStyleSheet("border: 1px solid black; padding: 10px; border-radius: 8px;")
        image_container.setStyleSheet("background: transparent; border: none;")
        # --- PNG image ABOVE the STL widget ---
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("background: transparent;")
        self.logo_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.logo_label.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.logo_label.setStyleSheet("background-color: rgba(0,0,0,0); border: none;")

        logo_path = "logo_white.png"  # <-- update to your actual path
        pix = QPixmap(logo_path)
        if not pix.isNull():
            # scale to fit width nicely, keep aspect
            self.logo_label.setPixmap(
                pix.scaled(210, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.logo_label.setText("logo_white.png not found")

        image_layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setFixedSize(250, 250)
        self.gl_widget.setCameraPosition(distance=100)

        self.gl_widget.setBackgroundColor((14, 26, 43, 255))
        your_mesh = mesh.Mesh.from_file("bare_assembly.stl")
        vertices = your_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        image_layout.addWidget(self.gl_widget, alignment=Qt.AlignmentFlag.AlignTop)
        image_layout.addStretch(1)
        image_layout.setContentsMargins(0, 0, 0, 130)

        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        mesh_item = gl.GLMeshItem(
            meshdata=mesh_data, smooth=False, drawFaces=True, drawEdges=True,
            edgeColor=(0.3, 0.3, 0.3, 1), 
            color=(0.9, 0.9, 0.9, 1)
        )
        mesh_item.setGLOptions('opaque')
        self.gl_widget.addItem(mesh_item)

        
        right_panel.addWidget(image_container, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)

        main_layout.addLayout(right_panel, 0, 1, alignment=Qt.AlignmentFlag.AlignTop)
       
        

        #texbox bottom
        bottom_panel = QHBoxLayout()
        self.bottom_widget = QWidget()
        self.bottom_widget.setLayout(bottom_panel)
        #self.bottom_widget.setStyleSheet("background-color:  white; color: white;  border-radius: 3px")
        self.bottom_widget.setStyleSheet("""
            background-color: #1c2b3a;
            color: white;
            border-radius: 3px;
        """)
        self.bottom_widget.setFixedHeight(200)

        self.feed_box = QTextEdit()
        self.feed_box.setReadOnly(True)
        self.feed_box.setStyleSheet("""
            QTextEdit {
                background-color: #1c2b3a; 
                color: #FFFFFF; 
                border-radius: 3px;
                padding: 8px;
                padding-bottom: 12px;
            }
        """)
        self.feed_box.setFixedHeight(200)
        # Set document margins and line height to prevent text clipping
        self.feed_box.document().setDocumentMargin(3)
        # Ensure the viewport has proper margins
        self.feed_box.setViewportMargins(0, 0, 0, 8)
        bottom_panel.addWidget(self.feed_box)

        # ✅ Add bottom panel to row 1, spanning 2 columns
        main_layout.addWidget(self.bottom_widget, 2, 0, 1, 2)
        self.setLayout(main_layout)
        #self.feed_box.setText(f'Welcome to CosmicDAQ GUI')
        self.feed_box.setText(f' -- Welcome to CosmicDAQ Graphical User Interface -- \nOperating System: {platform.system()}')


        self.port_dropdown.addItem("Select Port") 
        
        t1 = time.time()
        ports = serial_ports()
        if (time.time()-t1)>6:
           self.feed_box.setText('Listing ports is taking unusually long...')
        for port in ports:
            self.port_dropdown.addItem(port)
        if not ports:
            self.port_dropdown.addItem("No ports found")
        control_btns.addWidget(self.port_dropdown)
        #adding all buttons
        left_panel.addLayout(control_btns)

        self.port_dropdown.activated.connect(lambda idx: self.handle_port_selected(self.port_dropdown.itemText(idx)))
        

        refresh_btn = QToolButton()
        refresh_btn.setIcon(QIcon("refresh_white.png"))
        refresh_btn.setIconSize(QSize(20, 20))
        refresh_btn.setStyleSheet(button_style)
        control_btns.addWidget(refresh_btn)
        refresh_btn.clicked.connect(self.refresh_ports)
        refresh_btn.setToolTip("Refresh Serial Ports")

        self.theme_toggle = QToolButton()
        self.theme_toggle.setText("☀️ / 🌙")
        self.theme_toggle.setStyleSheet(button_style)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        self.theme_toggle.setToolTip("Toggle Light/Dark Theme")
        control_btns.addWidget(self.theme_toggle)
        
        # Close/Quit button
        close_btn = QToolButton()
        close_btn.setText("X")
        close_btn.setStyleSheet(button_style)
        close_btn.clicked.connect(self.close_application)
        close_btn.setToolTip("Close Application")
        control_btns.addWidget(close_btn)
        

    def close_application(self):
        """Properly close the application and clean up resources."""
        # Stop any active recording
        if hasattr(self, 'read_serial_active') and self.read_serial_active:
            self.stop_file()
        
        # Close the application
        self.feed_box.append("Closing application...")
        QApplication.quit()

    def save_plot(self):
        """Save the current plot to a file."""
        # Get the current plot type from the toolbar
        plot_type = getattr(self.toolbar, 'plot_type', 'plot')
        
        # Determine default filename based on plot type and data source
        if hasattr(self, 'cw') and hasattr(self.cw, 'file_path'):
            # Use the loaded file's base name
            base_name = self.cw.file_path.replace('.txt', '')
        else:
            # Use generic name for live data
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"plot_{timestamp}"
        
        default_filename = f"{base_name}_{plot_type}.pdf"
        
        # Open file dialog to choose save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            default_filename,
            "PDF Files (*.pdf);;PNG Files (*.png);;All Files (*)"
        )
        
        if filename:
            try:
                self.static_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                self.feed_box.append(f"Plot saved to: {filename}")
            except Exception as e:
                self.feed_box.append(f"Error saving plot: {str(e)}")

    def apply_theme(self):
        t = self.themes[self.current_theme]

        # Window
        self.setStyleSheet(f"background-color: {t['bg']}; color: {t['fg']};")

        # Feed box + stats
        self.feed_box.setStyleSheet(f"background-color: {t['panel']}; color: {t['fg']}; border-radius: 3px;")
        
        #self.bottom_widget.setStyleSheet(
        #    f"""
        #    background-color: {t['panel']};
        #    color: {t['fg']};
        #    border: 1px solid black;
        #    border-radius: 3px;
        #    """
        #)
        '''
        QPushButton, QToolButton {
                background-color: #122c3d;
                color: #00ffcc;              /* keep cyan text/icons */
                border: 1px solid white;     /* <-- white border */
                border-radius: 8px;
                padding: 6px;
            }

            QPushButton:hover, QToolButton:hover {
                background-color: #0e1a2b;
                border: 1px solid white;     /* <-- white border */
                color: #00ffff;
            }

            QPushButton:pressed, QToolButton:pressed {
                background-color: #0b1a29;
                border: 1px solid white;     /* <-- white border */
                color: #00cccc;
            }
            """
        

        button_style = f"""
            QPushButton {{
                background-color: {t['button']};
                color: {t['fg']};
                border: 1px solid {t['fg']};
                border-radius: 6px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {t['bg']};
                color: {t['fg']};
                border: 1px solid {t['fg']};
                border-radius: 6px;
                padding: 4px;
            }}
        """
        '''
        button_style = f"""
            QPushButton {{
                background-color: {t['button_background']};
                color: {t['button_text']};
                border: 1px solid {t['button_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {t['button_hover']};
                color: {t['button_text']};
                border: 1px solid {t['button_border']};
                border-radius: 6px;
                padding: 4px;
            }}
        """

        for btn in [self.rate_btn, self.deadtime_btn, self.adc_btn, self.pressure_btn,
            self.temperature_btn, self.acc_btn, self.gyro_btn, self.SiPM_btn, self.count_dist_btn]:
            btn.setStyleSheet(button_style)

        # Apply to all bottom buttons
        #for btn in [self.rate_btn, self.adc_btn, self.pressure_btn,
        #            self.temp_btn, self.acc_btn, self.gyro_btn]:
        #    btn.setStyleSheet(button_style)

        self.avg_stat1.setStyleSheet(f"background-color: {t['panel']}; color: {t['accent']}; border-radius: 8px;")
        self.binning_label.setStyleSheet(f"""
            color: {t['fg']};
            padding-right: 10px;
            font-size: 18px;
            font-weight: bold;
        """)
        
        # Custom bin input styling
        self.custom_bin_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {t['button_background']};
                color: {t['fg']};
                border: 1px solid {t['button_border']};
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                border: 2px solid {t['accent']};
            }}
        """)
        
        # update coord_label
        # Coordinates label
        self.coord_label.setStyleSheet(f"color: {t['fg']};")

        self.status_label.setStyleSheet(f"""
            font-family: 'Times New Roman', Times, serif;
            font-size: 16pt;
            font-weight: bold;
            color: {t['fg']};
            """)
        self.avg_stat1.setStyleSheet(f"""
            background-color: {t['panel']};
            color: {t['fg']};
            border-radius: 8px;
        """)
        
        if self.current_theme == "dark":
            bg = (14, 26, 43, 255)   # dark blue
        else:
            bg = (255, 255, 255, 255)  # pure white
        self.gl_widget.setBackgroundColor(bg)
        
        #logo_file = "logo_white.png" if self.current_theme == "dark" else "logo_black.png"
        # ✅ Update logo depending on theme
        logo_file = "logo_white.png" if self.current_theme == "dark" else "logo_black.png"
        pix = QPixmap(logo_file)
        if not pix.isNull():
            self.logo_label.setPixmap(
                pix.scaled(210, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.logo_label.setText(f"{logo_file} not found")

        for spine in self.static_ax.spines.values():
            spine.set_edgecolor(t["fg"])
        self.static_ax.grid(True, color=t["fg"], linestyle='--', alpha=0.3)

        # ✅ Legend (if one exists)
        leg = self.static_ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_color(t["fg"])

        for line in self.static_ax.get_lines():
            line.set_markerfacecolor(t["fg"])
            line.set_markeredgecolor(t["fg"])
            
        # Matplotlib figure
        self.static_ax.set_facecolor(t["bg"])
        self.static_canvas.figure.set_facecolor(t["bg"])
        self.static_ax.title.set_color(t["fg"])
        self.static_ax.xaxis.label.set_color(t["fg"])
        self.static_ax.yaxis.label.set_color(t["fg"])
        self.static_ax.tick_params(axis='x', colors=t["fg"])
        self.static_ax.tick_params(axis='y', colors=t["fg"])
        self.static_ax.grid(True, color=t["fg"], linestyle='--', alpha=0.3)
        self.static_canvas.draw()


    def recalc_cw(self):
        """Recalculate CWClass when binning changes."""
        if hasattr(self, "cw"):   # Only if a file has already been loaded
            # Get the file path
            if hasattr(self.cw, 'file_path'):
                file_name = self.cw.file_path
            else:
                self.feed_box.append("Error: Cannot recalculate - file path not stored")
                return
            
            # Reload with new bin size
            try:
                self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box)
                self.cw.file_path = file_name  # Store path again for next recalc
                self.feed_box.append(f"Recalculated with bin size = {self.selected_bin_time}s")
                self.run_rate()   # auto-refresh the plot
            except Exception as e:
                self.feed_box.append(f"Error recalculating: {str(e)}")
        else:
            self.feed_box.append("No data loaded yet. Load a file first.")
            
    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def load_file(self):
        if self.binning_selected == False: 
                self.feed_box.append("Select bin time first.")
        if self.binning_selected:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load File", "", "Text Files (*.txt);;All Files (*)")
            if file_name:
                # Extract just the filename from the full path
                file_basename = file_name.split('/')[-1]
                self.feed_box.append(f"Opening data file: {file_basename}")
                
                try:
                    self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box)
                    self.cw.file_path = file_name   # ⬅️ store so we can reload later
                    self.run_rate()
                    
                    self.data_ready2.emit(
                        f"Total Live Time: {self.cw.live_time_s:,.2f} s\n"
                        f"\n"
                        f"Total Singles: {self.cw.total_counts:,}\n"
                        f"Rate: {self.cw.count_rate:,.4f} ± {self.cw.count_rate_err:,.4f} Hz\n"
                        f"\n"
                        f"Total Coincidences: {self.cw.total_coincident:,}\n"
                        f"Rate: {self.cw.count_rate_coincident:,.4f} ± {self.cw.count_rate_err_coincident:,.4f} Hz"
                    )
                except (ValueError, IndexError) as e:
                    self.feed_box.append(f"Error loading file: {str(e)}")
                    return
                #if file_name:
                #self.cw = CWClass(file_name, self.selected_bin_time,self.feed_box)
                # ✅ Only if a file was actually chosen
                #self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box)

                # ✅ Automatically plot the rate plot after import
                #self.run_rate()

    def select_custom_bin(self):
        """Handle custom bin time input."""
        text = self.custom_bin_input.text().strip()
        if not text:
            return
        
        try:
            custom_value = int(text)
            if custom_value <= 0:
                self.feed_box.append("Error: Bin time must be a positive integer.")
                return
            
            # Set custom bin time
            self.binning_selected = True
            self.selected_bin_time = custom_value
            self.feed_box.append(f"Selected bin time: {custom_value}s")
            self.recalc_cw()
            
        except ValueError:
            self.feed_box.append("Error: Please enter a valid integer for bin time.")
    
    class NPlot():
        def __init__(self, 
                    data,
                    weights,
                    colors,
                    labels,
                    xmin,xmax,ymin,ymax,
                    ax,
                    figsize = [8,6],fontsize = 15,nbins = 101, alpha = 0.85,fit_gaussian=False,
                    xscale = 'log',yscale = 'log',xlabel = '',loc = 1,pdf_name='',lw=2, title=''):
            #
            # fg = "#ffffff" if self.parent.current_theme == "dark" else "#000000"
            self.static_ax = ax
            if xscale == 'log':
                bins = np.logspace(np.log10(xmin),np.log10(xmax),nbins)
            if xscale == 'linear':
                bins = np.linspace(xmin,xmax,nbins)
            self.static_ax.clear()
            
            self.static_ax.set_axisbelow(True)
            self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
            #self.static_ax.set_title(title, fontsize=fontsize + 1, color = "white")
            #self.static_ax.title.set_color("white")

            hist_data = []
            std = []
            bin_centers = []

            # Define the Gaussian function
            def gaussian(x, a, mu, sigma):
                return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
            
            for i in range(len(data)):
                valid_data = data[i][~np.isnan(data[i])]
                valid_weights = weights[i][~np.isnan(weights[i])]

                counts, bin_edges = np.histogram(valid_data, bins=bins, weights=valid_weights)
                bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                sum_weights_sqrd, _ = np.histogram(valid_data, bins=bins, weights=np.power(valid_weights, 2))

                hist_data.append(counts)
                upper_value = plusSTD(counts,sum_weights_sqrd)
                lower_value = subSTD(counts,sum_weights_sqrd)
                std.append([upper_value,lower_value])
                bin_centers.append(bin_center)
                fill_between_steps(bin_center, upper_value,lower_value,  color = colors[i],alpha = alpha,lw=lw,ax=self.static_ax)
                self.static_ax.plot([1e14,1e14], label = labels[i],color = colors[i],alpha = alpha,linewidth = 2)
                    

            self.static_ax.set_yscale(yscale)
            self.static_ax.set_xscale(xscale)
            self.static_ax.legend(fontsize=fontsize - 4, loc='upper right', fancybox=False, frameon=False)
            legend = self.static_ax.legend(fontsize=fontsize - 4, loc='upper right', fancybox=False, frameon=False)

            #for text in legend.get_texts():
            #    text.set_color('white')
            self.static_ax.set_xlabel(xlabel, size=fontsize)
            self.static_ax.set_ylabel(r'Rate/bin [s$^{-1}$]', size=fontsize)#3, color = fg)
            self.static_ax.set_xlim(xmin, xmax)
            self.static_ax.set_ylim(ymin, ymax)

            
            self.static_ax.tick_params(axis='both', which='major', labelsize=fontsize-3)
            self.static_ax.tick_params(axis='both', which='minor', labelsize=fontsize-3) 


            

            self.static_ax.figure.tight_layout()
            self.static_ax.figure.canvas.draw()   
            
    class ratePlot():
        def __init__(self,
                    time,
                    count_rates,
                    count_rates_err,
                    colors,
                    labels,
                    xmin,xmax,ymin,ymax,
                    ax,
                    figsize = [8,8],fontsize = 16, alpha = 0.9,
                    xscale = 'linear',yscale = 'linear',
                    xlabel = '',ylabel = '',
                    loc = 2, pdf_name='',title = ''):
            
            #fg = "#ffffff" if self.parent.current_theme == "dark" else "#000000"
            self.static_ax = ax
            self.static_ax.set_axisbelow(True)
            self.static_ax.clear()
            self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
            
            for i in range(len(count_rates)):
                self.static_ax.errorbar(time[i], 
                            count_rates[i],
                            xerr=0, yerr=count_rates_err[i],
                            fmt='o',label = labels[i], linewidth = 2, ecolor = colors[i], markersize = 2)             # theme-dependent)

            self.static_ax.set_yscale(yscale)
            self.static_ax.set_xscale(xscale)
            self.static_ax.set_ylabel(ylabel,size=fontsize, color='white')
            self.static_ax.set_xlabel(xlabel,size=fontsize, color='white')
            self.static_ax.set_xlim(xmin, xmax)
            self.static_ax.set_ylim(ymin, ymax)
            
            self.static_ax.tick_params(axis='both', which='major', labelsize=fontsize-3)
            self.static_ax.tick_params(axis='both', which='minor', labelsize=fontsize-3) 
            self.static_ax.xaxis.labelpad = 0 

            self.static_ax.legend(fontsize=fontsize - 4, loc='upper right', fancybox=False, frameon=False)
            legend = self.static_ax.legend(fontsize=fontsize - 4, loc='upper right', fancybox=False, frameon=False)

            for text in legend.get_texts():
                text.set_color('white')

            #legend = self.static_ax.legend(fontsize=fontsize-3,loc = 'upper right',  fancybox = False,frameon=False),
            # Set legend text to white
            #for text in legend.get_texts():
            #    text.set_color("white")
            
            #self.static_ax.set_title(title,fontsize=fontsize+1, color = "white")
            self.static_ax.figure.tight_layout()
            self.static_ax.figure.canvas.draw()   
            
    def get_live_cw_object(self):
        """Create a CWClass-like object from live recording data."""
        if not hasattr(self, 'live_data') or len(self.live_data['event_number']) == 0:
            return None
        
        # Create a simple object to hold the data
        class LiveCW:
            pass
        
        cw = LiveCW()
        cw.name = "Live Data"
        cw.file_path = "live"
        
        # Convert lists to numpy arrays
        cw.event_number = np.array(self.live_data['event_number'])
        cw.PICO_timestamp_s = np.array(self.live_data['PICO_timestamp_s'])
        cw.select_coincident = np.array(self.live_data['coincident'])
        cw.adc = np.array(self.live_data['adc'])
        cw.sipm = np.array(self.live_data['sipm'])
        cw.temperature = np.array(self.live_data['temperature'])
        cw.pressure = np.array(self.live_data['pressure'])
        cw.accel_x = np.array(self.live_data['accel_x'])
        cw.accel_y = np.array(self.live_data['accel_y'])
        cw.accel_z = np.array(self.live_data['accel_z'])
        cw.gyro_x = np.array(self.live_data['gyro_x'])
        cw.gyro_y = np.array(self.live_data['gyro_y'])
        cw.gyro_z = np.array(self.live_data['gyro_z'])
        
        # Calculate time-based values
        deadtime = np.array(self.live_data['deadtime'])
        event_deadtime_s = np.diff(np.append([0], deadtime))
        cw.event_deadtime_s = event_deadtime_s
        cw.PICO_event_livetime_s = np.diff(np.append([0], cw.PICO_timestamp_s)) - event_deadtime_s
        
        cw.PICO_total_time_s = np.max(cw.PICO_timestamp_s) - np.min(cw.PICO_timestamp_s)
        cw.total_deadtime_s = np.max(deadtime) - np.min(deadtime)
        cw.live_time_s = cw.PICO_total_time_s - cw.total_deadtime_s
        
        cw.total_counts = int(np.max(cw.event_number) - np.min(cw.event_number))
        cw.weights = np.ones(len(cw.event_number)) / cw.live_time_s
        
        cw.count_rate = cw.total_counts / cw.live_time_s
        cw.count_rate_err = np.sqrt(cw.total_counts) / cw.live_time_s
        
        # Calculate coincident counts
        cw.total_coincident = np.sum(cw.select_coincident)
        cw.count_rate_coincident = cw.total_coincident / cw.live_time_s
        cw.count_rate_err_coincident = np.sqrt(cw.total_coincident) / cw.live_time_s
        
        cw.total_non_coincident = cw.total_counts - cw.total_coincident
        cw.count_rate_non_coincident = cw.total_non_coincident / cw.live_time_s
        cw.count_rate_err_non_coincident = np.sqrt(cw.total_non_coincident) / cw.live_time_s
        
        # Bin the data
        bin_size = getattr(self, 'selected_bin_time', 30)
        bins = range(int(np.min(cw.PICO_timestamp_s)), int(np.max(cw.PICO_timestamp_s)), bin_size)
        
        counts, binEdges = np.histogram(cw.PICO_timestamp_s, bins=bins)
        bin_livetime, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.PICO_event_livetime_s)
        
        cw.binned_count_rate = counts / bin_livetime
        cw.binned_count_rate_err = np.sqrt(counts) / bin_livetime
        
        counts_coincident, _ = np.histogram(cw.PICO_timestamp_s[cw.select_coincident], bins=bins)
        cw.binned_count_rate_coincident = counts_coincident / bin_livetime
        cw.binned_count_rate_err_coincident = np.sqrt(counts_coincident) / bin_livetime
        
        counts_non_coincident, _ = np.histogram(cw.PICO_timestamp_s[~cw.select_coincident], bins=bins)
        cw.binned_count_rate_non_coincident = counts_non_coincident / bin_livetime
        cw.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident) / bin_livetime
        
        # Bin other measurements
        sum_pressure, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.pressure)
        count_pressure, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)
        
        sum_temperature, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.temperature)
        count_temperature, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)
        
        sum_accel_x, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.accel_x)
        count_accel_x, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_accel_x = sum_accel_x / np.maximum(count_accel_x, 1)
        
        sum_accel_y, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.accel_y)
        count_accel_y, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_accel_y = sum_accel_y / np.maximum(count_accel_y, 1)
        
        sum_accel_z, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.accel_z)
        count_accel_z, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_accel_z = sum_accel_z / np.maximum(count_accel_z, 1)
        
        sum_gyro_x, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.gyro_x)
        count_gyro_x, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_gyro_x = sum_gyro_x / np.maximum(count_gyro_x, 1)
        
        sum_gyro_y, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.gyro_y)
        count_gyro_y, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_gyro_y = sum_gyro_y / np.maximum(count_gyro_y, 1)
        
        sum_gyro_z, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=cw.gyro_z)
        count_gyro_z, _ = np.histogram(cw.PICO_timestamp_s, bins=bins)
        cw.binned_gyro_z = sum_gyro_z / np.maximum(count_gyro_z, 1)
        
        # Calculate binned deadtime percentage
        bin_deadtime, _ = np.histogram(cw.PICO_timestamp_s, bins=bins, weights=event_deadtime_s)
        cw.binned_deadtime_percentage = bin_deadtime / bin_size * 100
        
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        cw.binned_time_s = bincenters
        cw.binned_time_m = bincenters / 60.
        
        return cw

    def run_adc(self):
        self.toolbar.plot_type = "ADC"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        x_min = np.min(f1.adc)
        x_max = np.max(f1.adc)

        #if self.static_ax.get_xscale() == "log":
        #    # multiplicative padding (e.g. 10% wider on each side in log space)
        #    padding_factor = 0.1
        #    xmin = x_min / (1 + padding_factor)
        #    xmax = x_max * (1 + padding_factor)
        #else:
        #    # additive padding (10% of span)
        span = x_max - x_min
        padding = 0.1 * span
        xmin = x_min - padding
        xmax = x_max + padding


        c = self.NPlot(
        data=[ f1.adc,f1.adc[~f1.select_coincident],f1.adc[f1.select_coincident]],
        weights=[f1.weights,f1.weights[~f1.select_coincident],f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3],mycolors[1]],
        labels=[r'All Events:  ' + f'{f1.count_rate:.5f}' + '+/-' + f'{f1.count_rate_err:.5f}' +' Hz',
                r'Non-Coincident:  ' + f'{f1.count_rate_non_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_non_coincident:.5f}' +' Hz',
                r'Coincident: ' + f'{f1.count_rate_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_coincident:.5f}' +' Hz'],
        ax = self.static_ax,
        xmin=min(f1.adc), xmax= max(f1.adc),  ymin=0.1e-3, ymax=1.1,nbins=101,
        xlabel='Measured 12-bit ADC peak value [0-4095]',
        pdf_name= '_ADC.pdf',title = 'ADC Measurement')
        self.apply_theme()

    def run_temperature(self):
        self.toolbar.plot_type = "temperature"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        y_min = np.min(f1.binned_temperature)
        y_max = np.max(f1.binned_temperature)
        span = y_max - y_min
        padding = 0.1 * span   # 5% of the span

        # BMP280 temperature sensor relative accuracy: ±0.1°C (typical)
        # Reference: BMP280 Datasheet, Table 3 (Typical Performance)
        temperature_uncertainty = 0.1  # °C

        c = self.ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_temperature],
        count_rates_err = [np.ones(len(f1.binned_temperature)) * temperature_uncertainty],
        colors =[mycolors[5]],
        ax = self.static_ax,
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin = y_min - padding,ymax = y_max + padding,
        figsize = [7,5],
        fontsize = 16,alpha = 1,labels=['Temperature Data'],
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Temperature [$^{\circ}$C]',
        loc = 4,pdf_name='_temperature.pdf',title = 'Temperature Measurement')
        
        # Add error bar annotation
        self.static_ax.text(0.98, 0.02, f'Error bars: ±{temperature_uncertainty}°C', 
                           transform=self.static_ax.transAxes,
                           fontsize=10, horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()
    
    def run_gyro(self):
        self.toolbar.plot_type = "gyro"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        all_y = np.concatenate([
            f1.binned_gyro_x,
            f1.binned_gyro_y,
            f1.binned_gyro_z
            ])

        y_min = np.min(all_y)
        y_max = np.max(all_y)
        span = y_max - y_min
        padding = 0.1 * span   # 5% of the span

        ymin = y_min - padding
        ymax = y_max + padding

        # MPU6050 gyroscope typical noise: ±0.1 deg/s (at ±250 deg/s scale)
        # Reference: MPU6050 Datasheet, Section 6.2 (Gyroscope Specifications)
        gyro_uncertainty = 0.1  # deg/s

        c = self.ratePlot(
            time=[f1.binned_time_m, f1.binned_time_m, f1.binned_time_m],
            count_rates=[f1.binned_gyro_x, f1.binned_gyro_y, f1.binned_gyro_z],
            count_rates_err=[
                np.ones(len(f1.binned_gyro_x)) * gyro_uncertainty,
                np.ones(len(f1.binned_gyro_y)) * gyro_uncertainty,
                np.ones(len(f1.binned_gyro_z)) * gyro_uncertainty,
            ],
            colors=[mycolors[2], mycolors[3], mycolors[4]],
            labels=[r'Angular velocity X', r'Angular velocity Y', r'Angular velocity Z'],
            ax=self.static_ax,
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=ymin,
            ymax=ymax,
            figsize=[7, 5],
            fontsize=16,
            alpha=1,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Angular velocity [deg/s]',  # ← your original 'ylabel' string was cut off
        )
        
        # Add error bar annotation
        self.static_ax.text(0.98, 0.02, f'Error bars: ±{gyro_uncertainty} deg/s', 
                           transform=self.static_ax.transAxes,
                           fontsize=10, horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()
    

    def run_acc(self):
        self.toolbar.plot_type = "linear_acceleration"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        all_y = np.concatenate([
            f1.binned_accel_x,
            f1.binned_accel_y,
            f1.binned_accel_z
            ])
        y_min = np.min(all_y)
        y_max = np.max(all_y)
        span = y_max - y_min
        padding = 0.1 * span   # 5% of the span

        ymin = y_min - padding
        ymax = y_max + padding
        
        # MPU6050 accelerometer typical noise: ±1 mg (0.001 g)
        # Reference: MPU6050 Datasheet, Section 6.1 (Accelerometer Specifications)
        accel_uncertainty = 0.001  # g (1 mg)
        
        c = self.ratePlot(
            time=[f1.binned_time_m, f1.binned_time_m, f1.binned_time_m],
            count_rates=[f1.binned_accel_x, f1.binned_accel_y, f1.binned_accel_z],
            count_rates_err=[
                np.ones(len(f1.binned_accel_x)) * accel_uncertainty,
                np.ones(len(f1.binned_accel_y)) * accel_uncertainty,
                np.ones(len(f1.binned_accel_z)) * accel_uncertainty,
            ],
            colors=[mycolors[2], mycolors[3], mycolors[4]],
            labels=[r'Acceleration X', r'Acceleration Y', r'Acceleration Z'],
            ax=self.static_ax,
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=ymin,
            ymax=ymax,
            figsize=[7, 5],
            fontsize=16,
            alpha=1,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Acceleration [g]',  # ← your original 'ylabel' string was cut off
        )
        
        # Add error bar annotation
        self.static_ax.text(0.98, 0.02, f'Error bars: ±{accel_uncertainty*1000:.1f} mg', 
                           transform=self.static_ax.transAxes,
                           fontsize=10, horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()
    
    def run_pressure(self):
        self.toolbar.plot_type = "pressure"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        y_min = np.min(f1.binned_pressure)
        y_max = np.max(f1.binned_pressure)
        span = y_max - y_min
        #padding = 0.05 * span   # 5% of the span
        padding = max(0.05 * span, 1000)   # 5% of span OR 1000

        # BMP280 pressure sensor absolute accuracy: ±100 Pa (±1 hPa)
        # Reference: BMP280 Datasheet, Table 3 (Typical Performance)
        pressure_uncertainty = 100  # Pa
        
        c = self.ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_pressure],
        count_rates_err = [np.ones(len(f1.binned_pressure)) * pressure_uncertainty],
        colors =[mycolors[6]],
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin =y_min - padding,ymax =y_max + padding,
        figsize = [7,5],labels=['Pressure Data'],
        fontsize = 16,alpha = 1,
        ax = self.static_ax,
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Pressure [Pa]',
        loc = 4,pdf_name='_pressure.pdf',title = 'Pressure Measurement')
        
        # Add error bar annotation
        self.static_ax.text(0.98, 0.02, f'Error bars: ±{pressure_uncertainty} Pa', 
                           transform=self.static_ax.transAxes,
                           fontsize=10, horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()

    def run_voltage(self):
        self.toolbar.plot_type = "SiPM_pulse_height"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
        
        # Calculate xmin/xmax with protection for log scale (must be positive)
        sipm_min = min(f1.sipm)
        sipm_max = max(f1.sipm)
        xmin = max(0.01, sipm_min - 2)  # Ensure xmin is at least 0.1 for log scale
        xmax = sipm_max + 100
            
        c = self.NPlot(
        data=[f1.sipm, f1.sipm[~f1.select_coincident],f1.sipm[f1.select_coincident]],
        weights=[f1.weights, f1.weights[~f1.select_coincident],f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3],mycolors[1]],
        labels=[r'All Events:  ' + f'{f1.count_rate:.5f}' + '+/-' + f'{f1.count_rate_err:.5f}' +' Hz',
                r'Non-Coincident:  ' + f'{f1.count_rate_non_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_non_coincident:.5f}' +' Hz',
                r'Coincident: ' + f'{f1.count_rate_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_coincident:.5f}' +' Hz'],
        ax = self.static_ax,
        xmin=xmin, xmax=xmax, ymin=0.1e-3, ymax=1.1,xscale='log',
        xlabel='SiPM Peak Voltage [mV]',fit_gaussian=True,
        pdf_name='_SiPM_peak_voltage.pdf',title = 'SiPM Peak Voltage Measurement',)
        self.apply_theme()

    def run_rate(self):

        self.toolbar.plot_type = "rate" 
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
            
        #self.toolbar.plot_type = "rate"
        c = self.ratePlot(time = [f1.binned_time_m,f1.binned_time_m,f1.binned_time_m],
        count_rates = [f1.binned_count_rate,f1.binned_count_rate_non_coincident,f1.binned_count_rate_coincident],
        count_rates_err = [f1.binned_count_rate_err,f1.binned_count_rate_err_non_coincident,f1.binned_count_rate_err_coincident], 
        colors=[mycolors[7], mycolors[3], mycolors[1]],
        labels=[r'All Events: ' + f'{f1.count_rate:.5f}' + '+/-' + f'{f1.count_rate_err:.5f}' +' Hz', 
                r'Non-Coincident:  ' + f'{f1.count_rate_non_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_non_coincident:.5f}' +' Hz',
                r'Coincident:  ' + f'{f1.count_rate_coincident:.5f}' + '+/-' + f'{f1.count_rate_err_coincident:.5f}' +' Hz'],
        ax = self.static_ax,
        xmin = min(f1.binned_time_m) if len(f1.binned_time_m) > 0 else 0, 
        xmax = max(f1.binned_time_m) if len(f1.binned_time_m) > 0 else 1,
        ymin = 0,
        ymax = 1.3*max(f1.binned_count_rate) if len(f1.binned_count_rate) > 0 else 1,
        figsize = [7,5],
        fontsize = 16,alpha = 1,
        xscale = 'linear',yscale = 'linear',xlabel = 'Time since first event [min]',ylabel = r'Rate [s$^{-1}$]',
        loc = 1, pdf_name='_rate.pdf',title = 'Detector Count Rate')
        #print(f1.binned_count_rate_err_non_coincident)
        self.static_canvas.draw()
        
        # Add bin size annotation in upper left corner
        bin_size = getattr(self, 'selected_bin_time', 30)
        self.static_ax.text(0.01, 0.98, f'Time interval: {bin_size}s', 
                           transform=self.static_ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()

    def run_deadtime(self):
        self.toolbar.plot_type = "deadtime"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data available. Load a file or start recording.")
            return
        
        # Check if binned_deadtime_percentage exists
        if not hasattr(f1, 'binned_deadtime_percentage'):
            self.feed_box.append("Deadtime data not available for this dataset.")
            return
            
        y_min = np.min(f1.binned_deadtime_percentage)
        y_max = np.max(f1.binned_deadtime_percentage)
        
        # Use log scale, so set reasonable bounds
        ymin = max(0.01, y_min * 0.5)  # Don't go below 0.01% for log scale
        ymax = y_max * 2.0
        
        c = self.ratePlot(
            time=[f1.binned_time_m],
            count_rates=[f1.binned_deadtime_percentage],
            count_rates_err=[np.zeros(len(f1.binned_time_m))],  # No error bars for deadtime
            colors=[mycolors[6]],
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=ymin,
            ymax=ymax,
            figsize=[7, 5],
            labels=['Deadtime Percentage'],
            fontsize=16,
            alpha=1,
            ax=self.static_ax,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Deadtime Percentage [%]',
            loc=2,
            pdf_name='_deadtime.pdf',
            title='Deadtime Measurement'
        )
        
        # Add bin size annotation
        bin_size = getattr(self, 'selected_bin_time', 30)
        self.static_ax.text(0.01, 0.98, f'Time interval: {bin_size}s', 
                           transform=self.static_ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.apply_theme()

    def run_count_distribution(self):
        """Plot the distribution of event counts per time interval with Poisson overlay."""
        from scipy.stats import poisson
        
        self.toolbar.plot_type = "count_distribution"
        # Use live data if recording, otherwise use loaded file data
        f1 = self.get_live_cw_object() if self.read_serial_active else getattr(self, 'cw', None)
        
        if f1 is None:
            self.feed_box.append("No data loaded. Please load a file first.")
            return
        
        # Get the bin size
        bin_size = getattr(self, 'selected_bin_time', 30)
        
        # Calculate histogram of counts per bin for each event type
        bins = range(int(np.min(f1.PICO_timestamp_s)), int(np.max(f1.PICO_timestamp_s)), bin_size)
        
        # All events
        counts_all, _ = np.histogram(f1.PICO_timestamp_s, bins=bins)
        lambda_all = np.mean(counts_all)
        
        # Non-coincident events
        counts_non_coin, _ = np.histogram(f1.PICO_timestamp_s[~f1.select_coincident], bins=bins)
        lambda_non_coin = np.mean(counts_non_coin)
        
        # Coincident events
        counts_coin, _ = np.histogram(f1.PICO_timestamp_s[f1.select_coincident], bins=bins)
        lambda_coin = np.mean(counts_coin)
        
        # Determine the maximum count across all types
        max_count = int(max(np.max(counts_all), np.max(counts_non_coin), np.max(counts_coin)))
        count_bins = np.arange(0, max_count + 2) - 0.5  # Bin edges
        
        # Get observed distributions
        total_intervals = len(counts_all)
        
        obs_all, _ = np.histogram(counts_all, bins=count_bins)
        freq_all = obs_all / total_intervals
        
        obs_non_coin, _ = np.histogram(counts_non_coin, bins=count_bins)
        freq_non_coin = obs_non_coin / total_intervals
        
        obs_coin, _ = np.histogram(counts_coin, bins=count_bins)
        freq_coin = obs_coin / total_intervals
        
        # Generate Poisson distributions
        x_poisson = np.arange(0, max_count + 1)
        y_poisson_all = poisson.pmf(x_poisson, lambda_all)
        y_poisson_non_coin = poisson.pmf(x_poisson, lambda_non_coin)
        y_poisson_coin = poisson.pmf(x_poisson, lambda_coin)
        
        # Clear the plot
        self.static_ax.clear()
        
        # Bar width for grouped bars
        bar_width = 0.25
        x_positions = x_poisson
        
        # Plot observed distributions as grouped bars with colors matching other plots
        # All events: mycolors[7] (cyan/teal)
        self.static_ax.bar(x_positions - bar_width, freq_all, width=bar_width, alpha=0.7, 
                          color=mycolors[7], label='All Events (Obs)', edgecolor='white')
        
        # Non-coincident: mycolors[3] (orange)
        self.static_ax.bar(x_positions, freq_non_coin, width=bar_width, alpha=0.7, 
                          color=mycolors[3], label='Non-Coincident (Obs)', edgecolor='white')
        
        # Coincident: mycolors[1] (red)
        self.static_ax.bar(x_positions + bar_width, freq_coin, width=bar_width, alpha=0.7, 
                          color=mycolors[1], label='Coincident (Obs)', edgecolor='white')
        
        # Overlay Poisson distributions as lines
        self.static_ax.plot(x_poisson, y_poisson_all, color=mycolors[7], linestyle='--', 
                           linewidth=2, marker='o', markersize=4, 
                           label=f'All Poisson (λ={lambda_all:.2f})')
        
        self.static_ax.plot(x_poisson, y_poisson_non_coin, color=mycolors[3], linestyle='--', 
                           linewidth=2, marker='s', markersize=4,
                           label=f'Non-Coin Poisson (λ={lambda_non_coin:.2f})')
        
        self.static_ax.plot(x_poisson, y_poisson_coin, color=mycolors[1], linestyle='--', 
                           linewidth=2, marker='^', markersize=4,
                           label=f'Coincident Poisson (λ={lambda_coin:.2f})')
        
        # Set labels and title
        self.static_ax.set_xlabel(f'Number of Events per {bin_size}s Interval', fontsize=15)
        self.static_ax.set_ylabel('Probability', fontsize=15)
        self.static_ax.set_title('Event Count Distribution vs Poisson', fontsize=16, color='white')
        
        # Set x-axis to show reasonable number of integer ticks
        # Determine tick spacing to avoid overlap
        if max_count <= 10:
            tick_spacing = 1
        elif max_count <= 20:
            tick_spacing = 2
        elif max_count <= 50:
            tick_spacing = 5
        else:
            tick_spacing = 10
        
        x_ticks = np.arange(0, max_count + 1, tick_spacing)
        self.static_ax.set_xticks(x_ticks)
        
        # Add legend
        self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False, ncol=2)
        
        # Add statistics annotation on the right side
        stats_text = (f'All: μ={lambda_all:.2f}, σ={np.std(counts_all):.2f}\n'
                     f'Non-Coin: μ={lambda_non_coin:.2f}, σ={np.std(counts_non_coin):.2f}\n'
                     f'Coincident: μ={lambda_coin:.2f}, σ={np.std(counts_coin):.2f}\n'
                     f'Intervals: {total_intervals}')
        self.static_ax.text(0.98, 0.65, stats_text,
                           transform=self.static_ax.transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add bin size annotation in upper left
        self.static_ax.text(0.02, 0.98, f'Time interval: {bin_size}s', 
                           transform=self.static_ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Adjust layout to prevent title and labels from being cut off
        self.static_canvas.figure.tight_layout()
        
        self.static_canvas.draw()
        self.apply_theme()
        
        self.feed_box.append(f"Count distribution plotted. Mean counts per {bin_size}s: "
                           f"All={lambda_all:.2f}, Non-Coin={lambda_non_coin:.2f}, Coin={lambda_coin:.2f}")

    def stop_file(self):
        # First, signal the thread to stop
        self.read_serial_active = False
        self.feed_box.append("Stopping recording...")
        
        # Wait briefly for the thread to exit the loop
        import time
        time.sleep(0.2)
        
        # Now safely close the connections
        if hasattr(self, 'serial_connection') and self.serial_connection.is_open:
            self.serial_connection.close()
            self.feed_box.append("Serial connection closed.")
        
        if hasattr(self, 'data_file') and not self.data_file.closed:
            self.data_file.close()
            self.feed_box.append("Data file closed.")
        
        self.feed_box.append("Recording stopped and file closed.")

    def record_file(self):
        if not hasattr(self, 'serial_connection') or not self.serial_connection.is_open:
            self.feed_box.append("Select a valid port or connect first.")
            return
        if self.serial_connection.is_open:
            # Get current working directory
            cwd = os.getcwd()
            # Create a filename with timestamp like "CW_data_2025-07-01_15-30-00.txt"
            self.start_timestamp_dt = datetime.now()
            self.start_timestamp_str = self.start_timestamp_dt.strftime("%Y-%m-%d_%H-%M-%S")
            self.read_serial_active = True
            
            # Initialize live data storage for plotting
            self.live_data = {
                'event_number': [],
                'PICO_timestamp_s': [],
                'coincident': [],
                'adc': [],
                'sipm': [],
                'deadtime': [],
                'temperature': [],
                'pressure': [],
                'accel_x': [],
                'accel_y': [],
                'accel_z': [],
                'gyro_x': [],
                'gyro_y': [],
                'gyro_z': []
            }
            
            filename = f"CW_data_{self.start_timestamp_str}.txt"
            # Full path for the new file
            filepath = os.path.join(cwd, filename)
            # Open the file for writing and save the file handle
            self.data_file = open(filepath, "w")
            # Log to your feed_box GUI element
            self.feed_box.append(f"Created new data file: {filepath}")
            header_lines = [
        "###########################################################################################################################################################",
        "#                                                          CosmicWatch: The Desktop Muon Detector v3X",
        "#                                                                   Questions? saxani@udel.edu",
        "# Event  Timestamp[s]  Coincident[bool]  ADC[0-4095]  SiPM[mV]  Deadtime[s]  Temp[C]  Pressure[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date",
        "###########################################################################################################################################################"
    ]      
            
            for line in header_lines:
                self.data_file.write(line + "\n")
                self.data_ready.emit(line)
                self.read_serial_active = True 
                self.time_stamp = 0
            def read_serial_data():
                self.first_event_time = None
                self.last_screen_update_time = time.time()  # Track last screen update time
                self.coincidence_counter = 0
                deadtime = 0
                livetime = 0 
                self.events = 0
                self.rate = 0
                self.rate_error = 0 
                self.coincidence_rate = 0
                self.coincidence_rate_error

                while self.read_serial_active:

                    if self.serial_connection.in_waiting:
                        
                        data = self.serial_connection.readline().decode().replace('\r\n','')    # Wait and read data 
                        #print(data)
                        
                        data = data.split("\t")
                        if data[2].strip() == '1':
                            self.coincidence_counter += 1
                        
                        # Store live data for plotting
                        try:
                            if len(data) >= 10:  # Make sure we have enough fields
                                self.live_data['event_number'].append(float(data[0]))
                                self.live_data['PICO_timestamp_s'].append(float(data[1]))
                                self.live_data['coincident'].append(data[2].strip() == '1')
                                self.live_data['adc'].append(int(data[3]))
                                self.live_data['sipm'].append(float(data[4]))
                                self.live_data['deadtime'].append(float(data[5]))
                                self.live_data['temperature'].append(float(data[6]))
                                self.live_data['pressure'].append(float(data[7]))
                                
                                # Parse accelerometer data
                                accel_parts = data[8].split(':')
                                if len(accel_parts) == 3:
                                    self.live_data['accel_x'].append(float(accel_parts[0]))
                                    self.live_data['accel_y'].append(float(accel_parts[1]))
                                    self.live_data['accel_z'].append(float(accel_parts[2]))
                                
                                # Parse gyro data
                                gyro_parts = data[9].split(':')
                                if len(gyro_parts) == 3:
                                    self.live_data['gyro_x'].append(float(gyro_parts[0]))
                                    self.live_data['gyro_y'].append(float(gyro_parts[1]))
                                    self.live_data['gyro_z'].append(float(gyro_parts[2]))
                        except (ValueError, IndexError):
                            pass  # Skip malformed data
                        
                        ti = str(datetime.now()).split(" ")
                        comp_time = ti[-1]
                        data.append(comp_time)
                        #data[1] = comp_time
                        comp_date = ti[0].split('-')
                        data.append(comp_date[2] + '/' +comp_date[1] + '/' + comp_date[0]) #ti[0].replace('-','/')
                        for j in range(len(data)):
                            #print(data[j])d
                            
                            self.data_file.write(data[j]+'\t')
                            self.detector_name = data[-3]
                            if self.first_event_time is None:
                               self.first_event_time = float(data[1])
                               self.first_event = float(data[0])
                               self.first_dead_time = float(data[5])
                            self.time_stamp = float(data [1]) - self.first_event_time
                            deadtime = float(data[5]) - self.first_dead_time
                            livetime = (self.time_stamp) - deadtime
                            if livetime <= 0:
                                livetime = 1e-8
                            self.events = (float(data[0])-(self.first_event))
                            self.rate = self.events/livetime
                            self.rate_error = math.sqrt(self.rate) / livetime
                            self.coincidence_rate = self.coincidence_counter / livetime
                            self.coincidence_rate_error = math.sqrt(self.coincidence_counter) / livetime
                            hours = int(self.time_stamp // 3600)
                            minutes = int((self.time_stamp % 3600) // 60)
                            seconds = int(self.time_stamp % 60)
                            self.data_ready2.emit(
                                f"Run Time: {hours:02}:{minutes:02}:{seconds:02.0f}\n\n"
                                f"Total deadtime: {deadtime:02.3f}s\n\n"
                                f"Total Livetime: {livetime:02.2f}s\n\n"
                                f"Single Counts: {self.events}\n\n"
                                f"Single Count Rate: {self.rate:02.2f}±{self.rate_error:02.3f}Hz\n\n"
                                f"Coincidence Counts: {self.coincidence_counter}\n\n"
                                f"Coincidence Rate: {self.coincidence_rate:02.2f}±{self.coincidence_rate_error:02.3f}Hz\n\n"
)
                            self.last_screen_update_time = time.time() 
                            
    

                        self.data_file.write("\n")
                        #print(str(i+'\t') for i in data)
                        #print(*data, sep='\t')
                        event_number = int(data[0])
                        if event_number % 1 ==0:
                            self.data_file.flush()
    
                        column_widths = [8, 20, 10, 10, 10, 12, 10, 12, 25, 25, 10, 20, 12]
                        line_str = ''.join(str(item)[:width].ljust(width) for item, width in zip(data, column_widths))
                        self.data_ready.emit(line_str)
                    else:
                        if time.time() - self.last_screen_update_time >= 1.0:
                            self.time_stamp += 1.0  # increment displayed time by 1 second
                            hours = int(self.time_stamp // 3600)
                            minutes = int((self.time_stamp % 3600) // 60)
                            seconds = int(self.time_stamp % 60)
                            self.data_ready2.emit(
                                f"Run Time: {hours:02}:{minutes:02}:{seconds:02.0f}\n\n"
                                f"Total deadtime: {deadtime:02.3f}\n\n"
                                f"Total Livetime: {livetime:02.2f}\n\n"
                                f"Single Counts: {self.events}\n\n"
                                f"Single Count Rate: {self.rate:02.2f}±{self.rate_error:02.3f}Hz\n\n"
                                f"Coincedence Counts: {self.coincidence_counter}\n\n"
                                f"Coincedence Rate: {self.coincidence_rate:02.2f}±{self.coincidence_rate_error:02.3f}Hz\n\n"
                                
)
                            self.last_screen_update_time = time.time()
                            time.sleep(0.1)
                


                
        # Create and start the thread
        self.read_thread = threading.Thread(target=read_serial_data, daemon=True)
        self.read_thread.start()
         
        
        

# If the input is just a file name (no path separators), prepend cwd



            

if __name__ == '__main__':
    # Enable keyboard interrupt (Ctrl+C) to close the application
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    dashboard = FuturisticDashboard()
    dashboard.show()
    
    # Handle keyboard interrupt gracefully
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Closing application...")
        dashboard.close()
        sys.exit(0)