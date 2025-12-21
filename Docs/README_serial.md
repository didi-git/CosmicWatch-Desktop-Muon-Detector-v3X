# Using VS Code Serial Monitor with CosmicWatch

This guide explains how to use the VS Code "Serial Monitor" extension to observe the CosmicWatch Desktop Muon Detector's boot-up sequence and data logging output.

## 1. Install the Serial Monitor Extension
- Open VS Code.
- Go to the Extensions view (Ctrl+Shift+X or Cmd+Shift+X).
- Search for **Serial Monitor** (by Microsoft) and install it.

## 2. Connect the Detector
- Plug your CosmicWatch detector into a USB port on your computer.
- Wait a few seconds for the device to be recognized.

## 3. Select the Serial Port
- Open the Serial Monitor panel in VS Code (look for the plug icon in the Activity Bar or use the Command Palette: `Serial Monitor: Start Monitoring`).
- In the port selection dropdown, choose the port that looks similar to:
  
  `/dev/tty.usbmodem2101 - Raspberry Pi`
  
  (The exact name may vary, but it will start with `/dev/tty.usbmodem` on macOS.)

## 4. Start Monitoring
- Click **Start Monitor**.
- You should see the detector's boot-up sequence and data output in the Serial Monitor window.
- To catch the whole boot sequence: 
   + Hold the detector reset button down
   + As soon as the `/dev/tty.usbmodem*` port appears in the `Serial Monitor: Port` list select it
   + Hit `Serial Monitor: Start Monitoring` right after that

## 5. Example Output
```
# Welcome to the CosmicWatchDAQ (Version: v3X.26)
# (2451ms) Detecting physical microSD card ... found.
# (2452ms) Initializing microSD card communication ... complete.
# (2490ms) Initializing microSD card file ... complete.
# (2557ms) MicroSD card file name: AxLab_M_034.txt
# (2563ms) Initializing microSD card config.txt ... overwriting config.txt with default values ...complete.
# (2597ms) Initiallizing Trigger pins ... complete.
# (2645ms) Linear Fit to PWM treshold: y = 0.473471 * x + -13.813021
# (2645ms) Initializing PWM to set threshold ... complete.
# (2645ms) Measured HV to SiPM = 28.688 V
# (2645ms) Overvoltage (Measured, Expected) = 4.188 V, 5.697 V
# (2645ms) Updating splash screen info ... complete.
# (4001ms) Checking detector name ... complete.
# (4001ms) Total space: 3492 kB, Used: 2068 kB, Free: 1424 kB
# (4002ms) Trigger Threshold (Measured, Expected) : (70.87 mV, 80.00 mV) ... passed.
# (4012ms) Checking: HV to SiPM voltage: 28.71 V ...passed.
# (4022ms) Checking: Signal line voltage: 18.62 mV
# (4023ms) No coincidence detector found during bootup. 
# (4026ms) Measuring ADC0 baseline ... 40.54 LSB ± 2.18 LSB ... within expected range.
# (4026ms) Launching CosmicWatch v3X.26 Detector! 
################################################################################################################################################
#                                               CosmicWatch: The Desktop Muon Detector v3X.26
#                                                     Questions? spencer.axani@gmail.com
#                                                             Detector Name: AxLab                                                               
# Event  Timestamp[s]  Flag  ADC[12b]  SiPM[mV]  Deadtime[s]  Temp[C]  Press[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name
################################################################################################################################################
1       0.313744        0       848     43.7    0.000069        24.9    101394  -0.003:-0.006:-1.005    0.1:-0.1:0.1    AxLab
2       0.623214        0       352     18.1    0.000863        24.9    101394  -0.003:-0.006:-0.995    -0.1:0.0:0.0    AxLab
```

## Notes
- If you do not see any output, double-check the port selection and ensure the detector is powered on.
- The output includes both the boot-up sequence and real-time event data.
- You can copy and save the output for later analysis.
