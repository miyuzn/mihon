import socket
import signal
import sys
import sensor
import pickle
import pandas as pd
import math
from datetime import datetime

# ------------ Configuration ------------
# Raw sensor UDP ports
PORT_L = 13250
PORT_R = 13251
# Decimation target FPS (sensors send at 100Hz)
target_fps = 50
# Experiment date folder
exp_date = "0707"
exp_dir = f"./exp/{exp_date}/"
# Calibration CSV paths
LEFT_CALIB_FILE = "./cali/0814/25l.csv"
RIGHT_CALIB_FILE = "./cali/0814/25r.csv"
# GUI receiver
SEND_IP = "127.0.0.1"
SEND_PORT = 54000

# ------------ Global Buffers ------------
# Raw parsed sensor data
raw_data_list_l = []
raw_data_list_r = []
# Calibrated & decimated records
data_records_l = []
data_records_r = []

# ------------ Utility Functions ------------
def load_fitting_parameters(file_path):
    df = pd.read_csv(file_path)
    return {row['Sensor']: (row['k'], row['alpha']) for _, row in df.iterrows()}


def calibrate_pressure(raw_values, params):
    calibrated = []
    for i, raw in enumerate(raw_values):
        sensor_id = i + 1
        if sensor_id in params:
            k, alpha = params[sensor_id]
            v = raw / 1000.0
            if v <= 0.312:
                pressure = 0
            else:
                R = 5000 * 0.312 / (v - 0.312)
                if not math.isfinite(R):
                    pressure = 0
                else:
                    pressure = (R / k) ** (1 / alpha)
                    if pressure < 0.01 or pressure > 50:
                        pressure = 0
            calibrated.append(pressure)
        else:
            calibrated.append(0)
    return calibrated


def save_and_exit(sig, frame):
    # Timestamp for filenames
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw data via sensor module
    raw_file_l = f"{exp_dir}{now}_left.csv"
    raw_file_r = f"{exp_dir}{now}_right.csv"
    print(f"\nSaving raw data to {raw_file_l}, {raw_file_r}...")
    sensor.save_sensor_data_to_csv(raw_data_list_l, raw_file_l)
    sensor.save_sensor_data_to_csv(raw_data_list_r, raw_file_r)

    # Save calibrated data
    calib_file_l = f"{exp_dir}{now}_left_f.csv"
    calib_file_r = f"{exp_dir}{now}_right_f.csv"
    pd.DataFrame(data_records_l).to_csv(calib_file_l, index=False)
    pd.DataFrame(data_records_r).to_csv(calib_file_r, index=False)
    print(f"Saved calibrated data to {calib_file_l}, {calib_file_r}")

    sys.exit(0)


# ------------ Main Routine ------------
if __name__ == "__main__":
    # Ensure output directory exists
    import os
    os.makedirs(exp_dir, exist_ok=True)

    # Bind SIGINT (Ctrl+C) to save_and_exit
    signal.signal(signal.SIGINT, save_and_exit)

    # Load calibration parameters
    params_left = load_fitting_parameters(LEFT_CALIB_FILE)
    params_right = load_fitting_parameters(RIGHT_CALIB_FILE)

    # Setup raw data receivers
    sock_l = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_l.bind(("", PORT_L))
    sock_r.bind(("", PORT_R))

    # Setup sender to GUI
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    counter = 0
    print("Receiving sensor data (100Hz). Press Ctrl+C to stop and save.")

    while True:
        # Receive raw bytes
        data_l_bytes, addr_l = sock_l.recvfrom(1024)
        data_r_bytes, addr_r = sock_r.recvfrom(1024)

        # Parse and store raw data for saving
        sd_l = sensor.parse_sensor_data(data_l_bytes)
        sd_r = sensor.parse_sensor_data(data_r_bytes)
        raw_data_list_l.append(sd_l)
        raw_data_list_r.append(sd_r)

        counter += 1
        if counter % 100 == 0:
            print(f"\rReceived {counter} packets...", end='')

        # Decimate to target FPS
        if counter % int(100 / target_fps) == 0:
            # Calibrate pressure
            calib_l = calibrate_pressure(sd_l.pressure_sensors, params_left)
            calib_r = calibrate_pressure(sd_r.pressure_sensors, params_right)

            # Extract accel & gyro
            acc_l = sd_l.accelerometer
            gyro_l = sd_l.gyroscope
            acc_r = sd_r.accelerometer
            gyro_r = sd_r.gyroscope

            # Build packet dictionaries
            rec_l = {'timestamp': sd_l.timestamp}
            rec_r = {'timestamp': sd_r.timestamp}
            for idx, val in enumerate(calib_l): rec_l[f"P{idx+1}"] = val
            for idx, val in enumerate(calib_r): rec_r[f"P{idx+1}"] = val
            rec_l.update({'ax': acc_l[0], 'ay': acc_l[1], 'az': acc_l[2]})
            rec_r.update({'ax': acc_r[0], 'ay': acc_r[1], 'az': acc_r[2]})
            rec_l.update({'gx': gyro_l[0], 'gy': gyro_l[1], 'gz': gyro_l[2]})
            rec_r.update({'gx': gyro_r[0], 'gy': gyro_r[1], 'gz': gyro_r[2]})

            # Send to GUI listener
            try:
                message = pickle.dumps((rec_l, rec_r))
                send_sock.sendto(message, (SEND_IP, SEND_PORT))
            except Exception as e:
                print(f"Error sending UDP packet: {e}", file=sys.stderr)

            # Store calibrated record for saving
            data_records_l.append(rec_l)
            data_records_r.append(rec_r)
