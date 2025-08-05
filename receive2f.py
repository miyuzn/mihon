import socket
import pickle
import sensor
import pandas as pd
import math
from datetime import datetime
import signal
import sys

# 参数配置
exp_date = "0707"
target_fps = 50

# UDP 接收设置
RECV_IP = "127.0.0.1"
RECV_PORT = 53000

# UDP 发送设置（可供其他程序订阅）
SEND_IP = "127.0.0.1"   # 目标程序所在 IP
SEND_PORT = 54000       # 目标程序监听的 UDP 端口

# 全局保存校正后数据
data_records_l = []
data_records_r = []

# 创建发送 socket
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    left_filename = f"./exp/{exp_date}/{now}_f_left.csv"
    right_filename = f"./exp/{exp_date}/{now}_f_right.csv"
    pd.DataFrame(data_records_l).to_csv(left_filename, index=False)
    pd.DataFrame(data_records_r).to_csv(right_filename, index=False)
    print(f"Saved calibrated data to {left_filename} and {right_filename}")
    sys.exit(0)

def main():
    global params_left, params_right
    params_left  = load_fitting_parameters(f"./cali/left.csv")
    params_right = load_fitting_parameters(f"./cali/right.csv")

    # 接收 socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RECV_IP, RECV_PORT))

    # 捕捉 Ctrl+C
    signal.signal(signal.SIGINT, save_and_exit)

    counter = 0
    print("Receiving and calibrating sensor data, press Ctrl+C to save.")
    while True:
        packet, _ = sock.recvfrom(4096)
        data_l, data_r = pickle.loads(packet)
        sd_l = sensor.parse_sensor_data(data_l)
        sd_r = sensor.parse_sensor_data(data_r)

        counter += 1
        if counter % 100 == 0:
            print(f"\rReceived {counter} packets...", end='')

        # 降采样到 target_fps
        if counter % (100 / target_fps) == 0:
            # 压力校正
            calib_l = calibrate_pressure(sd_l.pressure_sensors, params_left)
            calib_r = calibrate_pressure(sd_r.pressure_sensors, params_right)

            # 提取加速度和陀螺仪
            acc_l  = sd_l.accelerometer  # [ax, ay, az]
            gyro_l = sd_l.gyroscope      # [gx, gy, gz]
            acc_r  = sd_r.accelerometer
            gyro_r = sd_r.gyroscope

            # 组装要发送的字典
            rec_l = {'timestamp': sd_l.timestamp}
            rec_r = {'timestamp': sd_r.timestamp}

            # 填充 P1…Pn
            for idx, val in enumerate(calib_l):
                rec_l[f"P{idx+1}"] = val
            for idx, val in enumerate(calib_r):
                rec_r[f"P{idx+1}"] = val

            # 填充加速度
            rec_l['ax'], rec_l['ay'], rec_l['az'] = acc_l
            rec_r['ax'], rec_r['ay'], rec_r['az'] = acc_r

            # 填充陀螺仪
            rec_l['gx'], rec_l['gy'], rec_l['gz'] = gyro_l
            rec_r['gx'], rec_r['gy'], rec_r['gz'] = gyro_r

            # 打包并发送
            try:
                message = pickle.dumps((rec_l, rec_r))
                send_sock.sendto(message, (SEND_IP, SEND_PORT))
            except Exception as e:
                print(f"Error sending UDP packet: {e}", file=sys.stderr)

            # 本地保存，等待 Ctrl+C 写 CSV
            data_records_l.append(rec_l)
            data_records_r.append(rec_r)

if __name__ == "__main__":
    main()
