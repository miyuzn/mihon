"""
demo_gui_v3.py
=================

Realtime UDP heat-map viewer for the G-CU.

This program provides a GUI application to visualise sensor data from the G-CU
as a real-time heat-map, with recording and playback functionality.

Main features
-------------
* **Realtime visualisation** – 5×5 heat-map updated via UDP packets (default port 31415).
* **Colour-map control** – adjustable min/max range with numeric spinboxes.
* **Recording** – save incoming data into CSV format (time, device ID, sensors, IMU).
* **Playback** – load CSV logs, scrub with a slider, and play/pause at adjustable speed.
* **Offset-correction option** – averages the first *N* frames after Start and subtracts
  them as baseline. Controlled by a new *Offset* toggle next to *Record*.  
  • Keeps the same offset across repeated measurements while ON.  
  • Turning it OFF clears the baseline and re-learns next time it is enabled.  
  • Clear status messages indicate “learning offset” progress and “baseline applied”.
* **UDP port re-bind** – a spinbox and *Rebind* button allow changing the UDP port
  at runtime without restarting the program. The socket is closed/recreated safely.
* **Status display** – clear textual updates for receiving, recording, playback,
  offset learning, and errors.

Other details
-------------
* Sensor values and 9-axis IMU data are decoded from G-CU packets.
* CSV logs are automatically organised by date/session folders.
* Safe exit: sockets and files are closed when the window is closed.

"""

from __future__ import annotations

import csv
import os
import re
import signal
import socket
import struct
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# ---------------------------- Protocol constants ---------------------------- #
START_MARKER: int = 0x5A
END_MARKER: int = 0xA5
UDP_PORT: int = 13250  # default port number (円周率3.1415 -> 31415)

# --------------------------- Visualisation settings ------------------------- #
ROWS: int = 7
COLS: int = 5
POLL_INTERVAL_MS: int = 10
UI_REFRESH_MS: int = 40
IMU_KEYS = [
    "magn_x",
    "magn_y",
    "magn_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
]
_S_RE = re.compile(r"^S_(\d+)$")

# --------------------------- Offset‑correction ------------------------------ #
OFFSET_SAMPLES: int = 100  # number of frames to average for the offset
ALPHA = 0.003  # 0.001〜0.01で調整
K_TAU = 3.0  # デッドバンドの幅 = K_TAU*MAD

# --------------------------------------------------------------------------- #
#                               Data handling                                 #
# --------------------------------------------------------------------------- #


def decode_gcu_packet(
        packet: bytes) -> Optional[Tuple[int, str, Dict[str, float], Dict[str, float]]]:
    """Decode a G‑CU packet and extract its embedded timestamp."""
    if len(packet) < 10 or packet[0] != START_MARKER or packet[1] != START_MARKER:
        return None
    device_id = packet[2]
    secs = struct.unpack_from('<I', packet, 4)[0]
    ms = struct.unpack_from('<H', packet, 8)[0]
    pkt_time = datetime.fromtimestamp(secs + ms / 1000.0).strftime('%H:%M:%S.%f')
    sensor_cnt = packet[3]

    pos = 10
    sensor_values: Dict[str, float] = {}
    for i in range(sensor_cnt):
        if pos + 4 > len(packet):
            break
        sensor_values[f'S_{i + 1}'] = struct.unpack_from('<I', packet, pos)[0]
        pos += 4

    imu_vals: Dict[str, float] = {}
    if pos + 36 <= len(packet):
        imu_tuple = struct.unpack_from('<9f', packet, pos)
        for k, v in zip(IMU_KEYS, imu_tuple):
            imu_vals[k] = float(v)
    # pos += 36  # 今後の拡張用

    if not sensor_values:
        return None

    return (device_id, pkt_time, sensor_values, imu_vals)


# --------------------------------------------------------------------------- #
#                                    GUI                                      #
# --------------------------------------------------------------------------- #
class HeatmapWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('G‑CU Pressure Heat‑Map')

        # ---------------------------- Graphics ----------------------------- #
        self._glw = pg.GraphicsLayoutWidget()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self._glw.setBackground('w')
        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setLabel('left', 'Row')
        self._plot.setLabel('bottom', 'Col')
        self._image = pg.ImageItem()
        self._plot.addItem(self._image)
        vb = self._plot.getViewBox()
        vb.invertX(True)
        vb.invertY(True)
        cmap = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0]),
            color=[(0, 0, 255), (0, 255, 0), (255, 0, 0)],
        )
        self._image.setLookupTable(cmap.getLookupTable())
        self._cbar = pg.ColorBarItem(colorMap=cmap, values=(0, 1), interactive=False)
        self._glw.addItem(self._cbar, row=0, col=1)
        self._cbar.setImageItem(self._image)
        self._recent_z = []  # type: list[np.ndarray]
        self._recent_cap = int(2000 // UI_REFRESH_MS)

        # ---------------------------- Controls ----------------------------- #
        self._start_btn = QPushButton('Start')
        self._stop_btn = QPushButton('Stop')
        self._start_btn.clicked.connect(self._start_receiving)
        self._stop_btn.clicked.connect(self._stop_receiving)
        self._stop_btn.setEnabled(False)

        self._record_cb = QCheckBox('Record')
        self._record_cb.setToolTip('Save data on Start')
        self._record_cb.stateChanged.connect(self._toggle_recording)
        self._recording = False
        self._log_file: Optional[csv.writer] = None
        self._log_handle: Optional[object] = None

        # Offset‑correction toggle
        self._offset_cb = QCheckBox('Offset')
        self._offset_cb.setToolTip('Average the first %d frames and subtract as baseline' %
                                   OFFSET_SAMPLES)
        self._offset_cb.stateChanged.connect(self._toggle_offset)
        self._offset_enabled = False  # user intent – toggle state
        self._offset_ready = False  # True once baseline has been learnt
        self._offset_baseline: Optional[np.ndarray] = None
        self._offset_mad: Optional[np.ndarray] = None
        self._offset_accum: list[np.ndarray] = []

        # Color‑bar range controls (Enter to apply)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setPrefix('Min: ')
        self._min_spin.setDecimals(3)
        self._min_spin.setRange(0.0, 1e6)
        self._min_spin.setValue(300.0)
        self._min_spin.editingFinished.connect(self._update_levels)

        self._max_spin = QDoubleSpinBox()
        self._max_spin.setPrefix('Max: ')
        self._max_spin.setDecimals(3)
        self._max_spin.setRange(0.0, 1e6)
        self._max_spin.setValue(2000.0)
        self._max_spin.editingFinished.connect(self._update_levels)

        self._prev_min = 0.0
        self._prev_max = 2000.0

        # -------------------------- Playback ------------------------------ #
        self._load_btn = QPushButton('Load')
        self._play_btn = QPushButton('Play')
        self._unload_btn = QPushButton('Unload')
        self._play_btn.setEnabled(False)
        self._unload_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load_log)
        self._play_btn.clicked.connect(self._toggle_playback)
        self._unload_btn.clicked.connect(self._unload_playback)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        self._playback_data: list[np.ndarray] = []
        self._playback_idx = 0
        self._in_playback = False

        # -------------------------- Status -------------------------------- #
        self._status = QLabel('Waiting for packets…')

        # -------------------------- Port Setting -------------------------------- #
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(UDP_PORT)  # 既定は今まで通り
        self._rebind_btn = QPushButton('Rebind')
        self._rebind_btn.setToolTip('Rebind UDP socket to the specified port')
        self._rebind_btn.clicked.connect(self._on_rebind_clicked)

        # --------------------------- Layout -------------------------------- #
        port_row = QHBoxLayout()
        port_row.addStretch()
        port_row.addWidget(QLabel("Port:"))
        port_row.addWidget(self._port_spin)
        port_row.addWidget(self._rebind_btn)

        live_row = QHBoxLayout()
        live_row.addWidget(self._start_btn)
        live_row.addWidget(self._stop_btn)
        live_row.addWidget(self._record_cb)
        live_row.addWidget(self._offset_cb)  # new toggle placed next to Record
        live_row.addWidget(self._min_spin)
        live_row.addWidget(self._max_spin)
        # live_row.addWidget(QLabel('Port:'))
        # live_row.addWidget(self._port_spin)
        # live_row.addWidget(self._rebind_btn)

        pb_row = QHBoxLayout()
        pb_row.addWidget(self._load_btn)
        pb_row.addWidget(self._play_btn)
        pb_row.addWidget(self._unload_btn)
        pb_row.addWidget(self._slider)

        main = QVBoxLayout()
        main.addWidget(self._glw)
        main.insertLayout(0, port_row)
        main.addLayout(live_row)
        main.addLayout(pb_row)
        main.addWidget(self._status)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        # ---------------------- Runtime data ------------------------------ #
        self._data = np.zeros((ROWS, COLS))
        self._levels = (self._prev_min, self._prev_max)
        self._receiving = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(('0.0.0.0', UDP_PORT))
        self._sock.setblocking(False)

        # Timers
        self._poll = QTimer()
        self._poll.timeout.connect(self._poll_udp)
        self._refresh = QTimer()
        self._refresh.timeout.connect(self._refresh_image)
        self._refresh.start(UI_REFRESH_MS)

        # ---------------------- Logging data ------------------------------ #
        self._writers = {}
        self._session_dir = None
        self._session_start = None

    # ------------------------------------------------------------------ #
    #                           Toggle handlers                          #
    # ------------------------------------------------------------------ #
    def _toggle_recording(self, state: int) -> None:
        self._recording = bool(state)
        msg = 'Recording enabled' if self._recording else 'Recording disabled'
        self._status.setText(msg)

    def _toggle_offset(self, state: int) -> None:
        self._offset_enabled = bool(state)
        if self._offset_enabled:
            # (Re)start learning phase
            self._offset_ready = False
            self._offset_accum.clear()
            self._status.setText('Offset: will learn baseline at Start')
        else:
            self._offset_ready = False
            self._offset_baseline = None
            self._offset_mad = None
            self._status.setText('Offset: disabled')

    # ------------------------------------------------------------------ #
    #                     Setting Prot Number                            #
    # ------------------------------------------------------------------ #

    def _rebind_socket(self, port: int) -> bool:
        """Recreate UDP socket and bind to a new port at runtime."""
        was_receiving = self._receiving
        try:
            # 受信中なら一時停止
            if was_receiving:
                self._receiving = False
                self._poll.stop()
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)

            # 旧ソケットを閉じ、新規作成して再bind
            try:
                self._sock.close()
            except Exception:
                pass

            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setblocking(False)
            self._sock.bind(('0.0.0.0', port))

            self._status.setText(f'UDP rebind OK: port={port}')

            # 受信を再開
            if was_receiving and self._playback_data == []:
                self._receiving = True
                self._poll.start(POLL_INTERVAL_MS)
                self._start_btn.setEnabled(False)
                self._stop_btn.setEnabled(True)

            return True

        except OSError as e:
            QMessageBox.critical(self, 'Bind error', f'Failed to bind UDP port {port}\n{e}')
            self._status.setText('UDP rebind failed')
            return False

    def _on_rebind_clicked(self) -> None:
        port = int(self._port_spin.value())
        self._rebind_socket(port)

    # ------------------------------------------------------------------ #
    #                           Range control                            #
    # ------------------------------------------------------------------ #

    def _update_levels(self) -> None:
        vmin = self._min_spin.value()
        vmax = self._max_spin.value()
        if vmin < 0 or vmax <= vmin:
            QMessageBox.warning(self, 'Invalid Range', 'Min must be ≥ 0 and less than Max')
            self._min_spin.blockSignals(True)
            self._max_spin.blockSignals(True)
            self._min_spin.setValue(self._prev_min)
            self._max_spin.setValue(self._prev_max)
            self._min_spin.blockSignals(False)
            self._max_spin.blockSignals(False)
            return
        self._levels = (vmin, vmax)
        self._image.setLevels(self._levels)
        self._cbar.setLevels(self._levels)
        self._prev_min = vmin
        self._prev_max = vmax

    # ------------------------------------------------------------------ #
    #                       Measurement control                          #
    # ------------------------------------------------------------------ #
    def _start_receiving(self) -> None:
        if self._playback_data:
            return  # ignore if in playback mode
        if not self._receiving:
            if self._recording:
                self._prepare_log()
            if self._offset_enabled and not self._offset_ready:
                self._status.setText('Learning offset…')
            else:
                self._status.setText('Receiving…')
            self._receiving = True
            self._poll.start(POLL_INTERVAL_MS)
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)

    def _stop_receiving(self) -> None:
        if self._receiving:
            self._receiving = False
            self._poll.stop()
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
                self._log_file = None
            self._status.setText('Stopped.')

    # ------------------------------------------------------------------ #
    #                            Logging                                 #
    # ------------------------------------------------------------------ #
    def _prepare_log(self) -> None:
        if self._log_handle:
            self._log_handle.close()
        root = os.path.join(os.getcwd(), 'log')
        date_dir = datetime.now().strftime('%Y%m%d')
        session_dir = os.path.join(root, date_dir)
        os.makedirs(session_dir, exist_ok=True)
        fname = datetime.now().strftime('%H%M%S') + '.csv'
        path = os.path.join(session_dir, fname)
        self._log_handle = open(path, 'w', newline='')
        writer = csv.writer(self._log_handle)
        # 先頭: 時刻, デバイスID
        header = ['device_id', 'time']
        # センサ値: S_1..S_{ROWS*COLS}
        header += [f'S_{i+1}' for i in range(ROWS * COLS)]
        # IMU 9軸
        header += IMU_KEYS
        writer.writerow(header)
        self._log_file = writer

    def _open_session_dir(self):
        root = os.path.join(os.getcwd(), 'log')
        date_dir = datetime.now().strftime("%f%m%d")
        self._session_dir = os.path.join(root, date_dir)
        os.makedirs(self._session_dir, exist_ok=True)
        self._session_start = datetime.now().strftime("%H%M%S_%f")

    def _ensure_writer(self, dev_id: int):
        if dev_id in self._writers:
            return self._writers[dev_id][0]
        if self._session_dir is None:
            self._open_session_dir()
        fname = f"{self._session_start}_dev{dev_id:02d}.csv"
        path = os.path.join(self._session_dir, fname)
        fh = open(path, 'w', newline='')
        w = csv.writer(fh)
        header = ['device_id', 'time'] + [f"S_{i+1}" for i in range(ROWS * COLS)] + IMU_KEYS
        w.writerow(header)
        self._writers[dev_id] = (w, fh)
        return w

    def _close_all_writers(self):
        for _, fh in self._writers.values():
            try:
                fh.close()
            except Exception:
                pass
        self._writers.clear()

    # ------------------------------------------------------------------ #
    #                             UDP poll                               #
    # ------------------------------------------------------------------ #
    def _poll_udp(self) -> None:
        if getattr(self, "_in_playback", False):
            return
        while True:
            try:
                packet, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            decoded = decode_gcu_packet(packet)
            if not decoded:
                continue
            dev_id, ts, sensor_vals, imu_vals = decoded
            flat = np.zeros(ROWS * COLS)
            per = len(flat) // len(sensor_vals)
            for idx, v in enumerate(sensor_vals.values()):
                flat[idx * per:(idx + 1) * per] = v

            # ---------------- Offset correction ------------------- #
            if self._offset_enabled:
                if not self._offset_ready:
                    # accumulate baseline frames
                    self._offset_accum.append(flat.copy())
                    if len(self._offset_accum) >= OFFSET_SAMPLES:
                        stack = np.stack(self._offset_accum, axis=0)  # (N, P)
                        # robust baseline: median to reduce outliers
                        self._offset_baseline = np.median(stack, axis=0)
                        mad = np.median(np.abs(stack - self._offset_baseline), axis=0)
                        self._offset_mad = np.maximum(mad, 1e-9)  # ゼロ割り防止
                        self._offset_ready = True
                        self._status.setText(
                            f'Offset: baseline from {OFFSET_SAMPLES} frames applied')
                    else:
                        # show progress – every 10 %
                        pct = int(100 * len(self._offset_accum) / OFFSET_SAMPLES)
                        self._status.setText(f'Learning offset… {pct}%')
                if self._offset_ready and self._offset_baseline is not None:
                    flat = flat - self._offset_baseline
                    # 共通モード除去（空間中央値）
                    flat = flat - np.median(flat)
                    # 条件付きEMAで baseline 更新（イベント画素は凍結）
                    if getattr(self, "_receiving", False) and self._offset_enabled:
                        assert self._offset_mad is not None
                        tau = K_TAU * (1.4826 * self._offset_mad)
                        mask_quiet = np.abs(flat) <= tau  # “静か”な画素だけ
                        # b ← b + α*(観測−b)  ただしここでの“観測”は共通モード除去前の値を使うなら別途保持
                        self._offset_baseline[mask_quiet] = (
                            1.0 - ALPHA) * self._offset_baseline[mask_quiet] + ALPHA * (
                                flat[mask_quiet] + np.median(flat))  # 参照の置き方は運用に合わせて

                # 表示のロバストクリップ（任意・簡易）
                if self._offset_mad is not None:
                    SIG = 1.4826 * self._offset_mad
                else:
                    # フォールバック: 現フレームの空間MAD（スカラー）を使う
                    spatial_med = np.median(flat)
                    spatial_mad = np.median(np.abs(flat - spatial_med))
                    SIG = np.maximum(1.4826 * spatial_mad, 1e-9)

                # ロバストな標準偏差みたいな指標
                flat = np.clip(flat / SIG, -5.0, 5.0)

                self._recent_z.append(flat.copy())
                if len(self._recent_z) > self._recent_cap:
                    self._recent_z.pop(0)
            # ------------------------------------------------------- #

            self._data = flat.reshape(ROWS, COLS)
            if self._recording and self._log_file:
                # センサー列（S_1..S_n）を順序固定で並べる
                sensor_keys = [f"S_{i+1}" for i in range(len(sensor_vals))]
                sensor_row = [sensor_vals.get(k, "") for k in sensor_keys]

                # IMU列（存在すれば）を固定順で並べる
                imu_keys = IMU_KEYS  # 例: ["magn_x","magn_y","magn_z","gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"]
                imu_row = [imu_vals.get(k, "") for k in imu_keys]

                # 1行を書き出し（id, 時刻, センサー, IMU）
                self._log_file.writerow([dev_id, ts] + sensor_row + imu_row)
            if not self._offset_enabled or self._offset_ready:
                # normal status during measurement
                self._status.setText(f'[{ts}] sensors: {len(sensor_vals)}')

            if getattr(self, "_in_playback", False):
                return


    # ------------------------------------------------------------------ #
    #                         Image refresh                              #
    # ------------------------------------------------------------------ #
    def _refresh_image(self) -> None:
        if getattr(self, "_in_playback", False) and self._playback_data:
            frame = self._playback_data[self._playback_idx]
            self._image.setImage(frame, levels=self._levels)
        else:
            # ライブ描画
            if self._offset_enabled and self._recent_z:
                zstack = np.concatenate(self._recent_z)  # 1DでOK
                vmin = float(np.quantile(zstack, 0.01))
                vmax = float(np.quantile(zstack, 0.99))
                if vmin >= vmax:  # 保険
                    vmin, vmax = -5.0, 5.0
                self._image.setImage(self._data, levels=(vmin, vmax))
                self._cbar.setLevels((vmin, vmax))
            else:
                self._image.setImage(self._data, levels=self._levels)
                self._cbar.setLevels(self._levels)

    # ------------------------------------------------------------------ #
    #                       Playback handlers                            #
    # ------------------------------------------------------------------ #
    def _load_log(self) -> None:
        if self._receiving:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select log',
            os.path.join(os.getcwd(), 'log'),
            'CSV (*.csv)',
        )
        if not path:
            return

        frames = []
        times = []
        need = ROWS * COLS

        try:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                rdr = csv.reader(f)
                first = next(rdr, None)
                if first is None:
                    return

                is_header = ("time" in first) or any(_S_RE.match(x or "") for x in first)

                if is_header:
                    header = first
                    name_to_idx = {name: i for i, name in enumerate(header)}

                    sensor_idx = []
                    for i, name in enumerate(header):
                        m = _S_RE.match(name or "")
                        if m:
                            sensor_idx.append((int(m.group(1)), i))
                    sensor_idx.sort(key=lambda t: t[0])
                    sensor_idx = [i for _, i in sensor_idx]

                    time_idx = name_to_idx.get("time")

                    if sensor_idx:
                        for row in rdr:
                            try:
                                sensor_vals = [float(row[i]) for i in sensor_idx if i < len(row)]
                            except (ValueError, IndexError):
                                continue
                            if not sensor_vals:
                                continue

                            flat = np.zeros(need, dtype=float)
                            per = max(1, need // len(sensor_vals))
                            for idx, v in enumerate(sensor_vals):
                                b = idx * per
                                e = min((idx + 1) * per, need)
                                if b >= need:
                                    break
                                flat[b:e] = v
                            frames.append(flat.reshape(ROWS, COLS))

                            t = row[time_idx] if (time_idx is not None and
                                                  time_idx < len(row)) else ""
                            times.append(t)

                    else:
                        for row in rdr:
                            if len(row) < need:
                                continue
                            vals = row[-need:]
                            try:
                                arr = np.array([float(x) for x in vals],
                                               dtype=float).reshape(ROWS, COLS)
                            except ValueError:
                                continue
                            frames.append(arr)
                            t = row[time_idx] if (time_idx is not None and
                                                  time_idx < len(row)) else ""
                            times.append(t)

                else:
                    row = first
                    if len(row) >= need:
                        try:
                            arr = np.array([float(x) for x in row[-need:]],
                                           dtype=float).reshape(ROWS, COLS)
                            frames.append(arr)
                            times.append("")
                        except ValueError:
                            pass
                    for row in rdr:
                        if len(row) < need:
                            continue
                        try:
                            arr = np.array([float(x) for x in row[-need:]],
                                           dtype=float).reshape(ROWS, COLS)
                        except ValueError:
                            continue
                        frames.append(arr)
                        times.append("")

        except OSError as e:
            QMessageBox.critical(self, "Open error", f"Failed to open CSV\n{e}")
            return

        if not frames:
            return

        self._playback_data.clear()
        self._playback_data = frames
        self._playback_times = times

        self._slider.setMaximum(len(self._playback_data) - 1)
        self._slider.setEnabled(True)
        self._play_btn.setEnabled(True)
        self._unload_btn.setEnabled(True)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._record_cb.setEnabled(True if not self._recording else False)
        self._offset_cb.setEnabled(False)

        self._playback_idx = 0
        self._slider.blockSignals(True)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_playback_frame()
        self._in_playback = True

    def _toggle_playback(self) -> None:
        if not hasattr(self, '_pb_timer'):
            self._pb_timer = QTimer()
            self._pb_timer.timeout.connect(self._next_frame)
        if not self._pb_timer.isActive():
            self._pb_timer.start(UI_REFRESH_MS)
            self._play_btn.setText('Pause')
        else:
            self._pb_timer.stop()
            self._play_btn.setText('Play')

            if getattr(self, "_in_playback", False) and self._playback_data:
                self._update_playback_frame()

    def _next_frame(self) -> None:
        self._playback_idx = (self._playback_idx + 1) % len(self._playback_data)
        self._slider.setValue(self._playback_idx)

    def _on_slider(self, idx: int) -> None:
        if not self._playback_data:
            return
        self._playback_idx = idx
        self._update_playback_frame()

    def _update_playback_frame(self) -> None:
        frame = self._playback_data[self._playback_idx]
        self._image.setImage(frame, levels=self._levels)
        self._status.setText(f'Playback {self._playback_idx + 1}/{len(self._playback_data)}')

        t = ""
        if hasattr(self, "_playback_times") and self._playback_times and self._playback_idx < len(
                self._playback_times):
            t = self._playback_times[self._playback_idx] or ""
        if t:
            self._status.setText(
                f'Playback {self._playback_idx + 1}/{len(self._playback_data)} | {t}')
        else:
            self._status.setText(f'Playback {self._playback_idx + 1}/{len(self._playback_data)}')

    def _unload_playback(self) -> None:
        if hasattr(self, '_pb_timer') and getattr(self, '_pb_timer').isActive():
            self._pb_timer.stop()
            self._play_btn.setText('Play')
        self._slider.blockSignals(True)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._playback_data.clear()
        if hasattr(self, "_playback_times"):
            self._playback_times.clear()
        self._slider.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._unload_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._record_cb.setEnabled(True)
        self._offset_cb.setEnabled(True)
        self._status.setText('Playback ended')
        self._in_playback = False

    # ------------------------------------------------------------------ #
    #                              Exit                                  #
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._log_handle:
            self._log_handle.close()
        if hasattr(self, '_pb_timer') and getattr(self, '_pb_timer'):
            self._pb_timer.stop()
        self._sock.close()
        return super().closeEvent(event)


# --------------------------------------------------------------------------- #
#                               Top‑level                                     #
# --------------------------------------------------------------------------- #


def _install_sigint_handler(app: QApplication) -> None:
    signal.signal(signal.SIGINT, lambda _sig, _frm: app.quit())


def main() -> None:
    app = QApplication(sys.argv)
    _install_sigint_handler(app)
    w = HeatmapWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
