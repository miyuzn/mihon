#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import math
import json
import time
import torch
import torch.nn as nn
import socket
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Configuration ===
PLAYBACK_MODE = False             # True: 播放已预测CSV；False: 在线UDP推理+可视化
CSV_FILE = './output/predicted_skeleton.csv'
SUMMARY_FILE = 'preprocess_metadata.json'    # 与 file.py 保持一致
WEIGHT_PATH = './best_skeleton_model_5_5.pth'
UDP_IP = '127.0.0.1'
UDP_PORT = 54000                  # 与 receive2f.py 的 SEND_PORT 一致
INTERVAL_MS = 20
# === End Configuration ===

# === Console print options ===
PRINT_COORDS = True          # 是否打印实时坐标到命令行
PRINT_STYLE = 'rows'         # 'rows' 每关节一组；'csv' 单行CSV
PRINT_EVERY = 1              # 每隔多少帧打印一次（1=每帧）


# ——— 骨架连线定义（与原 gui.py 相同） ———
bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # spine
    (5, 6), (6, 7), (7, 8),               # left arm
    (9, 10), (10, 11), (11, 12), (5, 9),  # right arm + shoulder
    (13, 14), (14, 15), (15, 16),         # left leg
    (17, 18), (18, 19), (19, 20), (13, 17)# right leg + hip
]

# ========= 来自 file.py 的函数与模型，确保与训练时一致 =========

def print_skeleton(step: int, pts: np.ndarray, style: str = 'rows'):
    """
    step: 帧编号
    pts: 形状为 (21, 3) 的关节坐标
    style: 'rows' -> [frame 000001] J00=(x,y,z) J01=...
           'csv'  -> step,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
    """
    if style == 'csv':
        flat = ','.join(f'{v:.3f}' for v in pts.reshape(-1))
        print(f'{step},{flat}', flush=False)
    else:
        parts = [f'J{i:02d}=({x:.2f},{y:.2f},{z:.2f})' for i, (x, y, z) in enumerate(pts)]
        print(f'[frame {step:06d}] ' + ' '.join(parts), flush=False)


def add_positional_encoding(x):
    """与 file.py 一致的正弦位置编码"""
    batch_size, seq_len, d_model = x.shape
    pe = torch.zeros(seq_len, d_model, device=x.device)
    position = torch.arange(seq_len, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:(d_model // 2)])
    return x + pe.unsqueeze(0)

def compute_exponential_weights(k, m):
    """与 file.py 一致的指数权重"""
    indices = torch.arange(k)
    weights = torch.exp(-m * indices)
    return weights / weights.sum()

class EnhancedSkeletonTransformer(nn.Module):
    """严格复刻 file.py 的结构与超参接口"""
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 num_joints, num_dims=3, dropout=0.1, seq_length=5, window_size=5):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.seq_length = seq_length
        self.window_size = window_size

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.decoder_feature_extractor = nn.Sequential(
            nn.Linear(num_joints * num_dims, d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_encoder_layers
        )

        self.predict = nn.Sequential(
            nn.Linear(d_model * seq_length, d_model * window_size)
        )

        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * num_dims)
        )

        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, decoder_input):
        batch_size = x.shape[0]

        feats = self.feature_extractor(x).unsqueeze(1)  # (B,1,d)
        dec_in = self.decoder_feature_extractor(decoder_input)  # (B,S,d)
        dec_in = add_positional_encoding(dec_in)

        memory = self.transformer_encoder(feats)                 # (B,1,d)
        dec_out = self.transformer_decoder(dec_in, memory)       # (B,S,d)

        predict = dec_out.reshape(batch_size, -1)                # (B,S*d)
        predict_next = self.predict(predict)                     # (B,W*d)
        predict_next = predict_next.reshape(batch_size, self.window_size, -1)  # (B,W,d)

        output = self.output_decoder(predict_next) * self.output_scale  # (B,W,63)
        return output

# ========= 预处理：与 file.py 等价（输入为 Series） =========

def preprocess_pressure_data(left_series, right_series, stats):
    """
    left_series/right_series: 以 Series 形式，顺序为
    P1..P35, gx,gy,gz, ax,ay,az
    """
    left_pressure  = left_series.iloc[:35]
    left_rotation  = left_series.iloc[35:38]
    left_accel     = left_series.iloc[38:41]

    right_pressure = right_series.iloc[:35]
    right_rotation = right_series.iloc[35:38]
    right_accel    = right_series.iloc[38:41]

    pressure_combined = pd.concat([left_pressure, right_pressure], axis=0).ffill().bfill()
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=0).ffill().bfill()
    accel_combined    = pd.concat([left_accel, right_accel], axis=0).ffill().bfill()

    def normalize_and_standardize(data, s):
        normed = (data - s['min']) / (s['max'] - s['min'] + 1e-8)
        return (normed - s['mean']) / (s['std'] + 1e-8)

    p = normalize_and_standardize(pressure_combined, stats['pressure'])
    r = normalize_and_standardize(rotation_combined, stats['rotation'])
    a = normalize_and_standardize(accel_combined,    stats['accel'])

    input_features = np.concatenate([p, r, a], axis=0)  # 35*2 + 3*2 + 3*2 = 70 + 6 + 6 = 82? 实际每侧35+3+3=41，总共82
    # 训练侧为双足 + 旋转 + 加速度的展开，因此总维度以训练时保存的 stats 为准
    return input_features

# ========= 在线推理器：一次加载，循环复用 =========

class StreamPredictor:
    def __init__(self, summary_stats, weight_path):
        self.seq_length = 5
        self.window_size = 5
        self.num_joints = 21
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 根据 stats 推断输入维度（与 file.py 保持一致）
        # pressure(70) + rotation(6) + accel(6) = 82
        self.input_dim = 82

        self.model = EnhancedSkeletonTransformer(
            input_dim=self.input_dim,
            d_model=512, nhead=8, num_encoder_layers=6,
            num_joints=self.num_joints, num_dims=3, dropout=0.1,
            seq_length=self.seq_length, window_size=self.window_size
        ).to(self.device)

        # 加载 checkpoint（与 file.py 一致）
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.summary = summary_stats
        self.skeleton_last = torch.zeros((1, self.seq_length, self.num_joints * 3), device=self.device)
        self.step = 0
        self.m = 1.0  # 指数权重衰减系数（与 file.py 一致）

    @staticmethod
    def rec_to_series(rec: dict) -> pd.Series:
        """将 UDP 字典转为 Series，顺序：P1..P35, gx,gy,gz, ax,ay,az"""
        vals = [rec[f"P{i}"] for i in range(1, 36)]
        vals += [rec['gx'], rec['gy'], rec['gz'], rec['ax'], rec['ay'], rec['az']]
        return pd.Series(vals)

    def predict_once(self, rec_l: dict, rec_r: dict) -> np.ndarray:
        """单帧预测：与 file.py 单步离线逻辑对齐，取 window_size 序列中的第 1 帧作为输出"""
        left_s  = self.rec_to_series(rec_l)
        right_s = self.rec_to_series(rec_r)

        feats = preprocess_pressure_data(left_s, right_s, self.summary).reshape(1, -1)
        input_tensor = torch.from_numpy(feats).float().to(self.device)

        # 每 20 步清零一次（与 file.py 的“if i % 20 == 0: reset”逻辑等效）
        if self.step % 20 == 0:
            self.skeleton_last.zero_()

        with torch.no_grad():
            seq_pred = self.model(input_tensor, self.skeleton_last).squeeze(0)  # (W,63)
            # file.py 在单步情形下最终得到的就是第一帧（见其 action/num 更新后的权重求和效果）
            fused = seq_pred[0, :]  # (63,)

        # 递推：把 fused 放到 skeleton_last 的最后一位
        self.skeleton_last[:, :-1, :] = self.skeleton_last[:, 1:, :].clone()
        self.skeleton_last[:, -1, :] = fused

        self.step += 1
        return fused.detach().cpu().numpy()  # (63,)

# ========= 保存函数（保持与原 gui.py 一致的列名风格） =========

def save_predictions(pred_list, output_file='./output/predicted_skeleton.csv'):
    if not pred_list:
        print("[Live ] No predictions to save.")
        return
    arr = np.vstack(pred_list)  # (N,63)
    num_joints = arr.shape[1] // 3
    cols = []
    for i in range(num_joints):
        cols += [f'X.{i*2+1}', f'Y.{i*2+1}', f'Z.{i*2+1}']
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(output_file, index=False)
    print(f"[Live ] Predictions saved to {output_file}")

# ========= 主流程 =========

def main():
    df_skel = None
    total = 0
    idx = 0
    # 数据源与（在线模式下的）模型初始化
    if PLAYBACK_MODE:
        df_skel = pd.read_csv(CSV_FILE)
        total = len(df_skel)
        idx = 0
        print(f"[Playback] loading {CSV_FILE} ({total} frames)")
        predictor = None
    else:
        # 读取与 file.py 一致的统计量文件
        with open(SUMMARY_FILE, 'r') as f:
            raw = json.load(f)
        summary = {k: {kk: np.array(vv) for kk, vv in st.items()} for k, st in raw.items()}

        predictor = StreamPredictor(summary_stats=summary, weight_path=WEIGHT_PATH)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False)
        preds = []
        print(f"[Live ] listening on {UDP_IP}:{UDP_PORT}")
        print(f"[Live ] device: {predictor.device}")

    # 3D 可视化初始化
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.set_axis_off()
    x_min, x_max = -300, 300
    y_min, y_max = 0, 2000
    z_min, z_max = 0, 200
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    ax.view_init(elev=0, azim=0, roll=90)
    num_joints = 21
    scatter = ax.scatter([], [], [], c='cyan', s=20)
    lines = []
    for _ in bones:
        line, = ax.plot([], [], [], c='white', linewidth=2)
        lines.append(line)
    plt.title('Skeleton Viewer (Matplotlib)', color='white')

    def update(frame):
        nonlocal idx, df_skel, total
        if PLAYBACK_MODE:
            row = df_skel.iloc[idx].values
            pts = row.reshape(-1, 3)
            idx = (idx + 1) % total
        else:
            try:
                pkt, _ = sock.recvfrom(4096)
            except BlockingIOError:
                return scatter, *lines
            rec_l, rec_r = pickle.loads(pkt)
            out63 = predictor.predict_once(rec_l, rec_r)  # (63,)
            preds.append(out63)
            pts = out63.reshape(num_joints, 3)
            if PRINT_COORDS and (predictor.step % PRINT_EVERY == 0):
                print_skeleton(predictor.step, pts, style=PRINT_STYLE)

        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for (i, j), line in zip(bones, lines):
            xs = [pts[i, 0], pts[j, 0]]
            ys = [pts[i, 1], pts[j, 1]]
            zs = [pts[i, 2], pts[j, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        return scatter, *lines

    anim = animation.FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.show()

    if not PLAYBACK_MODE:
        save_predictions(preds)
        print("[Live ] all predictions saved.")

if __name__ == '__main__':
    main()
