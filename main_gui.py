#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import socket
import pickle
import time

from vispy import scene, app

# ——————————————————————————————————————————————
# 以下内容与原始 main.py 完全一致，只是添加了 Vispy 相关的导入
# ——————————————————————————————————————————————

class EnhancedSkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 num_joints, num_dims=3, dropout=0.1, seq_length=3, window_size=5):
        super().__init__()

        # Save key hyperparameters as class attributes
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.seq_length = seq_length
        self.window_size = window_size

        # Feature extractor for the encoder input
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model)  # Project input features to d_model dimension
        )

        # Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder positional embedding
        self.decoder_feature_extractor = nn.Linear(
            num_joints * num_dims, d_model
        )

        # Transformer decoder layer configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_encoder_layers
        )

        # Prediction head: from encoder output to next-step embedding
        self.predict = nn.Sequential(
            nn.Linear(seq_length * d_model, window_size * d_model),
            nn.ReLU()
        )

        # Final output decoder: from d_model back to joint coordinates
        self.output_decoder = nn.Linear(
            window_size * d_model, window_size * num_joints * num_dims
        )

        # Learnable scaling for output magnitudes
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, decoder_input):
        """
        Forward pass of the transformer-based skeleton predictor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            decoder_input (Tensor): Decoder input tensor of shape (batch_size, seq_length, num_joints * num_dims)

        Returns:
            Tensor: Predicted joint sequence of shape (batch_size, window_size, num_joints * num_dims)
        """
        batch_size = x.shape[0]

        # Encode the input features
        features = self.feature_extractor(x)              # (batch_size, d_model)
        features = features.unsqueeze(1)                  # (batch_size, 1, d_model)
        memory = self.transformer_encoder(features)       # (batch_size, 1, d_model)

        # Prepare decoder input
        dec_in = self.decoder_feature_extractor(decoder_input)  # (batch_size, seq_length, d_model)
        dec_out = self.transformer_decoder(dec_in, memory)      # (batch_size, seq_length, d_model)

        # Flatten and predict next embedding
        dec_out_flat = dec_out.reshape(batch_size, -1)          
        next_emb = self.predict(dec_out_flat)                  

        # Decode to joint coordinates
        output = self.output_decoder(next_emb)                  
        output = output * self.output_scale                     
        output = output.reshape(batch_size, self.window_size, -1)
        return output


def add_positional_encoding(x):
    """
    Add positional encoding to the input tensor.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

    Returns:
        Tensor: The input tensor with added positional encoding, same shape as x
    """
    batch_size, seq_len, d_model = x.shape
    pe = torch.zeros(seq_len, d_model, device=x.device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:(d_model // 2)])
    pe = pe.unsqueeze(0)
    return x + pe


def preprocess_pressure_data(left_data, right_data, value):
    """圧力、回転、加速度データの前処理"""
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[0, :35]
    left_rotation = left_data.iloc[0, 35:38]
    left_accel = left_data.iloc[0, 38:41]

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[0, :35]
    right_rotation = right_data.iloc[0, 35:38]
    right_accel = right_data.iloc[0, 38:41]

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=0).ffill().bfill()
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=0).ffill().bfill()
    accel_combined = pd.concat([left_accel, right_accel], axis=0).ffill().bfill()

    def normalize_and_standardize(data, stats):
        # Min-Max normalization
        normed = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
        # Standardization
        standardized = (normed - stats['mean']) / (stats['std'] + 1e-8)
        return standardized

    pressure_processed = normalize_and_standardize(pressure_combined, value['pressure'])
    rotation_processed = normalize_and_standardize(rotation_combined, value['rotation'])
    accel_processed = normalize_and_standardize(accel_combined, value['accel'])

    # すべての特徴量を結合（246次元になるはず）
    input_features = np.concatenate([
        pressure_processed.values,
        rotation_processed.values,
        accel_processed.values,
    ], axis=0)

    return input_features


def predict_skeleton(input, summary):
    """
    按照原始逻辑，接收 {'left': df_l, 'right': df_r} 字典
    并返回 shape=(1, num_joints*3) 的 NumPy 预测数组。
    """
    pres_left = input['left']
    pres_right = input['right']

    input_features = preprocess_pressure_data(pres_left, pres_right, summary)
    input_features = input_features.reshape(1, -1)

    # ハイパーパラメータ（原始程序保持不变）
    seq_length = 3
    window_size = 1
    input_dim = input_features.shape[1]
    num_joints = 21

    # デバイスの設定
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # モデルの初期化（固定パラメータを使用）
    model = EnhancedSkeletonTransformer(
        input_dim=input_dim,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_joints=num_joints,
        num_dims=3,
        dropout=0.1,
        seq_length=seq_length,
        window_size=window_size
    ).to(device)

    # 骨架“过去”数据的占位（原始程序之所以 shape 为 (seq_length, num_joints*3)）
    skeleton_data = torch.zeros((1, seq_length, num_joints * 3), device=device)

    # 推理
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_features).float().to(device)
        prediction_tensor = model(input_tensor, skeleton_data)
    predictions = prediction_tensor.cpu().numpy()

    return predictions


def rec_to_df(rec):
    """
    将接收到的 rec 字典转换为 1×41 的 DataFrame：
      0–34    : P1–P35
      35–37   : gx, gy, gz
      38–40   : ax, ay, az
    """
    values = [rec[f"P{i}"] for i in range(1, 36)]
    values += [rec['gx'], rec['gy'], rec['gz']]
    values += [rec['ax'], rec['ay'], rec['az']]
    return pd.DataFrame([values])


def save_predictions(predictions, output_file='./output/predicted_skeleton.csv'):
    df_predictions = pd.DataFrame()
    for prediction in predictions:
        num_joints = prediction.shape[1] // 3
        columns = []
        for i in range(num_joints):
            columns.extend([f'X.{i * 2 + 1}', f'Y.{i * 2 + 1}', f'Z.{i * 2 + 1}'])
        df_pred = pd.DataFrame(prediction, columns=columns)
        df_predictions = pd.concat([df_predictions, df_pred], ignore_index=True)
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# ——————————————————————————————————————————————
# Vispy 实时可视化主循环
# ——————————————————————————————————————————————

def main():
    # 加载归一化统计量
    with open('summary_stats.json', 'r') as f:
        raw = json.load(f)
    summary = {sensor: {k: np.array(v) for k, v in stats.items()}
               for sensor, stats in raw.items()}

    # UDP 接收设置
    UDP_IP = '127.0.0.1'
    UDP_PORT = 54000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    # 初始化 Vispy 3D 窗口
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
    canvas.title = 'Real-time Skeleton (Vispy)'
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 2.5

    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.set_data(np.zeros((1, 3)),
                     face_color=(0.2, 0.7, 1.0, 1.0),
                     size=6)

    predictions = []

    def update(event):
        try:
            packet, _ = sock.recvfrom(4096)
        except BlockingIOError:
            return

        rec_l, rec_r = pickle.loads(packet)

        df_l = rec_to_df(rec_l)
        df_r = rec_to_df(rec_r)

        input_data = {'left': df_l, 'right': df_r}

        print("Starting prediction process...")
        pred = predict_skeleton(input_data, summary)  # shape = (1, num_joints*3)
        predictions.append(pred)
        print(pred)
        print("Prediction completed!\n")

        pts = pred.reshape(-1, 3)
        scatter.set_data(pts,
                         face_color=(0.2, 0.7, 1.0, 1.0),
                         size=6)

    # 50 Hz 更新
    timer = app.Timer(interval=0.02, connect=update, start=True)

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        save_predictions(predictions)
        print("All predictions have been saved. Exiting.")


if __name__ == "__main__":
    main()
