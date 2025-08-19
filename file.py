import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os

def add_positional_encoding(x):
    """
    Add positional encoding to the input tensor.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

    Returns:
        Tensor: The input tensor with added positional encoding, same shape as x
    """
    batch_size, seq_len, d_model = x.shape

    # Initialize the positional encoding matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model, device=x.device)

    # Create a tensor of positions (0, 1, ..., seq_len-1), shape (seq_len, 1)
    position = torch.arange(seq_len, device=x.device).unsqueeze(1)

    # Compute the denominator term for the sinusoidal functions, shape (d_model/2,)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))

    # Apply sine to even indices in the embedding dimension
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cosine to odd indices in the embedding dimension
    # If d_model is odd, slicing ensures no index out of range
    pe[:, 1::2] = torch.cos(position * div_term[:(d_model // 2)])

    # Reshape to (1, seq_len, d_model) so it can be broadcasted across the batch dimension
    pe = pe.unsqueeze(0)

    # Add positional encoding to the input tensor
    return x + pe
def compute_exponential_weights(k, m):
    indices = torch.arange(k)  # Generate i = 0, 1, ..., k-1
    weights = torch.exp(-m * indices)  # Compute w_i = exp(-m * i)
    return weights / weights.sum()  # Normalize weights to sum to 1
class EnhancedSkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1, seq_length=3, window_size=5):
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Feedforward layer size
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Apply normalization before attention and feedforward
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Decoder input feature extractor
        self.decoder_feature_extractor = nn.Sequential(
            nn.Linear(num_joints * num_dims, d_model)  # Project joint input to d_model dimension
        )

        # Transformer decoder layer configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_encoder_layers
        )

        # Linear layer to expand decoder output over prediction window
        self.predict = nn.Sequential(
            nn.Linear(d_model * seq_length, d_model * window_size)  # From seq_length tokens to window_size predictions
        )

        # Output decoder for generating final joint coordinates
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * num_dims)  # Final output shape: (batch, window_size, num_joints * num_dims)
        )

        # Learnable output scaling factor
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, decoder_input):
        """
        Forward pass of the EnhancedSkeletonTransformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            decoder_input (Tensor): Decoder input tensor of shape (batch_size, seq_length, num_joints * num_dims)

        Returns:
            Tensor: Predicted joint sequence of shape (batch_size, window_size, num_joints * num_dims)
        """
        batch_size = x.shape[0]

        # Encode the input features
        features = self.feature_extractor(x)              # (batch_size, d_model)
        features = features.unsqueeze(1)                  # Add sequence dimension: (batch_size, 1, d_model)

        # Process decoder input
        decoder_input = self.decoder_feature_extractor(decoder_input)  # (batch_size, seq_length, d_model)
        # Optional: Add positional encoding to decoder_input
        decoder_input = add_positional_encoding(decoder_input)

        # Pass through transformer encoder and decoder
        transformer_output = self.transformer_encoder(features)        # (batch_size, 1, d_model)
        transformer_output = self.transformer_decoder(decoder_input, transformer_output)  # (batch_size, seq_length, d_model)

        # Reshape for prediction
        predict = transformer_output.reshape(batch_size, -1)           # Flatten to (batch_size, seq_length * d_model)
        predict_next = self.predict(predict)                           # (batch_size, window_size * d_model)
        predict_next = predict_next.reshape(batch_size, self.window_size, -1)  # (batch_size, window_size, d_model)

        # Generate output joint coordinates and apply scaling
        output = self.output_decoder(predict_next)                     # (batch_size, window_size, num_joints * num_dims)
        output = output * self.output_scale                            # Apply learnable scaling

        return output


import time


def preprocess_pressure_data(left_data, right_data, value):
    """圧力、回転、加速度データの前処理"""
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:35]
    left_rotation = left_data.iloc[35:38]
    left_accel = left_data.iloc[38:41]
    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:35]
    right_rotation = right_data.iloc[35:38]
    right_accel = right_data.iloc[38:41]

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=0)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=0)
    accel_combined = pd.concat([left_accel, right_accel], axis=0)

    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 正規化と標準化
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
        pressure_processed,
        rotation_processed,
        accel_processed,
    ], axis=0)

    return input_features


def predict_skeleton(input, summary):
    # データの読み込みと前処理
    pres_left = input['left']
    pres_right = input['right']

    input_features = preprocess_pressure_data(pres_left, pres_right, summary)
    input_features = input_features.reshape(1, input_features.shape[0])

    # 入力の次元数を取得
    seq_length = 5
    window_size = 5
    m = 1
    input_dim = input_features.shape[1]
    num_joints = 21

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # チェックポイントの読み込み（weights_only=Trueを追加）
    checkpoint = torch.load('./weight/best_skeleton_model_3_1.pth', map_location=device, weights_only=True)

    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    action = torch.zeros((10, window_size, 63)).to(device)
    num = np.zeros(10)
    print(action.shape)
    # 予測の実行
    print("Making predictions...")
    start_time = time.time()
    predictions = torch.zeros(1, 63).to(device)
    with torch.no_grad():
        skeleton_last = torch.zeros((seq_length, 63))
        skeleton_last = skeleton_last.unsqueeze(0).to(device)
        for i in range(1):
            input_tensor = torch.FloatTensor(input_features)[i].to(device)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            if i % 20 == 0:
                skeleton_last = torch.zeros_like(skeleton_last)
            skeleton_predict_seq = model(input_tensor, skeleton_last)
            skeleton_predict_seq = skeleton_predict_seq.squeeze(0)
            skeleton_predict = torch.zeros(63).to(device)
            for j in range(window_size):
                action[i + j, int(num[i + j])] = skeleton_predict_seq[j, :]
                num[i + j] += 1
                # print(f"j={j},i+j={i+j},num[i+j}]={int(num[i+j])}")
            weights = compute_exponential_weights(int(num[i]), m).to(device)
            for j in range(int(num[i])):
                skeleton_predict += weights[j] * action[i, int(num[i]) - 1 - j]
            predictions[i] = skeleton_predict
            for j in range(seq_length - 1):
                skeleton_last[0, j] = skeleton_last[0, j + 1]
            skeleton_last[0, seq_length - 1] = skeleton_predict
    '''
    # 予測の実行
    print("Making predictions...")
    predictions=torch.zeros(min_length,63).to(device)
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_features).to(device)
        input_tensor=input_tensor.to(device)
        print(input_tensor.shape,skeleton_data.shape)
        predictions=model(input_tensor,skeleton_data)
    '''
    print(f"Prediction shape: {predictions.shape}")

    predictions = predictions.cpu().numpy()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Prediction took {elapsed_time} seconds")
    return predictions


def save_predictions(predictions, output_file='./output/predicted_skeleton.csv'):
    # 予測結果をデータフレームに変換
    num_joints = predictions.shape[1] // 3
    columns = []
    for i in range(num_joints):
        columns.extend([f'X.{i * 2 + 1}', f'Y.{i * 2 + 1}', f'Z.{i * 2 + 1}'])

    df_predictions = pd.DataFrame(predictions, columns=columns)
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def main():
    skeleton_data = pd.read_csv('./data/20241115test3/Opti-track/Take 2024-11-15 03.50.00 PM.csv')
    pressure_data_left = pd.read_csv('./data/20241115test3/insoleSensor/20241115_155500_left.csv', skiprows=1)
    pressure_data_right = pd.read_csv('./data/20241115test3/insoleSensor/20241115_155500_right.csv', skiprows=1)

    with open('summary_stats.json', 'r') as f:
        summary = json.load(f)

    summary = {
        sensor: {k: np.array(v) for k, v in stat.items()}
        for sensor, stat in summary.items()
    }

    input = {}
    input['left'] = pressure_data_left.iloc[0, :]
    input['right'] = pressure_data_right.iloc[0, :]

    print("Starting prediction process...")
    predictions = predict_skeleton(input, summary)

    print("\nSaving predictions...")
    save_predictions(predictions)
    print(predictions)

    print("Prediction process completed successfully!")


if __name__ == "__main__":
    main()