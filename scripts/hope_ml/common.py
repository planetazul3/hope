import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import math

__all__ = [
    "SimpleTransformerV2",
    "prepare_features",
    "contrastive_loss",
    "focal_loss",
    "ts2vec_mask",
]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformerV2(nn.Module):
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_layers=2, max_seq_len=32):
        super(SimpleTransformerV2, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.kaiming_normal_(self.cls_token, mode='fan_in', nonlinearity='relu')
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.direction_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.volatility_head = nn.Linear(d_model, 1)

    def generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, return_feat=False, apply_ts2vec_mask=False, mask_p=0.5):
        B, L, _ = x.shape
        x = self.input_proj(x)
        
        if apply_ts2vec_mask and self.training:
            mask = torch.bernoulli(torch.full((B, L, 1), 1 - mask_p, device=x.device))
            x = x * mask

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_encoder(x)
        causal_mask = self.generate_causal_mask(L + 1, x.device)
        
        out = self.transformer_encoder(x, mask=causal_mask, is_causal=True)
        feat = out[:, 0, :]
        
        if return_feat:
            return feat
            
        return self.direction_head(feat), self.volatility_head(feat)

def prepare_features(prices, seq_len=32):
    if len(prices) <= seq_len + 3:
        raise ValueError(f"prepare_features requires at least {seq_len + 4} price samples")

    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)

    vol = pd.Series(returns).rolling(window=10, min_periods=1).std(ddof=0).fillna(0).values
    long_term_vol = pd.Series(returns).rolling(window=50, min_periods=1).std(ddof=0).fillna(0).values
    vol_ratio = vol / (long_term_vol + 1e-8)

    norm_magnitudes = magnitudes / (vol + 1e-8)

    h = np.array([0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921])
    g = np.array([-0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690])
    
    db2_a1 = np.zeros(len(prices) - 1)
    db2_d1 = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        if i >= 3:
            p_window = np.array([prices[i+1], prices[i], prices[i-1], prices[i-2]])
            db2_a1[i] = np.dot(p_window, h)
            db2_d1[i] = np.dot(p_window, g)
        else:
            db2_a1[i] = 0.0
            db2_d1[i] = 0.0

    p_curr = prices[1:]
    db2_a1_norm = db2_a1 / (p_curr + 1e-8)

    streaks, reversals = [], []
    last_trend_direction, last_direction, curr_streak, ticks_since_reversal = 0, 0, 0, 0
    for d in directions:
        if d == 0: curr_streak = 0
        elif d == last_direction: curr_streak += 1
        else: curr_streak = 1
        if (d == 1 and last_trend_direction == -1) or (d == -1 and last_trend_direction == 1):
            ticks_since_reversal = 1
        elif d != 0: ticks_since_reversal += 1
        if d != 0: last_trend_direction = d
        streaks.append(curr_streak)
        reversals.append(ticks_since_reversal)
        last_direction = d

    norm_streaks = np.log1p(np.array(streaks, dtype=np.float32))
    norm_reversals = np.log1p(np.array(reversals, dtype=np.float32))

    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol, db2_a1_norm, db2_d1, vol_ratio], axis=1)

    x, y_dir, y_vol = [], [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y_dir.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        y_vol.append(vol[i+seq_len])

    x_tensor = torch.from_numpy(np.array(x, dtype=np.float32))
    y_dir_tensor = torch.from_numpy(np.array(y_dir, dtype=np.float32)).unsqueeze(1)
    y_vol_tensor = torch.from_numpy(np.array(y_vol, dtype=np.float32)).unsqueeze(1)

    n_samples = x_tensor.shape[0]
    pos_ratio = float(torch.sum(y_dir_tensor) / n_samples) * 100
    logging.info(f"prepare_features: shape={tuple(x_tensor.shape)}, samples={n_samples}, class_balance={pos_ratio:.2f}% positive")
    for col_idx, col_name in enumerate(["direction", "norm_magnitude", "norm_streak", "norm_reversal", "vol", "db2_a1", "db2_d1", "vol_ratio"]):
        col = x_tensor[:, :, col_idx]
        logging.info(f"  feature[{col_idx}] {col_name}: mean={col.mean().item():.4f}, std={col.std().item():.4f}")

    return x_tensor, y_dir_tensor, y_vol_tensor

def contrastive_loss(feat1, feat2, temperature=0.1):
    batch_size = feat1.shape[0]
    feat1 = nn.functional.normalize(feat1, dim=1)
    feat2 = nn.functional.normalize(feat2, dim=1)

    logits = torch.matmul(feat1, feat2.T) / temperature
    labels = torch.arange(batch_size).to(feat1.device)

    loss1 = nn.functional.cross_entropy(logits, labels)
    loss2 = nn.functional.cross_entropy(logits.T, labels)
    return (loss1 + loss2) / 2.0

def focal_loss(output, target, pos_weight=None, gamma=2.0, smoothing=0.05):
    target_smooth = target * (1 - smoothing) + 0.5 * smoothing
    bce_loss = nn.functional.binary_cross_entropy(output, target_smooth, reduction='none')
    focal_pt = torch.where(target_smooth > 0.5, output, 1 - output)
    if pos_weight is not None:
        weight = torch.where(target == 1, pos_weight, torch.ones_like(output))
    else:
        weight = torch.ones_like(output)
    loss = weight * (1 - focal_pt) ** gamma * bce_loss
    return torch.mean(loss)

def ts2vec_mask(x, mask_p=0.5):
    return x
