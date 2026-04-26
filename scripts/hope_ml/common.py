import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

__all__ = [
    "CausalPadding1d",
    "GatedTCNBlock",
    "SEModule",
    "GatedTCNV4",
    "prepare_features",
    "contrastive_loss",
    "focal_loss",
    "block_mask",
]

# --- Architecture: GatedTCN (Optimized for tract-onnx) ---

class CausalPadding1d(nn.Module):
    def __init__(self, padding):
        super(CausalPadding1d, self).__init__()
        self.padding = padding
    def forward(self, x):
        return nn.functional.pad(x, (self.padding, 0))

class GatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.pad = CausalPadding1d(padding)
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.proj(x)
        x_pad = self.pad(x)
        f = torch.tanh(self.conv_filter(x_pad))
        g = torch.sigmoid(self.conv_gate(x_pad))
        return (f * g) + res

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = torch.mean(x, dim=2)
        y = self.fc(y).view(y.size(0), y.size(1), 1)
        return x * y

class GatedTCNV4(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_blocks=4):
        super(GatedTCNV4, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            GatedTCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_blocks)
        ])
        self.se = SEModule(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.vol_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_feat=False):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.se(x)
        feat, _ = torch.max(x, dim=2)

        if return_feat: return feat

        return self.classifier(feat), self.vol_head(feat)

# --- Feature Engineering: DWT Integration ---

def prepare_features(prices, seq_len=32):
    if len(prices) <= seq_len + 1:
        raise ValueError(
            f"prepare_features requires at least {seq_len + 2} price samples "
            f"(seq_len + 2), but got {len(prices)}."
        )

    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)

    vol = pd.Series(returns).rolling(window=10, min_periods=1).std(ddof=0).fillna(0).values
    long_term_vol = pd.Series(returns).rolling(window=50, min_periods=1).std(ddof=0).fillna(0).values
    vol_ratio = vol / (long_term_vol + 1e-8)

    norm_magnitudes = magnitudes / (vol + 1e-8)

    p_curr = prices[1:]
    p_prev = prices[:-1]
    a1 = (p_curr + p_prev) / np.sqrt(2)
    d1 = (p_curr - p_prev) / np.sqrt(2)

    a1_norm = a1 / (p_curr + 1e-8)

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

    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol, a1_norm, d1, vol_ratio], axis=1)

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
    for col_idx, col_name in enumerate(["direction", "norm_magnitude", "norm_streak", "norm_reversal", "vol", "a1_norm", "d1", "vol_ratio"]):
        col = x_tensor[:, :, col_idx]
        logging.info(f"  feature[{col_idx}] {col_name}: mean={col.mean().item():.4f}, std={col.std().item():.4f}")
    print(f"Class balance: {pos_ratio:.2f}% positive labels")

    return x_tensor, y_dir_tensor, y_vol_tensor

# --- Training Utilities ---

def contrastive_loss(feat1, feat2, temperature=0.1):
    batch_size = feat1.shape[0]
    feat1 = nn.functional.normalize(feat1, dim=1)
    feat2 = nn.functional.normalize(feat2, dim=1)

    logits = torch.matmul(feat1, feat2.T) / temperature
    labels = torch.arange(batch_size).to(feat1.device)

    loss1 = nn.functional.cross_entropy(logits, labels)
    loss2 = nn.functional.cross_entropy(logits.T, labels)
    return (loss1 + loss2) / 2.0

def focal_loss(output, target, pos_weight, gamma=2.0, smoothing=0.05):
    target_smooth = target * (1 - smoothing) + 0.5 * smoothing
    bce_loss = nn.functional.binary_cross_entropy(output, target_smooth, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)
    weight = torch.where(target == 1, pos_weight, torch.ones_like(output))
    loss = weight * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)

def block_mask(x, mask_ratio=0.15, block_size=4, seed=None):
    x = x.clone()
    b, l, c = x.shape
    num_blocks = l // block_size
    if num_blocks == 0:
        return x
    num_masked_blocks = int(num_blocks * mask_ratio)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    noise_mask = torch.ones(b, l, 1, dtype=torch.bool, device=x.device)
    for i in range(b):
        masked_indices = rng.choice(num_blocks, num_masked_blocks, replace=False)
        for idx in masked_indices:
            start_idx = idx * block_size
            end_idx = (idx + 1) * block_size
            x[i, start_idx : end_idx, :] = 0
            noise_mask[i, start_idx : end_idx, :] = False

    # Inject Gaussian noise to non-masked elements
    noise = torch.randn_like(x) * 0.01
    x = x + noise * noise_mask.float()
    return x
