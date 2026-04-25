import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

# --- Architecture: GatedTCN (Optimized for tract-onnx) ---

class Chpadding1d(nn.Module):
    def __init__(self, padding):
        super(Chpadding1d, self).__init__()
        self.padding = padding
    def forward(self, x):
        return nn.functional.pad(x, (self.padding, 0))

class GatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.pad = Chpadding1d(padding)
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
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
    vol = pd.Series(returns).rolling(window=10, min_periods=1).std().fillna(0).values
    long_term_vol = pd.Series(returns).rolling(window=50, min_periods=1).std().fillna(0).values
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
        
    return (torch.from_numpy(np.array(x, dtype=np.float32)), 
            torch.from_numpy(np.array(y_dir, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.array(y_vol, dtype=np.float32)).unsqueeze(1))

# --- Training Utilities ---

def contrastive_loss(feat1, feat2, temperature=0.1):
    batch_size = feat1.shape[0]
    feat1 = nn.functional.normalize(feat1, dim=1)
    feat2 = nn.functional.normalize(feat2, dim=1)
    
    logits = torch.matmul(feat1, feat2.T) / temperature
    labels = torch.arange(batch_size).to(feat1.device)
    
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def focal_loss(output, target, pos_weight, gamma=2.0, smoothing=0.05):
    target_smooth = target * (1 - smoothing) + 0.5 * smoothing
    bce_loss = nn.functional.binary_cross_entropy(output, target_smooth, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)
    weight = torch.where(target == 1, pos_weight, torch.tensor(1.0).to(output.device))
    loss = weight * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)

def block_mask(x, mask_ratio=0.15, block_size=4):
    x = x.clone()
    b, l, c = x.shape
    num_blocks = l // block_size
    num_masked_blocks = int(num_blocks * mask_ratio)
    
    for i in range(b):
        masked_indices = np.random.choice(num_blocks, num_masked_blocks, replace=False)
        for idx in masked_indices:
            x[i, idx*block_size : (idx+1)*block_size, :] = 0
            
    # Inject Gaussian noise to non-masked elements
    noise = torch.randn_like(x) * 0.01
    mask = (x != 0).float()
    x = x + noise * mask
    return x
