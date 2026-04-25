import torch
import torch.nn as nn
import torch.onnx
import math
import pandas as pd
import numpy as np
import os
import copy
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Architecture: GatedTCN (Optimized for tract-onnx) ---

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

class GatedTCN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_blocks=4):
        super(GatedTCN, self).__init__()
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

# --- 2. Feature Engineering: DWT Integration ---

def prepare_features(prices, seq_len=32):
    print(f"Preparing features with Haar DWT (N={len(prices)})...")
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
    vol = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
    
    # Base features
    norm_magnitudes = magnitudes / (vol + 1e-8)
    
    # DWT Haar Level 1: Approximation (Trend) and Detail (Noise)
    # A1 = (x_t + x_t-1) / sqrt(2), D1 = (x_t - x_t-1) / sqrt(2)
    # We apply this to the raw prices
    p_curr = prices[1:]
    p_prev = prices[:-1]
    a1 = (p_curr + p_prev) / np.sqrt(2)
    d1 = (p_curr - p_prev) / np.sqrt(2)
    
    # Normalizing A1 to keep it scale-invariant
    a1_norm = a1 / (p_curr + 1e-8)
    
    # Streaks and Reversals
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
    
    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol, a1_norm, d1], axis=1)
    
    x, y_dir, y_vol = [], [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y_dir.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        y_vol.append(vol[i+seq_len])
        
    return (torch.from_numpy(np.array(x, dtype=np.float32)), 
            torch.from_numpy(np.array(y_dir, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.array(y_vol, dtype=np.float32)).unsqueeze(1))

# --- 3. Self-Supervised Learning: Contrastive Objective ---

def contrastive_loss(feat1, feat2, temperature=0.1):
    batch_size = feat1.shape[0]
    feat1 = nn.functional.normalize(feat1, dim=1)
    feat2 = nn.functional.normalize(feat2, dim=1)
    
    # Similarity matrix
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

# --- 4. Training Loop ---

def train_and_export():
    csv_path = "data/ticks.csv"
    seq_len = 32
    input_dim = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        df = pd.read_csv(csv_path, header=None, names=['epoch', 'quote'])
        prices = df['quote'].values.astype(np.float32)
        x_all, y_dir_all, y_vol_all = prepare_features(prices, seq_len)
    except Exception as e:
        print(f"Loading synthetic data: {e}")
        x_all = torch.randn(2000, seq_len, input_dim)
        y_dir_all = torch.randint(0, 2, (2000, 1)).float()
        y_vol_all = torch.rand(2000, 1)

    split = int(len(x_all) * 0.8)
    train_ds = TensorDataset(x_all[:split], y_dir_all[:split], y_vol_all[:split])
    val_ds = TensorDataset(x_all[split:], y_dir_all[split:], y_vol_all[split:])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = GatedTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Phase 1: Contrastive Pre-training (Noise-Resilient Representations)
    print("Starting Phase 1: Contrastive Pre-training...")
    for epoch in range(5): # Short pre-training
        model.train()
        total_cl_loss = 0
        for bx, _, _ in train_loader:
            bx = bx.to(device)
            # Create augmented views
            bx_aug1 = bx + torch.randn_like(bx) * 0.02
            bx_aug2 = bx + torch.randn_like(bx) * 0.02
            
            optimizer.zero_grad()
            f1 = model(bx_aug1, return_feat=True)
            f2 = model(bx_aug2, return_feat=True)
            
            loss = contrastive_loss(f1, f2)
            loss.backward()
            optimizer.step()
            total_cl_loss += loss.item()
        print(f"Pre-train Epoch {epoch}, CL Loss: {total_cl_loss/len(train_loader):.4f}")

    # Phase 2: Supervised Fine-tuning
    print("Starting Phase 2: Supervised Fine-tuning...")
    num_pos = torch.sum(y_dir_all[:split]).item()
    pos_weight = (split - num_pos) / num_pos if num_pos > 0 else 1.0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc = 0
    for epoch in range(20):
        model.train()
        for bx, by_dir, by_vol in train_loader:
            bx, by_dir, by_vol = bx.to(device), by_dir.to(device), by_vol.to(device)
            optimizer.zero_grad()
            out_dir, out_vol = model(bx)
            l_dir = focal_loss(out_dir, by_dir, torch.tensor(pos_weight).to(device))
            l_vol = nn.functional.mse_loss(out_vol, by_vol)
            loss = l_dir + 0.2 * l_vol
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by_dir, _ in val_loader:
                out_dir, _ = model(bx.to(device))
                vp.extend(out_dir.cpu().numpy().flatten())
                vt.extend(by_dir.cpu().numpy().flatten())
        
        auc = roc_auc_score(vt, vp)
        scheduler.step(auc)
        print(f"Epoch {epoch}, AUC: {auc:.4f}")
        if auc > best_auc: best_auc = auc

    # Export
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim).to(device)
    class ExportModel(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): 
            d, _ = self.m(x)
            return d
    torch.onnx.export(ExportModel(model), dummy, "model.onnx", opset_version=11, input_names=['input'], output_names=['output'])
    print(f"Exported GatedTCN V4 (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    train_and_export()
