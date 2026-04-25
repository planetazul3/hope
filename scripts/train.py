import torch
import torch.nn as nn
import torch.onnx
import math
import pandas as pd
import numpy as np
import os
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

class Chpadding1d(nn.Module):
    """Causal padding for 1D convolution."""
    def __init__(self, padding):
        super(Chpadding1d, self).__init__()
        self.padding = padding
    def forward(self, x):
        return nn.functional.pad(x, (self.padding, 0))

class GatedTCNBlock(nn.Module):
    """Gated Dilated Convolutional Block with Residual Connection."""
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
        x = f * g
        
        return x + res

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module."""
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        # Use simple mean instead of AdaptiveAvgPool1d for tract stability
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, l
        y = torch.mean(x, dim=2)
        y = self.fc(y).view(y.size(0), y.size(1), 1)
        return x * y

class GatedTCN(nn.Module):
    """Gated TCN with SE Attention and Multi-task Heads."""
    def __init__(self, input_dim=7, hidden_dim=64, num_blocks=4):
        super(GatedTCN, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        self.blocks = nn.ModuleList([
            GatedTCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_blocks)
        ])
        
        self.se = SEModule(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary head for volatility prediction (training only)
        self.vol_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Input shape: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.se(x)
        
        # Global Max Pooling: (B, C, L) -> (B, C)
        feat, _ = torch.max(x, dim=2)
        
        direction = self.classifier(feat)
        volatility = self.vol_head(feat)
        
        return direction, volatility

def load_data_from_csv(csv_path, limit=200000):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Run 'make export' first.")
    df = pd.read_csv(csv_path, header=None, names=['epoch', 'quote'], nrows=limit)
    return df['quote'].values.astype(np.float32)

def prepare_features(prices, seq_len=32):
    print(f"Preparing features (N={len(prices)})...")
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
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
        
    vol = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
    
    norm_magnitudes = magnitudes / (vol + 1e-8)
    norm_streaks = np.log1p(np.array(streaks, dtype=np.float32))
    norm_reversals = np.log1p(np.array(reversals, dtype=np.float32))
    
    # Phase 2: Frequency Domain Features (Simple 4-point FFT Mag proxy)
    # We use moving standard deviation of 2-tick and 4-tick returns as frequency proxies
    freq_hf = pd.Series(returns).rolling(window=2).std().fillna(0).values
    freq_lf = pd.Series(returns).rolling(window=4).std().fillna(0).values
        
    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol, freq_hf, freq_lf], axis=1)
    
    x, y_dir, y_vol = [], [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y_dir.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        # Target for auxiliary head: future volatility
        y_vol.append(vol[i+seq_len])
        
    return (torch.from_numpy(np.array(x, dtype=np.float32)), 
            torch.from_numpy(np.array(y_dir, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.array(y_vol, dtype=np.float32)).unsqueeze(1))

def focal_loss(output, target, pos_weight, gamma=2.0, smoothing=0.05):
    target_smooth = target * (1 - smoothing) + 0.5 * smoothing
    bce_loss = nn.functional.binary_cross_entropy(output, target_smooth, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)
    weight = torch.where(target == 1, pos_weight, torch.tensor(1.0).to(output.device))
    loss = weight * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)

def train_and_export():
    csv_path = "data/ticks.csv"
    seq_len = 32
    input_dim = 7 # 5 base + 2 freq
    
    try:
        prices = load_data_from_csv(csv_path)
        x_all, y_dir_all, y_vol_all = prepare_features(prices, seq_len)
    except Exception as e:
        print(f"Error loading CSV: {e}. Using synthetic data.")
        x_all = torch.randn(1000, seq_len, input_dim)
        y_dir_all = torch.randint(0, 2, (1000, 1)).float()
        y_vol_all = torch.rand(1000, 1)

    split_idx = int(len(x_all) * 0.8)
    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    y_dir_train, y_dir_val = y_dir_all[:split_idx], y_dir_all[split_idx:]
    y_vol_train, y_vol_val = y_vol_all[:split_idx], y_vol_all[split_idx:]

    train_ds = TensorDataset(x_train, y_dir_train, y_vol_train)
    val_ds = TensorDataset(x_val, y_dir_val, y_vol_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    num_pos = torch.sum(y_dir_train).item()
    pos_weight_val = (len(y_dir_train) - num_pos) / num_pos if num_pos > 0 else 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GatedTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_auc = 0.0
    best_model_state = None
    early_stop_patience = 10
    patience_counter = 0

    print(f"Training GatedTCN on {len(x_train)} samples...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_x, batch_y_dir, batch_y_vol in train_loader:
            batch_x, batch_y_dir, batch_y_vol = batch_x.to(device), batch_y_dir.to(device), batch_y_vol.to(device)
            batch_x = batch_x + torch.randn_like(batch_x) * 0.01
            
            optimizer.zero_grad()
            out_dir, out_vol = model(batch_x)
            
            loss_dir = focal_loss(out_dir, batch_y_dir, torch.tensor(pos_weight_val).to(device))
            # Auxiliary loss: MSE for volatility prediction
            loss_vol = nn.functional.mse_loss(out_vol, batch_y_vol)
            
            loss = loss_dir + 0.2 * loss_vol
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y_dir, _ in val_loader:
                batch_x, batch_y_dir = batch_x.to(device), batch_y_dir.to(device)
                out_dir, _ = model(batch_x)
                all_preds.extend(out_dir.cpu().numpy().flatten())
                all_targets.extend(batch_y_dir.cpu().numpy().flatten())
        
        val_auc = roc_auc_score(all_targets, all_preds)
        val_acc = accuracy_score(all_targets, np.array(all_preds) > 0.5)
        print(f"Epoch {epoch}, Loss: {train_loss/len(x_train):.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience: break
            
    if best_model_state: model.load_state_dict(best_model_state)

    # Export to ONNX (Excluding volatility head for inference efficiency)
    model.eval()
    class InferenceModel(nn.Module):
        def __init__(self, trained_model):
            super().__init__()
            self.model = trained_model
        def forward(self, x):
            direction, _ = self.model(x)
            return direction
            
    infer_model = InferenceModel(model)
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    torch.onnx.export(
        infer_model, dummy_input, "model.onnx",
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )
    print("Export complete: model.onnx (GatedTCN)")

if __name__ == "__main__":
    train_and_export()
