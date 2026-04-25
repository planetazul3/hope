import torch
import torch.nn as nn
import torch.onnx
import math
import pandas as pd
import numpy as np
import os
import copy
from sklearn.metrics import roc_auc_score, accuracy_score

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=32, nhead=4, num_layers=3, max_seq_len=32, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        x = self.input_norm(x)
        x = self.embedding(x)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Task 4 from previous audit: Causal attention mask
        if mask is None:
            sz = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(sz).to(x.device)
        
        x = self.transformer_encoder(x, mask=mask)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

def load_data_from_csv(csv_path, limit=200000):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Run 'make export' first.")
    
    df = pd.read_csv(csv_path, header=None, names=['epoch', 'quote'], nrows=limit)
    return df['quote'].values.astype(np.float32)

def prepare_features(prices, seq_len=32):
    print(f"Preparing features (N={len(prices)})...")
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
    streaks = []
    reversals = []
    last_trend_direction = 0
    last_direction = 0
    curr_streak = 0
    ticks_since_reversal = 0
    
    for d in directions:
        if d == 0:
            curr_streak = 0
        elif d == last_direction:
            curr_streak += 1
        else:
            curr_streak = 1
        
        if (d == 1 and last_trend_direction == -1) or (d == -1 and last_trend_direction == 1):
            ticks_since_reversal = 1
        elif d != 0:
            ticks_since_reversal += 1
            
        if d != 0:
            last_trend_direction = d
        streaks.append(curr_streak)
        reversals.append(ticks_since_reversal)
        last_direction = d
        
    vol = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
    
    # Task 1 from previous audit: Feature normalization
    norm_magnitudes = magnitudes / (vol + 1e-8)
    norm_streaks = np.log1p(np.array(streaks, dtype=np.float32))
    norm_reversals = np.log1p(np.array(reversals, dtype=np.float32))
        
    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol], axis=1)
    
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        
    return torch.from_numpy(np.array(x, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32)).unsqueeze(1)

# Task 4: Focal Loss implementation
def focal_loss(output, target, pos_weight, gamma=2.0):
    bce_loss = nn.functional.binary_cross_entropy(output, target, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)
    weight = torch.where(target == 1, pos_weight, torch.tensor(1.0).to(output.device))
    loss = weight * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)

def train_and_export():
    csv_path = "data/ticks.csv"
    seq_len = 32
    input_dim = 5
    
    try:
        prices = load_data_from_csv(csv_path)
        x_all, y_all = prepare_features(prices, seq_len)
    except Exception as e:
        print(f"Error loading CSV: {e}. Using synthetic data.")
        x_all = torch.randn(1000, seq_len, input_dim)
        y_all = torch.randint(0, 2, (1000, 1)).float()

    num_samples = len(x_all)
    split_idx = int(num_samples * 0.8)
    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    num_pos = torch.sum(y_train).item()
    num_neg = len(y_train) - num_pos
    pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
    print(f"Class imbalance: pos={num_pos}, neg={num_neg}, pos_weight={pos_weight_val:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleTransformer(input_dim=input_dim, max_seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Task 5: Scheduler based on ROC-AUC (max mode)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_auc = 0.0
    best_model_state = None
    early_stop_patience = 7
    patience_counter = 0

    print(f"Training V2 (CSV-based) on {len(x_train)} samples...")
    batch_size = 64
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        # Task 1: Shuffle dataset per epoch
        perm = torch.randperm(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuffled[i:i+batch_size].to(device)
            batch_y = y_train_shuffled[i:i+batch_size].to(device)
            
            # Task 3: Gaussian noise injection (data augmentation)
            noise = torch.randn_like(batch_x) * 0.01
            batch_x = batch_x + noise
            
            optimizer.zero_grad()
            output = model(batch_x)
            # Task 4: Use Focal Loss
            loss = focal_loss(output, batch_y, torch.tensor(pos_weight_val).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for i in range(0, len(x_val), batch_size):
                batch_x = x_val[i:i+batch_size].to(device)
                batch_y = y_val[i:i+batch_size].to(device)
                output = model(batch_x)
                loss = focal_loss(output, batch_y, torch.tensor(pos_weight_val).to(device))
                val_loss += loss.item() * len(batch_x)
                
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        
        avg_train_loss = train_loss / len(x_train)
        avg_val_loss = val_loss / len(x_val)
        
        # Task 2: Calculate ROC-AUC and Accuracy
        val_auc = roc_auc_score(all_targets, all_preds)
        val_acc = accuracy_score(all_targets, np.array(all_preds) > 0.5)
        
        print(f"Epoch {epoch}, Loss: T={avg_train_loss:.4f} V={avg_val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Task 5: Early stopping and checkpointing based on AUC
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  --> New best AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val AUC: {best_val_auc:.4f}")

    # Task 7 from previous audit: Static ONNX export (Batch size 1)
    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
    )
    print("Export complete: model.onnx (Static Batch Size: 1)")

if __name__ == "__main__":
    train_and_export()
