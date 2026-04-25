import torch
import torch.nn as nn
import torch.onnx
import math
import pandas as pd
import numpy as np
import os
import copy
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=32, nhead=4, num_layers=3, max_seq_len=32, dropout=0.1, pooling='mean'):
        super(SimpleTransformer, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pooling = pooling
        
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
        
        if mask is None:
            sz = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(sz).to(x.device)
        
        x = self.transformer_encoder(x, mask=mask)
        
        if self.pooling == 'mean':
            x = torch.mean(x, dim=1)
        else:
            x = x[:, -1, :]
        
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
    
    norm_magnitudes = magnitudes / (vol + 1e-8)
    norm_streaks = np.log1p(np.array(streaks, dtype=np.float32))
    norm_reversals = np.log1p(np.array(reversals, dtype=np.float32))
        
    features = np.stack([directions, norm_magnitudes, norm_streaks, norm_reversals, vol], axis=1)
    
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        
    return torch.from_numpy(np.array(x, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32)).unsqueeze(1)

# Task 3: Focal Loss with Label Smoothing
def focal_loss(output, target, pos_weight, gamma=2.0, smoothing=0.05):
    # Apply label smoothing
    target_smooth = target * (1 - smoothing) + 0.5 * smoothing
    
    bce_loss = nn.functional.binary_cross_entropy(output, target_smooth, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)
    weight = torch.where(target == 1, pos_weight, torch.tensor(1.0).to(output.device))
    loss = weight * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)

def train_and_export():
    # Setup Logger
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler
    fh = logging.FileHandler("training.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

    # Task 2: DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_pos = torch.sum(y_train).item()
    num_neg = len(y_train) - num_pos
    pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
    print(f"Class imbalance: pos={num_pos}, neg={num_neg}, pos_weight={pos_weight_val:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SimpleTransformer(input_dim=input_dim, max_seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_auc = 0.0
    best_model_state = None
    early_stop_patience = 7
    patience_counter = 0

    logger.info(f"Training V2 (CSV-based) on {len(x_train)} samples...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        # tqdm progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Data augmentation: noise
            noise = torch.randn_like(batch_x) * 0.01
            batch_x = batch_x + noise
            
            optimizer.zero_grad()
            output = model(batch_x)
            # Loss calculation with Focal Loss and Label Smoothing
            loss = focal_loss(output, batch_y, torch.tensor(pos_weight_val).to(device))
            loss.backward()
            
            # Gradient clipping to ensure training stability in deep Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            # tqdm progress bar for validation
            for batch_x, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = focal_loss(output, batch_y, torch.tensor(pos_weight_val).to(device))
                val_loss += loss.item() * len(batch_x)
                
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        
        avg_train_loss = train_loss / len(x_train)
        avg_val_loss = val_loss / len(x_val)
        
        val_auc = roc_auc_score(all_targets, all_preds)
        val_preds_binary = np.array(all_preds) > 0.5
        val_acc = accuracy_score(all_targets, val_preds_binary)
        
        # Task 4: Expanded metrics
        val_precision = precision_score(all_targets, val_preds_binary, zero_division=0)
        val_recall = recall_score(all_targets, val_preds_binary, zero_division=0)
        val_f1 = f1_score(all_targets, val_preds_binary, zero_division=0)
        
        logger.info("-" * 80)
        logger.info(f"Epoch {epoch:02d} | Loss: T={avg_train_loss:.4f} V={avg_val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")
        logger.info(f"         | P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}")
        logger.info("-" * 80)
        
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

    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
    )
    print("Export complete: model.onnx (Static Batch Size: 1)")

    # Task 14: ONNX Validation
    try:
        import onnx
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX validation successful: model.onnx is valid.")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")

if __name__ == "__main__":
    train_and_export()
