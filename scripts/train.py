import torch
import torch.nn as nn
import torch.onnx
import math
import pandas as pd
import numpy as np
import os

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

    def forward(self, x):
        x = self.input_norm(x)
        x = self.embedding(x)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

def load_data_from_csv(csv_path, limit=200000):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Run 'make export' first.")
    
    # Read CSV (expecting columns: epoch, quote)
    df = pd.read_csv(csv_path, nrows=limit)
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
        if d == 0: curr_streak = 0
        elif d == last_direction: curr_streak += 1
        else: curr_streak = 1
        
        if (d == 1 and last_trend_direction == -1) or (d == -1 and last_trend_direction == 1):
            ticks_since_reversal = 1
        elif d != 0:
            ticks_since_reversal += 1
            
        if d != 0: last_trend_direction = d
        streaks.append(curr_streak)
        reversals.append(ticks_since_reversal)
        last_direction = d
        
    vol = []
    for i in range(len(returns)):
        start = max(0, i - 19)
        vol.append(np.std(returns[start:i+1]))
        
    features = np.stack([directions, magnitudes, streaks, reversals, vol], axis=1)
    
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        
    return torch.from_numpy(np.array(x, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32)).unsqueeze(1)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleTransformer(input_dim=input_dim, max_seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    print(f"Training V2 (CSV-based) on {len(x_train)} samples...")
    batch_size = 64
    for epoch in range(15):
        model.train()
        train_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(x_val), batch_size):
                batch_x = x_val[i:i+batch_size].to(device)
                batch_y = y_val[i:i+batch_size].to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * len(batch_x)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss/len(x_train):.4f}, Val Loss: {val_loss/len(x_val):.4f}")
            
    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete: model.onnx")

if __name__ == "__main__":
    train_and_export()
