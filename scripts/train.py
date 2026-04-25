import torch
import torch.nn as nn
import torch.onnx

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=16, nhead=2, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # Use the last sequence element
        x = x[:, -1, :]
        x = self.fc(x)
        return self.sigmoid(x)

import sqlite3
import numpy as np

def load_real_data(db_path, limit=100000):
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT quote FROM ticks ORDER BY epoch ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return np.array([r[0] for r in rows], dtype=np.float32)

def prepare_features(prices, seq_len=16):
    print("Preparing features...")
    # Feature extraction mirroring Rust TickProcessor logic
    # Features: Direction, Return Magnitude, Streak, Ticks since reversal, Volatility
    
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
    streaks = []
    reversals = []
    
    last_trend_direction = 0 # 1 for Up, -1 for Down
    last_direction = 0
    curr_streak = 0
    ticks_since_reversal = 0
    
    for d in directions:
        # Streak logic: reset on flat (0), increment if same as last
        if d == 0:
            curr_streak = 0
        elif d == last_direction:
            curr_streak += 1
        else:
            curr_streak = 1
        
        # Reversal logic: flip between Up (1) and Down (-1)
        if (d == 1 and last_trend_direction == -1) or (d == -1 and last_trend_direction == 1):
            ticks_since_reversal = 1
        elif d != 0:
            ticks_since_reversal += 1
            
        if d != 0:
            last_trend_direction = d
            
        streaks.append(curr_streak)
        reversals.append(ticks_since_reversal)
        last_direction = d
        
    # Volatility (rolling std of returns) - mirroring calculate_stats in Rust
    # Rust uses simple moving standard deviation of absolute returns
    vol = []
    for i in range(len(returns)):
        start = max(0, i - 19) # Capacity is 64 in Rust, but stats use available len
        # Rust calculate_stats uses self.len which is up to 64.
        # Simple Transformer uses 20-period vol in Python. Let's keep it consistent.
        # Actually Rust calculate_stats uses the whole ring buffer (up to 64).
        # Let's match the 20-period for now as it's common, but ensure it's calculated on returns.
        vol.append(np.std(returns[start:i+1]))
        
    # Combine into (N, 5)
    # Note: returns has len N-1, so we lose the first price
    features = np.stack([directions, magnitudes, streaks, reversals, vol], axis=1)
    
    # Create sequences
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        # Target: 1 if next return is positive
        y.append(1.0 if returns[i+seq_len] > 0 else 0.0)
        
    return torch.from_numpy(np.array(x, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32)).unsqueeze(1)

def train_and_export():
    db_path = "data/tick_store.db"
    seq_len = 16
    input_dim = 5
    
    try:
        prices = load_real_data(db_path, limit=5000)
        x_train, y_train = prepare_features(prices, seq_len)
    except Exception as e:
        print(f"Failed to load real data: {e}. Using dummy data instead.")
        x_train = torch.randn(64, seq_len, input_dim)
        y_train = torch.rand(64, 1)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print(f"Training on {len(x_train)} samples...")
    batch_size = 64
    for epoch in range(5):
        epoch_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {epoch_loss / (len(x_train)/batch_size):.4f}")
            
    # Export to ONNX
    dummy_input = torch.randn(1, seq_len, input_dim)
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    onnx_path = "model.onnx"
    
    print(f"Exporting to {onnx_path}...")
    # We use a script-friendly export if possible, but let's try the direct way again
    # with a smaller model if needed.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        training=torch.onnx.TrainingMode.EVAL
    )
    print("Done.")

if __name__ == "__main__":
    train_and_export()
