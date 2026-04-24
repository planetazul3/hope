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
    # Simple feature extraction mirroring Rust code
    # Features: Direction, Return Magnitude, Streak, Ticks since reversal, Volatility
    
    returns = np.diff(prices)
    directions = np.sign(returns)
    magnitudes = np.abs(returns)
    
    # Simple streak and reversal calculation
    streaks = []
    reversals = []
    curr_streak = 0
    curr_reversal = 0
    last_dir = 0
    
    for d in directions:
        if d == last_dir:
            curr_streak += 1
            curr_reversal += 1
        else:
            curr_streak = 1
            curr_reversal = 1
            last_dir = d
        streaks.append(curr_streak)
        reversals.append(curr_reversal)
        
    # Volatility (rolling std of returns)
    vol = []
    for i in range(len(returns)):
        start = max(0, i - 19)
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
        output_names=['output']
    )
    print("Done.")

if __name__ == "__main__":
    train_and_export()
