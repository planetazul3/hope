import torch
import torch.nn as nn
import torch.onnx
import pandas as pd
import numpy as np
import os
import copy
import logging
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Import shared ML components (Audit Task 8 & 9)
from hope_ml.common import GatedTCNV4, prepare_features, contrastive_loss, focal_loss, block_mask

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
        logging.warning(f"Loading synthetic data: {e}")
        x_all = torch.randn(2000, seq_len, input_dim)
        y_dir_all = torch.randint(0, 2, (2000, 1)).float()
        y_vol_all = torch.rand(2000, 1)

    split = int(len(x_all) * 0.8)
    train_ds = TensorDataset(x_all[:split], y_dir_all[:split], y_vol_all[:split])
    val_ds = TensorDataset(x_all[split:], y_dir_all[split:], y_vol_all[split:])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = GatedTCNV4(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Phase 1: Contrastive Pre-training (Noise-Resilient Representations)
    print("Starting Phase 1: Contrastive Pre-training...")
    for epoch in range(5):
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
    logging.basicConfig(level=logging.INFO) # Audit Task 20
    train_and_export()
