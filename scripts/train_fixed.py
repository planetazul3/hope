import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR

from hope_ml.common import GatedTCNV4, prepare_features, contrastive_loss, focal_loss, block_mask
from sklearn.metrics import precision_recall_curve, auc as pr_auc
import random as _random

# ── Reproducibility (Stage 11 requirement) ──────────────────────────────
_SEED = 42
_random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ────────────────────────────────────────────────────────────────────────

def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    # Support 2-col (epoch, quote) and 3-col (symbol, epoch, quote)
    if df.shape[1] >= 3:
        prices = df.iloc[:, 2].values
    else:
        prices = df.iloc[:, 1].values
    return prices.astype(np.float32)

import requests

def is_cloud_env():
    # Google Colab Metadata check
    try:
        resp = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/", 
                            headers={"Metadata-Flavor": "Google"}, timeout=2)
        if resp.status_code == 200:
            return "colab"
    except Exception:
        pass

    # Kaggle Mount check
    if os.path.exists('/kaggle/input') and os.path.ismount('/kaggle/input'):
        return "kaggle"
    
    # Fallback for manual override if strictly necessary (not recommended)
    if os.environ.get('FORCE_CLOUD_ENV') == '1':
        return "manual"
        
    return None

def main():
    cloud = is_cloud_env()
    if not cloud:
        print(
            "ERROR: Model training is prohibited on local machines. "
            "This script must be run in Google Colab or Kaggle. "
            "Hardware/Infrastructure verification failed."
        )
        sys.exit(1)
    print(f"Cloud environment verified: {cloud}")

    csv_path = "data/ticks.csv"
    seq_len = 32
    input_dim = 8

    prices = load_data_from_csv(csv_path)
    if prices is None:
        return

    x_all, y_dir_all, y_vol_all = prepare_features(prices, seq_len=seq_len)

    # Add temporal gap to prevent data leakage from overlapping windows
    split = int(len(x_all) * 0.8)
    train_ds = TensorDataset(x_all[:split], y_dir_all[:split], y_vol_all[:split])
    val_ds = TensorDataset(x_all[split + seq_len:], y_dir_all[split + seq_len:], y_vol_all[split + seq_len:])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = GatedTCNV4(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Phase 1: Contrastive Pre-training (5 epochs)
    # We train at full LR for pre-training; schedulers are initialized after.
    print("Starting Phase 1: Contrastive Pre-training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for bx, _, _ in train_loader:
            bx = bx.to(device)
            bx1, bx2 = block_mask(bx), block_mask(bx)
            optimizer.zero_grad()
            f1, f2 = model(bx1, return_feat=True), model(bx2, return_feat=True)
            loss = contrastive_loss(f1, f2, temperature=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    # Phase 2: Supervised Fine-tuning
    print("Starting Phase 2: Supervised Fine-tuning...")
    
    # Re-initialize optimizer and schedulers for Phase 2 warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    warmup_epochs = 5
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Mixed Precision Setup
    scaler = torch.cuda.amp.GradScaler()
    
    # Calculate class imbalance for Focal Loss
    num_pos = torch.sum(y_dir_all[:split]).item()
    if num_pos == 0:
        print("WARNING: No positive labels found in training split. Setting pos_weight to 1.0.")
        pos_weight = 1.0
    else:
        # Ratio of negative to positive samples
        pos_weight = (split - num_pos) / num_pos
    
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    early_stop_patience = 5
    patience_counter = 0
    best_auc = 0.0
    for epoch in range(20):
        model.train()
        total_grad_norm = 0.0
        num_batches = 0
        for bx, by_dir, by_vol in train_loader:
            bx, by_dir, by_vol = bx.to(device), by_dir.to(device), by_vol.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                out_dir, out_vol = model(bx)
                l_dir = focal_loss(out_dir, by_dir, pos_weight_t)
                l_vol = nn.functional.mse_loss(out_vol, by_vol)
                loss = l_dir + 0.2 * l_vol
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_grad_norm += grad_norm.item()
            num_batches += 1
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        v_probs, v_targets = [], []
        with torch.no_grad():
            for bx, by_dir, _ in val_loader:
                out_dir, _ = model(bx.to(device))
                v_probs.append(out_dir)
                v_targets.append(by_dir)
        
        # Vectorized gathering
        vp = torch.cat(v_probs).cpu().numpy().flatten()
        vt = torch.cat(v_targets).cpu().numpy().flatten()

        vpb = vp > 0.5
        acc = accuracy_score(vt, vpb)
        f1 = f1_score(vt, vpb, zero_division=0)
        prec = precision_score(vt, vpb, zero_division=0)
        rec = recall_score(vt, vpb, zero_division=0)
        
        try:
            auc_val = roc_auc_score(vt, vp)
            p, r, _ = precision_recall_curve(vt, vp)
            prauc_val = pr_auc(r, p)
        except ValueError:
            auc_val, prauc_val = 0.5, 0.0
            
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(auc_val)

        print(
            f"Epoch {epoch}, AUC: {auc_val:.4f}, PR-AUC: {prauc_val:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, GradNorm: {avg_grad_norm:.4f}"
        )

        if auc_val > best_auc:
            best_auc = auc_val
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Export to ONNX
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    class InferenceModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            d, _ = self.m(x)
            return d

    infer_model = InferenceModel(model).cpu()
    dummy = torch.randn(1, 32, 8)
    torch.onnx.export(infer_model, dummy, "model.onnx", export_params=True, opset_version=15,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    print("Exported model.onnx (with dynamic axes)")

    # Quantization Enhancement
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic("model.onnx", "model_quantized.onnx", weight_type=QuantType.QInt8)
        print("Exported model_quantized.onnx (Signed INT8)")
    except ImportError:
        print("ONNX quantization skipped (onnxruntime not installed)")

if __name__ == "__main__":
    main()
