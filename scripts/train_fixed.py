import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR

# Add scripts to path
sys.path.append(os.path.abspath('scripts'))
from hope_ml.common import GatedTCNV4, prepare_features, contrastive_loss, focal_loss, block_mask

def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=['epoch', 'quote'])
    return df['quote'].values.astype(np.float32)

def main():
    if 'COLAB_GPU' not in os.environ and 'KAGGLE_URL_BASE' not in os.environ:
        print(
            "ERROR: Model training is prohibited on local machines. "
            "Upload data/ticks.csv to Google Colab or Kaggle and execute "
            "notebooks/train_transformer.ipynb in a cloud GPU environment. "
            "Scripts contain runtime guards that abort execution outside cloud environments."
        )
        sys.exit(1)

    csv_path = "data/ticks.csv"
    seq_len = 32
    input_dim = 8

    prices = load_data_from_csv(csv_path)
    if prices is None:
        return

    x_all, y_dir_all, y_vol_all = prepare_features(prices, seq_len=seq_len)

    split = int(len(x_all) * 0.8)
    train_ds = TensorDataset(x_all[:split], y_dir_all[:split], y_vol_all[:split])
    val_ds = TensorDataset(x_all[split:], y_dir_all[split:], y_vol_all[split:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = GatedTCNV4(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    warmup_epochs = 5
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Phase 1: Contrastive Pre-training (5 epochs)
    print("Starting Phase 1...")
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

    # Phase 2: Supervised Fine-tuning (up to 20 epochs with early stopping)
    print("Starting Phase 2...")
    num_pos = torch.sum(y_dir_all[:split]).item()
    pos_weight = (split - num_pos) / num_pos if num_pos > 0 else 1.0

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
            out_dir, out_vol = model(bx)
            l_dir = focal_loss(out_dir, by_dir, torch.tensor(pos_weight).to(device))
            l_vol = nn.functional.mse_loss(out_vol, by_vol)
            (l_dir + 0.2 * l_vol).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_grad_norm += grad_norm.item()
            num_batches += 1
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by_dir, _ in val_loader:
                out_dir, _ = model(bx.to(device))
                vp.extend(out_dir.cpu().numpy().flatten())
                vt.extend(by_dir.cpu().numpy().flatten())

        vpb = np.array(vp) > 0.5
        acc = accuracy_score(vt, vpb)
        f1 = f1_score(vt, vpb, zero_division=0)
        prec = precision_score(vt, vpb, zero_division=0)
        rec = recall_score(vt, vpb, zero_division=0)
        auc = roc_auc_score(vt, vp)
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(auc)

        print(
            f"Epoch {epoch}, AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, "
            f"Prec: {prec:.4f}, Rec: {rec:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, GradNorm: {avg_grad_norm:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
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
                      input_names=['input'], output_names=['output'])
    print("Exported model.onnx")

if __name__ == "__main__":
    main()
