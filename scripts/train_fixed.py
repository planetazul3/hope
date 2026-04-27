import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc as pr_auc
from torch.optim.lr_scheduler import LambdaLR
import random as _random
import contextlib
import traceback
import gc

from hope_ml.common import SimpleTransformerV2, prepare_features, contrastive_loss, focal_loss

# Institutional Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Strict Reproducibility
_SEED = 42
_random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_sanitize_data(csv_path, seq_len=32):
    """Loads data, engineers features, sanitizes NaN/Infs, and extracts 3D sequences."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"Loading raw data from: {csv_path}")
    
    # Handle both original format and the new format dynamically
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip')
    if df.shape[1] >= 3:
        df = df.rename(columns={0: 'epoch', 1: 'symbol', 2: 'quote'})
    else:
        df = df.rename(columns={0: 'epoch', 1: 'quote'})
        
    df['quote'] = pd.to_numeric(df['quote'], errors='coerce')
    df.dropna(subset=['quote'], inplace=True)

    # 1. FEATURE ENGINEERING (Generates NaNs at the edges)
    df['log_ret'] = np.log(df['quote'] / df['quote'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()

    # Future targets (Generates NaNs at the end)
    df['target_quote_future'] = df['quote'].shift(-10)
    df['target_direction'] = np.where(df['target_quote_future'] > df['quote'], 1.0, 0.0)
    df['target_volatility'] = df['volatility_20'].shift(-10)

    # 2. STRICT SANITIZATION BARRIER (Pandas 2D Level)
    logger.info(f"Dimensions before sanitization: {df.shape}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Dimensions after sanitization: {df.shape}")

    if df.empty:
        raise ValueError("CRITICAL: Dataset became empty after removing NaNs/Infs.")

    # 3. 3D SEQUENCE EXTRACTION (Preserving causality)
    features = df[['quote', 'log_ret', 'volatility_20']].values
    targets_dir = df['target_direction'].values
    targets_vol = df['target_volatility'].values

    X, y_dir, y_vol = [], [], []
    for i in range(len(df) - seq_len):
        X.append(features[i:i+seq_len])
        y_dir.append(targets_dir[i+seq_len-1])
        y_vol.append(targets_vol[i+seq_len-1])

    X_arr = np.array(X, dtype=np.float32)
    y_dir_arr = np.array(y_dir, dtype=np.float32)
    y_vol_arr = np.array(y_vol, dtype=np.float32)

    # Resilient prepare_features pass
    try:
        X_tensor, y_dir_tensor, y_vol_tensor = prepare_features(X_arr, y_dir_arr, y_vol_arr)
    except TypeError:
        logger.warning("Falling back to direct tensor conversion for features.")
        X_tensor = torch.tensor(X_arr)
        y_dir_tensor = torch.tensor(y_dir_arr)
        y_vol_tensor = torch.tensor(y_vol_arr)

    input_dim = X_tensor.shape[2]
    return X_tensor, y_dir_tensor, y_vol_tensor, input_dim

def generate_ed25519_keypair_and_sign(model_path):
    """Cryptographically signs the model if the key exists."""
    try:
        signing_key_hex = os.environ.get("MODEL_SIGNING_KEY")
        if not signing_key_hex:
            logger.warning("MODEL_SIGNING_KEY not found. The model is exported but NOT signed.")
            return

        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        key_bytes = bytes.fromhex(signing_key_hex)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
        public_key = private_key.public_key()
        
        with open(model_path, 'rb') as f:
            model_data = f.read()

        signature = private_key.sign(model_data)

        sig_path = f"{model_path}.sig"
        with open(sig_path, 'wb') as f:
            f.write(signature)
            
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        logger.info(f"Generated Ed25519 signature: {sig_path}")
        logger.info(f"Public Key (Hex, for MODEL_PUBLIC_KEY in .env): {pub_bytes.hex()}")
    except ImportError:
        logger.error("cryptography package not installed. Skipping Ed25519 signature generation.")
    except Exception as e:
        logger.error(f"Failed to generate Ed25519 signature: {e}")

def main(csv_path: str = None, log_dir: str = None):
    try:
        if log_dir is None:
            log_dir = os.environ.get("LOG_DIR", "logs")
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging to: {log_dir}")
        
        # Cloud Environment Fallbacks
        if csv_path is None:
            symbol = os.environ.get("DERIV_SYMBOL", "1HZ100V")
            file_name = f"{symbol}_ticks.csv"
            fallbacks = [
                f"/kaggle/input/ticks-csv/{file_name}",
                f"/kaggle/working/hope/data/{file_name}",
                f"/content/hope/data/{file_name}",
                f"/content/drive/MyDrive/hope/data/{file_name}",
                f"data/{file_name}",
                "data/ticks.csv"
            ]
            for path in fallbacks:
                if os.path.exists(path):
                    csv_path = path
                    break
                    
        if csv_path is None:
            raise FileNotFoundError("Could not automatically detect the ticks dataset.")

        seq_len = int(os.environ.get("TRANSFORMER_SEQUENCE_LENGTH", 32))
        
        # Load, Sanitize, and Extract
        X_all, y_dir_all, y_vol_all, input_dim = load_and_sanitize_data(csv_path, seq_len=seq_len)

        # Chronological Split (Strictly 80/20 to avoid time-series leakage)
        split = int(len(X_all) * 0.8)
        train_ds = TensorDataset(X_all[:split], y_dir_all[:split], y_vol_all[:split])
        val_ds = TensorDataset(X_all[split + seq_len:], y_dir_all[split + seq_len:], y_vol_all[split + seq_len:])
        
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {device} | Input Dimension: {input_dim}")

        model = SimpleTransformerV2(input_dim=input_dim, max_seq_len=seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # ---------------------------------------------------------
        # PHASE 1: TS2Vec Contrastive Pre-training
        # ---------------------------------------------------------
        logger.info("Starting Phase 1: TS2Vec Contrastive Pre-training...")
        for epoch in range(5):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/5", leave=False)
            for bx, _, _ in pbar:
                bx = bx.to(device)
                optimizer.zero_grad()
                f1 = model(bx, return_feat=True, apply_ts2vec_mask=True, mask_p=0.5)
                f2 = model(bx, return_feat=True, apply_ts2vec_mask=True, mask_p=0.5)
                loss = contrastive_loss(f1, f2, temperature=0.1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            logger.info(f"Phase 1 Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

        # ---------------------------------------------------------
        # PHASE 2: Supervised Fine-tuning with MTL & AMP
        # ---------------------------------------------------------
        logger.info("Starting Phase 2: Supervised Fine-tuning with MTL...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        num_pos = torch.sum(y_dir_all[:split]).item()
        pos_weight = 1.0 if num_pos == 0 else (split - num_pos) / num_pos
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

        early_stop_patience = 5
        patience_counter = 0
        best_auc = 0.0

        for epoch in range(20):
            model.train()
            total_grad_norm = 0.0
            num_batches = len(train_loader)

            pbar = tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/20", leave=False)
            for bx, by_dir, by_vol in pbar:
                bx, by_dir, by_vol = bx.to(device), by_dir.to(device), by_vol.to(device)
                optimizer.zero_grad()

                with (torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()):
                    out_dir, out_vol = model(bx)

                out_dir_f32, out_vol_f32 = out_dir.float(), out_vol.float()

                # Adjusted for dimensional matching
                l_dir = focal_loss(out_dir_f32, by_dir.unsqueeze(-1).float() if out_dir_f32.dim() > 1 else by_dir.float(), pos_weight_t)
                l_vol = nn.functional.huber_loss(out_vol_f32, by_vol.unsqueeze(-1).float() if out_vol_f32.dim() > 1 else by_vol.float())
                loss = l_dir + 0.2 * l_vol

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_grad_norm += grad_norm.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Validation
            model.eval()
            v_probs, v_targets = [], []
            with torch.no_grad():
                for bx, by_dir, _ in val_loader:
                    out_dir, _ = model(bx.to(device))
                    v_probs.append(out_dir)
                    v_targets.append(by_dir)

            vp = torch.cat(v_probs).cpu().numpy().flatten()
            vt = torch.cat(v_targets).cpu().numpy().flatten()
            vpb = vp > 0.5
            acc = accuracy_score(vt, vpb)
            f1 = f1_score(vt, vpb, zero_division=0)

            try:
                auc_val = roc_auc_score(vt, vp)
                p, r, _ = precision_recall_curve(vt, vp)
                prauc_val = pr_auc(r, p)
            except ValueError:
                logger.warning("Validation batch with single class. Defaulting metrics for safety.")
                auc_val, prauc_val = 0.5, 0.0

            avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(auc_val)

            logger.info(
                f"Phase 2 Epoch {epoch+1}, AUC: {auc_val:.4f}, PR-AUC: {prauc_val:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}, GradNorm: {avg_grad_norm:.4f}"
            )

            writer.add_scalar("Metrics/AUC", auc_val, epoch)
            writer.add_scalar("Metrics/PR-AUC", prauc_val, epoch)

            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(model.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # ---------------------------------------------------------
        # EXPORT & OPTIMIZATION
        # ---------------------------------------------------------
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
        dummy = torch.randn(1, seq_len, input_dim)
        onnx_path = "model.onnx"
        
        torch.onnx.export(infer_model, dummy, onnx_path, export_params=True, opset_version=15,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])
        logger.info(f"Exported {onnx_path} (Static Graph: 1x{seq_len}x{input_dim})")

        # Dynamic INT8 Quantization
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantized_onnx_path = "model_quantized.onnx"
            quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=QuantType.QInt8)
            os.replace(quantized_onnx_path, onnx_path)
            logger.info(f"Replaced {onnx_path} with quantized model.")
        except ImportError:
            logger.warning("ONNX quantization skipped (onnxruntime not installed)")
            
        # Cryptographic Model Signing
        generate_ed25519_keypair_and_sign(onnx_path)

    except Exception as e:
        logger.error(f"Critical failure in pipeline: {e}")
        traceback.print_exc()
    finally:
        # Secure Context Clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()