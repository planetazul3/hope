import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_recall_curve, auc as pr_auc
import random as _random
import contextlib
import traceback

from hope_ml.common import SimpleTransformerV2, prepare_features, contrastive_loss, focal_loss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
_SEED = 42
_random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] >= 3:
        prices = df.iloc[:, 2].values
    else:
        prices = df.iloc[:, 1].values
    return prices.astype(np.float32)

def generate_ed25519_keypair_and_sign(model_path):
    try:
        signing_key_hex = os.environ.get("MODEL_SIGNING_KEY")
        if not signing_key_hex:
            logger.warning("MODEL_SIGNING_KEY not found. Skipping Ed25519 signature.")
            return

        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        # Convert hex back to bytes
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
    # Path Agnosticism: Fallback detection for cloud environments
    if log_dir is None:
        log_dir = os.environ.get("LOG_DIR", "logs")
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging to: {log_dir}")
    if csv_path is None:
        symbol = os.environ.get("DERIV_SYMBOL", "1HZ100V")
        file_name = f"{symbol}_ticks.csv"
        fallbacks = [
            f"/kaggle/input/ticks-csv/{file_name}",
            f"/kaggle/working/hope/data/{file_name}",
            f"/content/hope/data/{file_name}",
            f"/content/drive/MyDrive/hope/data/{file_name}",
            f"data/{file_name}"
        ]
        for path in fallbacks:
            if os.path.exists(path):
                csv_path = path
                break
        
    if csv_path is None:
        # Final default fallback if nothing exists yet
        symbol = os.environ.get("DERIV_SYMBOL", "1HZ100V")
        csv_path = f"data/{symbol}_ticks.csv" 
        
    seq_len = int(os.environ.get("TRANSFORMER_SEQUENCE_LENGTH", 32))
    input_dim = 8

    prices = load_data_from_csv(csv_path)
    if prices is None:
        return

    logger.info("Preparing features...")
    x_all, y_dir_all, y_vol_all = prepare_features(prices, seq_len=seq_len)

    split = int(len(x_all) * 0.8)
    train_ds = TensorDataset(x_all[:split], y_dir_all[:split], y_vol_all[:split])
    val_ds = TensorDataset(x_all[split + seq_len:], y_dir_all[split + seq_len:], y_vol_all[split + seq_len:])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = SimpleTransformerV2(input_dim=input_dim, max_seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Phase 1: Contrastive Pre-training (TS2Vec)
    logger.info("Starting Phase 1: TS2Vec Contrastive Pre-training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/5", leave=False)
        for bx, _, _ in pbar:
            bx = bx.to(device)
            optimizer.zero_grad()
            # Random crop could be added, but applying temporal masking in latent space is key:
            f1 = model(bx, return_feat=True, apply_ts2vec_mask=True, mask_p=0.5)
            f2 = model(bx, return_feat=True, apply_ts2vec_mask=True, mask_p=0.5)
            loss = contrastive_loss(f1, f2, temperature=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        logger.info(f"Phase 1 Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Phase 2: Supervised Fine-tuning
    logger.info("Starting Phase 2: Supervised Fine-tuning with MTL...")
    
    # Freeze encoder layers except the heads initially or use low-LR
    # Using low-LR approach for the whole model to simplify, as per "frozen/low-LR encoder"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    warmup_epochs = 5
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    num_pos = torch.sum(y_dir_all[:split]).item()
    if num_pos == 0:
        logger.warning("No positive labels found in training split. Setting pos_weight to 1.0.")
        pos_weight = 1.0
    else:
        pos_weight = (split - num_pos) / num_pos
    
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
                l_dir = focal_loss(out_dir, by_dir, pos_weight_t)
                # Continuous 10-tick forward MSE/Huber Loss:
                l_vol = nn.functional.huber_loss(out_vol, by_vol)
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
            
            # Log batch loss
            step = epoch * num_batches + pbar.n
            writer.add_scalar("Loss/Batch", loss.item(), step)
            writer.add_scalar("GradNorm/Batch", grad_norm.item(), step)

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

        # Log epoch metrics
        writer.add_scalar("Metrics/AUC", auc_val, epoch)
        writer.add_scalar("Metrics/PR-AUC", prauc_val, epoch)
        writer.add_scalar("Metrics/Accuracy", acc, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Optimization/LearningRate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Optimization/AvgGradNorm", avg_grad_norm, epoch)

        if auc_val > best_auc:
            best_auc = auc_val
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Export to ONNX (Static Execution Graphs)
    try:
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    except Exception as e:
        logger.error(f"Failed to load best model: {e}")
        return
        
    model.eval()

    class InferenceModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            d, _ = self.m(x)
            return d

    infer_model = InferenceModel(model).cpu()
    
    # Static batch size of 1 and sequence length of 32
    dummy = torch.randn(1, seq_len, input_dim)
    
    onnx_path = "model.onnx"
    try:
        torch.onnx.export(infer_model, dummy, onnx_path, export_params=True, opset_version=15,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])
        logger.info(f"Exported {onnx_path} (Static Graph: 1x32x8)")
    except Exception:
        logger.error(f"Failed to export ONNX: {traceback.format_exc()}")
        return

    # Dynamic INT8 Quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_onnx_path = "model_quantized.onnx"
        quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=QuantType.QInt8)
        logger.info(f"Exported {quantized_onnx_path} (Signed INT8)")
        
        # Replace original ONNX with quantized version for deployment
        os.replace(quantized_onnx_path, onnx_path)
        logger.info(f"Replaced {onnx_path} with quantized model.")
    except ImportError:
        logger.warning("ONNX quantization skipped (onnxruntime not installed)")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        
    # Cryptographic Model Signing
    generate_ed25519_keypair_and_sign(onnx_path)

if __name__ == "__main__":
    main()
