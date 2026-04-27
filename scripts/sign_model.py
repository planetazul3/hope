#!/usr/bin/env python3
"""
Model Signer Utility v2.0
Secures ONNX models with Ed25519 digital signatures for the Rust StrategyEngine.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Requirement: pip install pynacl python-dotenv
try:
    import nacl.signing
    import nacl.encoding
    import nacl.exceptions
except ImportError:
    print("❌ Error: Missing 'pynacl'. Install it with: pip install pynacl")
    sys.exit(1)

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Signer")

class ModelSigner:
    def __init__(self, key_hex: str):
        try:
            self.signing_key = nacl.signing.SigningKey(
                key_hex, 
                encoder=nacl.encoding.HexEncoder
            )
            self.verify_key = self.signing_key.verify_key
        except Exception as e:
            raise ValueError(f"Invalid Private Key: {e}")

    def sign(self, model_path: Path, sig_path: Path):
        """Generates a detached signature for the model."""
        logger.info(f"Signing model: {model_path.name}")
        
        with open(model_path, "rb") as f:
            model_data = f.read()

        # Generate signature
        signed = self.signing_key.sign(model_data)
        signature = signed.signature

        with open(sig_path, "wb") as f:
            f.write(signature)
        
        logger.info(f"✅ Signature created: {sig_path.name}")
        logger.info(f"Public Key (Hex): {self.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode()}")

    def verify(self, model_path: Path, sig_path: Path):
        """Self-verification to ensure the signature is valid."""
        if not sig_path.exists():
            return False
            
        with open(model_path, "rb") as f:
            model_data = f.read()
        with open(sig_path, "rb") as f:
            signature = f.read()

        try:
            self.verify_key.verify(model_data, signature)
            logger.info("🛠️ Self-verification: PASSED")
            return True
        except nacl.exceptions.BadSignatureError:
            logger.error("❌ Self-verification: FAILED (Signature mismatch)")
            return False

def main():
    parser = argparse.ArgumentParser(description="Ed25519 Model Signing Utility for Hope ML")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to the .onnx file")
    parser.add_argument("--output", type=str, help="Path for the .sig file (default: model.onnx.sig)")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    
    args = parser.parse_args()
    load_dotenv(args.env)

    # 1. Environment Validation
    priv_key_hex = os.environ.get("MODEL_SIGNING_KEY")
    if not priv_key_hex:
        logger.error(f"MODEL_SIGNING_KEY not found in {args.env}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    sig_path = Path(args.output) if args.output else Path(f"{args.model}.sig")

    # 2. Processing
    try:
        signer = ModelSigner(priv_key_hex)
        signer.sign(model_path, sig_path)
        
        # 3. Integrity Check
        if not signer.verify(model_path, sig_path):
            sys.exit(1)
            
        logger.info("Model is ready for Rust Engine deployment.")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()