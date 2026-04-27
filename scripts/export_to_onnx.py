import torch
import torch.nn as nn
import os
import sys
from hope_ml.common import SimpleTransformerV2

def export():
    # Load model configuration from environment or defaults
    seq_len = int(os.environ.get("TRANSFORMER_SEQUENCE_LENGTH", 32))
    input_dim = 8
    
    model = SimpleTransformerV2(input_dim=input_dim, max_seq_len=seq_len)
    
    pth_path = "best_model.pth"
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found.")
        return

    print(f"Loading weights from {pth_path}...")
    model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
    model.eval()

    class InferenceModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            # Extract only the direction probability
            d, _ = self.m(x)
            return d

    infer_model = InferenceModel(model)
    dummy_input = torch.randn(1, seq_len, input_dim)
    onnx_path = "model.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        infer_model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=15,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output']
    )
    print("Export complete.")

    # Sign the model if signer script is available
    signer_script = "scripts/sign_model.py"
    if os.path.exists(signer_script):
        print("Signing model...")
        os.system(f"python3 {signer_script} --model {onnx_path}")

if __name__ == "__main__":
    export()
