import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(16 * 5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 16, 5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

def export():
    model = SimpleModel()
    dummy_input = torch.randn(1, 16, 5)
    torch.onnx.export(model, dummy_input, "model.onnx")
    print("Exported simple model to model.onnx")

if __name__ == "__main__":
    export()
