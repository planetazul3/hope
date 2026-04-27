import unittest

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


class TestModelExport(unittest.TestCase):
    def test_simple_model_exports_valid_onnx(self):
        import onnxruntime
        import numpy as np

        model = SimpleModel()
        dummy_input = torch.randn(1, 16, 5)
        export_path = "tests/fixtures/test_model.onnx"
        torch.onnx.export(model, dummy_input, export_path)

        sess = onnxruntime.InferenceSession(export_path)
        test_input = np.random.randn(1, 16, 5).astype(np.float32)
        outputs = sess.run(None, {sess.get_inputs()[0].name: test_input})
        output = outputs[0]

        self.assertEqual(output.shape, (1, 1))
        self.assertGreaterEqual(float(output[0, 0]), 0.0)
        self.assertLessEqual(float(output[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
