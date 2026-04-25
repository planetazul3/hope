import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from hope_ml.common import focal_loss, contrastive_loss, block_mask, prepare_features


class TestHopeMlCommon(unittest.TestCase):
    def test_focal_loss_returns_nonnegative_scalar(self):
        output = torch.sigmoid(torch.randn(16, 1))
        target = torch.randint(0, 2, (16, 1)).float()
        pos_weight = torch.tensor(1.5)
        loss = focal_loss(output, target, pos_weight)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_contrastive_loss_returns_nonnegative_scalar(self):
        feat1 = torch.randn(8, 64)
        feat2 = torch.randn(8, 64)
        loss = contrastive_loss(feat1, feat2, temperature=0.1)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_block_mask_preserves_shape(self):
        x = torch.randn(4, 32, 8)
        out = block_mask(x)
        self.assertEqual(out.shape, x.shape)

    def test_prepare_features_returns_correct_shapes(self):
        rng = np.random.default_rng(0)
        prices = rng.normal(100.0, 0.1, 300).cumsum().astype(np.float32)
        seq_len = 32
        x, y_dir, y_vol = prepare_features(prices, seq_len=seq_len)

        n = x.shape[0]
        self.assertEqual(x.shape, (n, seq_len, 8))
        self.assertEqual(y_dir.shape, (n, 1))
        self.assertEqual(y_vol.shape, (n, 1))

        unique_dirs = set(y_dir.squeeze().tolist())
        self.assertTrue(unique_dirs.issubset({0.0, 1.0}))
        self.assertTrue((y_vol >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
