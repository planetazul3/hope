"""Shared ML utilities for the Hope trading engine.

Provides the GatedTCN V4 architecture definition, feature engineering,
contrastive and focal loss functions, and block masking augmentation.
"""
from hope_ml.common import (
    SimpleTransformerV2,
    prepare_features,
    contrastive_loss,
    focal_loss,
    ts2vec_mask,
)

__all__ = [
    "SimpleTransformerV2",
    "prepare_features",
    "contrastive_loss",
    "focal_loss",
    "ts2vec_mask",
]
