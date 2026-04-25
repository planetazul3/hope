"""Shared ML utilities for the Hope trading engine.

Provides the GatedTCN V4 architecture definition, feature engineering,
contrastive and focal loss functions, and block masking augmentation.
"""
from hope_ml.common import (
    CausalPadding1d,
    GatedTCNBlock,
    SEModule,
    GatedTCNV4,
    prepare_features,
    contrastive_loss,
    focal_loss,
    block_mask,
)

__all__ = [
    "CausalPadding1d",
    "GatedTCNBlock",
    "SEModule",
    "GatedTCNV4",
    "prepare_features",
    "contrastive_loss",
    "focal_loss",
    "block_mask",
]
