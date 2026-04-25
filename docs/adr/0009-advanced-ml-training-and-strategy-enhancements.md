# 0009: Advanced ML Training and Strategy Enhancements

*   **Status**: Accepted
*   **Date**: 2024-05-21

## Context

To further improve model generalization and strategy robustness, several advanced machine learning and algorithmic enhancements were required. The model needed to better handle temporal dependencies without data leakage, and the strategy engine required more flexibility in its dynamic thresholding.

## Decision

We have implemented the following enhancements:

1.  **Causal Attention Masking**: Modified the Transformer architecture to use a square subsequent mask, ensuring that predictions only depend on past information and preventing future data leakage.
2.  **Focal Loss**: Replaced weighted Binary Cross-Entropy with Focal Loss to better address class imbalance and focus the model on "hard" examples (misclassified market signals).
3.  **Data Augmentation & Shuffling**: Introduced Gaussian noise injection (0.01 factor) during training and per-epoch dataset shuffling to prevent memorization and improve generalization.
4.  **ROC-AUC Based Optimization**: Shifted the primary training metric to ROC-AUC. Checkpointing and the learning rate scheduler now monitor validation AUC to maximize the model's discriminative power.
5.  **Configurable Strategy Modifiers**: Refactored the `StrategyEngine` to accept `volatility_penalty` and `momentum_reward` as parameters, moving away from hardcoded values.
6.  **Enhanced Feature Preprocessing**: Implemented robust normalization (`log1p` on streaks/reversals and magnitude/volatility scaling) across both Python and Rust environments.

## Consequences

*   **Robustness**: The model is less likely to overfit due to data augmentation and causal constraints.
*   **Sensitivity**: Focal Loss improves the detection of rare but profitable market shifts.
*   **Flexibility**: Operators can now tune momentum rewards and volatility penalties via environment variables (`STRATEGY_MOMENTUM_REWARD`, `STRATEGY_VOLATILITY_PENALTY`) without recompiling the code.
*   **Visibility**: Training logs now include AUC and accuracy metrics, providing better insight into model quality.
