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

7. **Gradient Clipping**: Added norm clipping (1.0) to prevent exploding gradients during Transformer training.
8. **Label Smoothing**: Integrated smoothing (0.05) into Focal Loss to reduce model overconfidence.
9. **Native Batching**: Switched to PyTorch `DataLoader` and `TensorDataset` for standardized data handling.
10. **Rich Metrics**: Expanded validation monitoring to include Accuracy, Precision, Recall, and F1-Score.

## Consequences

*   **Robustness**: The model is less likely to overfit due to data augmentation, causal constraints, and label smoothing.
*   **Stability**: Gradient clipping ensures more stable convergence during training.
*   **Flexibility**: Operators can now tune momentum rewards, volatility penalties, and return ratios via environment variables (`STRATEGY_MOMENTUM_REWARD`, `STRATEGY_VOLATILITY_PENALTY`, `STRATEGY_MIN_RETURN_RATIO`) without recompiling the code.
*   **Visibility**: Training logs now include a comprehensive suite of classification metrics, providing deeper insight into model performance.
