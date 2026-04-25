# 0008: ML Training Pipeline Optimizations

*   **Status**: Accepted
*   **Date**: 2024-05-20

## Context

The initial Transformer V2 training pipeline was functional but lacked robust training controls. As market data is inherently noisy and often imbalanced (buy vs sell pressure), the model was prone to overfitting or failing to converge optimally. Additionally, the integration with the Rust `tract` engine benefited from strictly defined tensor shapes.

## Decision

We have implemented a suite of optimizations in the ML training pipeline (`scripts/train.py` and `notebooks/train_transformer.ipynb`):

1.  **Early Stopping & LR Scheduling**: Introduced `ReduceLROnPlateau` to dynamically adjust learning rates and an early stopping mechanism to preserve the best model weights based on validation loss.
2.  **Class Weight Balancing**: Implemented manual weighting in the Binary Cross-Entropy loss to handle market direction imbalances.
3.  **Vectorized Feature Engineering**: Refactored volatility calculations using `pandas` rolling windows to replace inefficient Python loops, improving preprocessing speed.
4.  **Static ONNX Export**: Switched from dynamic batch axes to a fixed batch size of 1. This ensures perfect alignment with the Rust `tract` inference engine's expected input shapes and prevents runtime rank mismatches.
5.  **Dynamic Backtesting**: Updated the backtest engine to load `AppConfig` and use the configured `transformer_sequence_length`, ensuring the simulation matches the production runtime.

## Consequences

*   **Stability**: Training is more predictable and less likely to diverge.
*   **Performance**: Preprocessing is significantly faster.
*   **Safety**: Static ONNX models reduce the risk of runtime panics in the production trading loop.
*   **Accuracy**: The backtester now provides a high-fidelity simulation by sharing the same configuration as the live engine.
