# ADR 0007: Neural Inference Integration

- Status: Accepted
- Date: 2026-04-24

## Context

The current `GaussianModel` (ADR 0006) provides a parametric estimate of continuation probability based on rolling volatility and drift. While statistically sound, it assumes price action follows a simple Brownian motion process, which fails to capture complex micro-patterns like momentum exhaustion or regime-dependent reversals.

We aim to move toward a **Transformer-based sequence model** that can learn probabilistic patterns from raw tick features.

## Decision

Integrate a neural inference runtime into the Rust engine to support non-parametric probability models.

1.  **Feature Set**: The model will consume a sequence of features per tick:
    - Direction (1, 0, -1)
    - Return Magnitude (absolute change)
    - Momentum Streak (count of consecutive ticks in same direction)
    - Ticks Since Reversal (reset on direction change)
    - Rolling Volatility
2.  **Runtime**: Use a lightweight, zero-dependency (if possible) or C-linked runtime like `tract` (pure Rust) or `onnxruntime` to execute exported `.onnx` models.
3.  **Trait Implementation**: The inference logic will be encapsulated in a `TransformerModel` that implements the `ProbabilityModel` trait defined in `src/strategy.rs`.

## Consequences

- **Latency**: Neural inference adds a non-trivial computational cost per tick (aiming for <1ms).
- **Complexity**: The engine now depends on an external model file and a neural runtime.
- **Precision**: The model should provide higher edge by learning non-linear dependencies in tick sequences.
- **Portability**: The model must be trained externally (Python/PyTorch) and exported to ONNX.

## Requirements for TickProcessor

To support this, `TickProcessor` must be extended to track "Return Magnitude" and "Ticks Since Reversal" in its sliding window snapshots.
