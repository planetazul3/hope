# 0011: Gated TCN with Squeeze-and-Excitation

*   **Status**: Accepted
*   **Date**: 2026-04-24

## Context
The current Transformer V2 architecture, while competent, presents several challenges in the 1Hz tick-trading environment:
1. **Low SNR**: Transformers treat every tick equally in initial layers, making them prone to fitting noise in ultra-short (32-tick) sequences.
2. **Over-parameterization**: The multi-head attention mechanism introduces significant overhead for marginal gains over local temporal patterns.
3. **Inference Latency**: While acceptable, the quadratic complexity of attention scores is less efficient than convolutional alternatives.

## Decision
We will transition the primary ML backbone from a Transformer to a **Gated Temporal Convolutional Network (TCN)** with **Squeeze-and-Excitation (SE)** blocks.

### Key Architectural Changes:
1. **Causal Dilated Convolutions**: Replaces attention with dilated kernels (rates 1, 2, 4, 8) to capture multi-scale patterns with a fixed receptive field.
2. **Gating Mechanism**: Uses `Tanh` filter and `Sigmoid` gate (WaveNet-style) to learn non-linear feature selection, effectively acting as a learnable noise filter.
3. **Squeeze-and-Excitation (SE)**: Adds channel-wise attention to dynamically recalibrate feature maps based on market regime.
4. **Frequency-Domain Features**: Expands the input vector from 5 to 7 dimensions by adding HF (2-tick std) and LF (4-tick std) proxies.
5. **Multi-Task Learning**: Adds an auxiliary volatility prediction head during training to regularize the shared encoder.

## Consequences
* **Improved Noise Suppression**: The convolutional inductive bias is better suited for the low-SNR tick data.
* **Lower Latency**: TCNs map more efficiently to `tract-onnx` primitives, targeting <100µs inference.
* **Training Stability**: The reduced parameter count (approx. 30K vs 100K) reduces the risk of overfitting.
* **Compatibility**: All operations remain standard ONNX ops, ensuring no breakage in the Rust runtime.
