# 0017: Production Hardening via Canonical Transformer, TS2Vec, and Daubechies Wavelets

**Status**: Accepted

## Context

The previous ML architecture relied on GatedTCN V4 with Haar wavelets and block masking augmentation. While computationally efficient, this topology was mathematically sub-optimal for isolating true structural trends from 1Hz microstructure noise. Block masking forced the network to memorize noise topology rather than understanding the underlying state representations. Furthermore, dynamic allocations in the Rust hot path during inference posed a risk to deterministic latency constraints.

## Decision

We have decided to rigorously align the system to a production-grade blueprint emphasizing deterministic Rust execution, ONNX graph optimizations, and hierarchical self-supervised representation learning:

1.  **Architecture Reversion**: Revert to the Canonical Causal Transformer Encoder ($L=32$, $O(L^2)$ mathematical optimality) with a prepended `[CLS]` token for instance-level aggregation, feeding two distinct linear heads (Direction and Volatility).
2.  **TS2Vec Pre-training**: Replace raw block masking with TS2Vec (Hierarchical Contrastive Learning) applying contextual consistency using random cropping and timestamp masking strictly applied in the latent space with InfoNCE loss.
3.  **Feature Engineering**: Upgrade the Discrete Wavelet Transform (DWT) from Haar (db1) to Daubechies db2 (4-tap) filter bank implemented as $O(N)$ FIR filters avoiding FFTs.
4.  **Hardware Constraints & Security**: Force a static batch size of 1 and sequence length of 32 for ONNX export, coupled with dynamic INT8 Quantization (`QuantType.QInt8`). Enforce cryptographic model signing via Ed25519 before runtime execution.
5.  **Rust Determinism**: Ensure the `tract-onnx` execution environment operates on a strictly Zero-Allocation Hot Path using pre-allocated `ndarray` buffers.

## Consequences

-   **Supersedes**: ADR 0011 (GatedTCN Architecture).
-   **Amends**: ADR 0012 (Noise-Resilient Training).
-   **Improvements**: Highly improved deterministic execution, robust latent representation resilient to microstructure noise, and guaranteed system integrity via cryptographic signing and strict memory management.
