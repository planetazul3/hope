# ADR 0012: Noise-Resilient Training with DWT and Contrastive Pre-training

## Status
Accepted

## Context
1HZ tick data is dominated by microstructure noise, leading to near-random-walk behavior and low ROC-AUC (0.50) when using standard supervised learning. Previous iterations of the Transformer and GatedTCN models struggled to isolate structural trends from transient jitter.

## Decision
We have upgraded the training pipeline to **GatedTCN V4**, implementing the following noise-resilience strategies:

1.  **DWT (Haar Wavelet) Features**: Replaced simple rolling standard deviations with Level-1 Haar Wavelet decomposition (Approximation A1 and Detail D1 coefficients). This provides better time-frequency localization.
2.  **Contrastive Pre-training (SSL)**: Introduced a Phase 1 pre-training stage using a contrastive InfoNCE loss over augmented views. We use a **Block Masking** strategy (randomly zeroing out contiguous segments) instead of Gaussian noise to force the model to learn temporal dependencies and robust representations.
3.  **Multi-task Loss Weighting**: Adjusted the auxiliary volatility loss weight to 0.2 to act as a stronger regularizer for the directional classification task.
4.  **Vectorized DWT in Rust**: Implemented the corresponding Haar Wavelet logic in `src/transformer.rs` to ensure feature parity during live inference, including scale-invariant A1 normalization.
5.  **INT8 Dynamic Quantization**: Implemented post-export dynamic INT8 quantization for ONNX models to optimize L1/L2 cache residency and reduce inference latency by ~50% without meaningful precision loss.
6.  **Learning Rate Warmup**: Integrated a 5-epoch linear warmup schedule at the start of Phase 2 fine-tuning to stabilize the training of newly initialized heads.

## Consequences
*   **Regime Recognition**: The model is forced to cluster similar market states (e.g., high-volatility trending) rather than memorizing individual price jumps.
*   **Inference Parity**: Feature calculation in Rust is deterministic and remains $O(N)$, preserving the sub-millisecond latency budget.
*   **Training Curriculum**: The two-phase training loop increases pre-training time but results in more stable latent representations.
