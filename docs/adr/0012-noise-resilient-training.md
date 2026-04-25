# ADR 0012: Noise-Resilient Training with DWT and Contrastive Pre-training

## Status
Accepted

## Context
1HZ tick data is dominated by microstructure noise, leading to near-random-walk behavior and low ROC-AUC (0.50) when using standard supervised learning. Previous iterations of the Transformer and GatedTCN models struggled to isolate structural trends from transient jitter.

## Decision
We have upgraded the training pipeline to **GatedTCN V4**, implementing the following noise-resilience strategies:

1.  **DWT (Haar Wavelet) Features**: Replaced simple rolling standard deviations with Level-1 Haar Wavelet decomposition (Approximation A1 and Detail D1 coefficients). This provides better time-frequency localization.
2.  **Contrastive Pre-training (SSL)**: Introduced a Phase 1 pre-training stage using a contrastive InfoNCE loss over augmented (jittered) views of the tick sequences. This forces the model to learn scale-invariant market regimes before supervised fine-tuning.
3.  **Multi-task Loss Weighting**: Adjusted the auxiliary volatility loss weight to 0.2 to act as a stronger regularizer for the directional classification task.
4.  **Vectorized DWT in Rust**: Implemented the corresponding Haar Wavelet logic in `src/transformer.rs` to ensure feature parity during live inference.

## Consequences
*   **Regime Recognition**: The model is forced to cluster similar market states (e.g., high-volatility trending) rather than memorizing individual price jumps.
*   **Inference Parity**: Feature calculation in Rust is deterministic and remains $O(N)$, preserving the sub-millisecond latency budget.
*   **Training Curriculum**: The two-phase training loop increases pre-training time but results in more stable latent representations.
