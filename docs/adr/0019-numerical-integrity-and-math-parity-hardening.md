# ADR 0019: Numerical Integrity and Math Parity Hardening

## Status
Accepted

## Context
During the exhaustive April 2026 multi-dimensional audit, two critical mathematical risks were identified:
1. **Normalization Discrepancy**: The Rust engine (`transformer.rs`) used a redundant epsilon offset when normalizing Wavelet (DB2) coefficients compared to the Python training pipeline (`common.py`). In low-price regimes, this led to a factor-of-2 difference in input features, causing inference bias.
2. **Floating-Point Accumulation Drift**: The `TickProcessor` used an incremental O(1) update logic for sums and squared sums. Over hundreds of thousands of ticks, rounding errors in floating-point subtractions could accumulate, leading to drift in volatility and drift metrics.
3. **Ghost Trades on Noise**: The strategy filter could be bypassed in zero-volatility markets if the return magnitude calculation drifted slightly above zero.

## Decision
1. **Math Parity**:
    - Aligned the normalization formula in `src/transformer.rs` to match `scripts/hope_ml/common.py` exactly: `db2_a1 / (price + 1e-8)`.
2. **Periodic Stats Reset**:
    - Implemented a `recalculate_sums` method in `TickProcessor`.
    - The engine now triggers a full re-summing of the ring buffer every 1,000 ticks to clear any accumulated floating-point error.
3. **Explicit Volatility Gating**:
    - Added an explicit check in `src/strategy.rs` to block any signal if `volatility < 1e-9`, preventing execution on numerical noise.
4. **Data Export Fidelity**:
    - Standardized `epoch` as `INTEGER` in the database schema and migration tools.
    - Added CSV headers to the export tool for improved interoperability.

## Consequences
- **Accuracy**: Inference features are now bit-perfect matches for the training distribution.
- **Stability**: Long-running engine sessions will maintain precise statistical metrics indefinitely.
- **Safety**: Execution is protected against edge-case noise in flat markets.
- **Traceability**: Exported datasets are now self-documenting with headers.
