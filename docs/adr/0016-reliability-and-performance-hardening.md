# ADR 0016: Reliability and Performance Hardening

## Status
Accepted

## Context
Following a comprehensive project audit, several critical issues were identified:
1.  **Latency**: 1-tick delay in trade execution due to rate limiter constraints.
2.  **Reliability**: Silent event drops in WebSocket client under high load.
3.  **Correctness**: Off-by-one errors in `TickProcessor` ring buffer logic.
4.  **Integrity**: Potential data loss in `TickLogger` and `export_db.py` during shutdown or large exports.

## Decision
We implemented a series of hardening measures:
1.  **Rate Limiter Bypass**: Introduced a `bypass_rate_limit` flag in `ExecutionEngine` to allow immediate buy execution after receiving a proposal, eliminating the 1-tick latency.
2.  **Decoupled WebSocket Writes**: Decoupled socket writes into a dedicated task to prevent blocking the read loop.
3.  **Enforced Backpressure**: Replaced `try_send` with async `send().await` for WebSocket events to ensure financial state updates are never dropped.
4.  **Graceful Shutdown**: Implemented `Drop` traits and thread joining in `TickLogger` to ensure all pending data is flushed to disk.
5.  **Math Fixes**: Corrected `TickProcessor` index calculation and stats normalization.

## Consequences
- **Positive**: Significantly lower execution latency and higher financial reliability.
- **Positive**: Data integrity is guaranteed for audit logs and parquet exports.
- **Neutral**: The WebSocket read loop may now experience backpressure if the engine is slow, which is desirable for consistency.
