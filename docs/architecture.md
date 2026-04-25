# Architecture

## Overview

`hope` is a WebSocket-first, event-driven trading system for Deriv synthetic indices. The system is structured as a Rust library (`src/lib.rs`) with dedicated binaries for the live trading engine (`src/main.rs`) and strategy backtesting (`src/bin/backtest.rs`).

## Runtime Flow

1.  **Transport**: `src/websocket_client.rs` maintains a single persistent Deriv WebSocket connection with automatic reconnection logic.
2.  **Event Routing**: Incoming WebSocket messages are routed into the engine as typed events (Tick, TradeUpdate, ApiError).
3.  **Tick Processing**: `src/tick_processor.rs` maintains a fixed-size ring buffer of 64 entries. It computes volatility and drift in O(1) time using incremental running sums over a configurable `VOLATILITY_WINDOW`.
4.  **State Management**: `src/fsm.rs` enforces explicit trade lifecycle transitions: `Idle`, `OrderPending`, `InPosition`, and `Cooldown`.
5.  **Strategy Evaluation**: `src/strategy.rs` evaluates probability models (e.g., Gaussian or Transformer) during the `Idle` state. Signals are only generated when specific thresholds and streaks (e.g., 2 ticks in the same direction matching the trend) are met.
6.  **Execution Control**: `src/execution.rs` manages API rate limiting and per-tick call slots using a `PermitGuard` to ensure reliable command delivery.
7.  **Risk Management**: `src/risk.rs` tracks consecutive losses and sessions metrics, triggering a cooldown state after three losses.
8.  **Audit Logging**: `src/tick_logger.rs` writes high-resolution audit records asynchronously with restrictive file permissions (`0o600`).

## Model Architecture

### GatedTCN V4 (Noise-Resilient Learning)
The engine utilizes a Gated Temporal Convolutional Network (V4) to identify micro-patterns in tick sequences while suppressing microstructure noise.
- **Sequence Window**: 32 ticks (~30-45 seconds of market data).
- **Features**: 8-dimensional input per tick (5 base features + 2 Haar Wavelet DWT coefficients A1/D1).
- **Structure**: 4-layer Causal Dilated Convolutions with Squeeze-and-Excitation (SE) channel attention.
- **Training**: Two-phase curriculum:
    1. **Contrastive Pre-training**: Learns noise-resilient representations via InfoNCE loss on jittered views.
    2. **Supervised Fine-tuning**: Optimizes directional classification (Focal Loss) and volatility prediction (MSE).
- **Workflow**: Vectorized preprocessing and DWT decomposition ensuring sub-100µs inference latency.

## Core Boundaries

-   **System Layer**: Handles transport, tick buffering, FSM transitions, execution gating, risk controls, and logging.
-   **Model Layer**: Provides probability estimations. The system supports any model satisfying the `ProbabilityModel` trait.

## Data & Simulation

-   **Live Engine**: Consumes real-time WebSocket data.
-   **Backtest Engine**: Consumes historical ticks from CSV (exported from SQLite via `scripts/export_db.py`).
-   **Structure**: Both engines share the same core logic modules via the `hope` library crate.

## Operational Constraints

-   No dynamic heap allocations in the hot path (Tick -> Strategy -> Signal).
-   Strict "One Trade at a Time" enforcement at the FSM level.
-   O(1) complexity for all per-tick statistical calculations.
-   Secure handling of API errors (no raw payloads logged).
