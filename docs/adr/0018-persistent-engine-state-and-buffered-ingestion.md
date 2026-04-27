# ADR 0018: Persistent Engine State and Buffered Ingestion

## Status
Accepted

## Context
Project `hope` operates as a long-running trading engine. Two critical issues were identified during the April 2026 audit:
1. **Volatile State Loss**: The Rust engine maintained active trade IDs and FSM states only in memory. A process crash or restart during an open trade would result in the loss of tracking for that trade, potentially leading to duplicate positions and risk management violations.
2. **Synchronous Ingestion Bottleneck**: The Python `tick_collector.py` performed a synchronous SQLite `COMMIT` for every single tick in live mode. Under high symbol loads, this caused significant disk I/O pressure and potential ingestion delays.

## Decision
1. **Engine State Persistence**:
    - Implement a `state.json` file in the project root to persist the `active_contract_id` and the current `TradingState`.
    - The Rust engine will save its state on every transition attempt (even if the transition is rejected) to ensure disk/memory synchronization during complex recovery sequences.
    - On startup, the engine will attempt to load `state.json`. If an active contract is found, it will automatically resubscribe and transition to the appropriate state (usually `InPosition` or `Recovery`).
2. **Buffered Tick Ingestion**:
    - Introduce a `write_buffer` in `tick_collector.py` for live mode.
    - Ticks are collected into the buffer and flushed to SQLite every 1 second or every 100 ticks, whichever comes first.
    - Use a dedicated `ThreadPoolExecutor` for database operations to prevent saturating the main event loop's executor.

## Consequences
- **Robustness**: The system can now survive restarts and crashes without losing track of open positions.
- **Performance**: Disk I/O during live collection is reduced by up to 99% (from 1 commit/tick to 1 commit/second).
- **Complexity**: Added a new dependency on `serde` (already present) and introduced a small (1s) persistence delay for live data. This delay is acceptable as the engine subscribes directly to the WebSocket and does not rely on the collector's SQLite database for real-time execution.
- **Maintenance**: A new `state.json` file is managed by the engine.
