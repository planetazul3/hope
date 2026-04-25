# Hope Trading Engine Runbook

## Operational Guidelines

### 1. "One Trade at a Time" Enforcement
The engine manages exactly one active trade at a time.
-   When `InPosition`, new signals are ignored.
-   State transitions are strictly enforced via `src/fsm.rs`.
-   The transition from `Idle` to `OrderPending` is atomic upon signal generation.

### 2. Performance & Latency
The hot path (Tick -> Evaluation -> Signal) is zero-allocation.
-   All statistics are calculated in O(1).
-   Processing latency is logged in `tick_audit.log`.
-   The engine skips trades if tick processing exceeds `DERIV_MAX_TICK_LATENCY_MS`.

### 3. Safety & Security
-   **No Secret Logging**: Error payloads from Deriv are filtered to remove potential tokens or identifiers.
-   **Restrictive Permissions**: Audit logs are created with `0o600` permissions (Unix).
-   **API Slot Reliability**: `PermitGuard` ensures that the single API slot per tick is only used if the message is successfully sent.

## Maintenance & Commands

### Live Trading Engine
Start the engine in live or demo mode (controlled by `.env`):
```bash
make run
```
To audit with detailed WebSocket frames:
```bash
LOG_LEVEL=debug make run
```

### Strategy Backtesting
1.  **Prepare Data**: Export your latest collected ticks from SQLite to CSV:
    ```bash
    make export
    ```
2.  **Run Simulation**:
    ```bash
    make backtest
    ```
    The simulation uses the exact same `TickProcessor` and `StrategyEngine` as the live system.

### Verification & Consolidation
Run the full verification suite before any deployment:
```bash
make verify
```
Generate an auditable snapshot for AI analysis:
```bash
make consolidate
```

## Troubleshooting
-   **Invalid State Transition**: Indicates a race condition was blocked by the FSM. The engine resets to `Idle`.
-   **InternalQueueFull**: Outbound command channel is saturated. Check if the WebSocket connection is hanging.
-   **Insufficient History**: Transformer model needs more ticks before it can predict. The engine will skip signals until history is filled.
