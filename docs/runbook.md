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

### Strategy Backtesting & Training
1.  **Collect Data**: Fetch historical ticks from Deriv API:
    ```bash
    make collect
    ```
    *Note: You can stop this gracefully at any time with Ctrl+C.*

2.  **Prepare Data**: Export your latest collected ticks from SQLite to CSV:
    ```bash
    make export
    ```
2.  **Train Model** (Optional): Upload `data/ticks.csv` to a cloud platform like Google Colab or Kaggle and run the training notebook using `notebooks/colab_training.ipynb` or `notebooks/kaggle_training.ipynb`.
3.  **Run Simulation**:
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

## Model Configuration

### Canonical Causal Transformer (Production Hardened)
The system defaults to a Canonical Causal Transformer Encoder with Multi-Task Learning.
- **Requirement**: `TRANSFORMER_SEQUENCE_LENGTH` in `.env` must be set to **32**.
- **Features**: 8-dimensional input including Daubechies (db2) FIR filter bank Approximation and Detail coefficients.
- **Inference**: Handled by `tract` in Rust. Executes a static graph (1x32x8) with dynamic INT8 Quantization.
- **Optimization**: Zero-allocation hot path with pre-allocated buffers for sub-ms latency.

### Strategy Thresholds
The `StrategyEngine` uses dynamic modifiers that can be tuned in `.env`:
- `STRATEGY_MOMENTUM_REWARD`: Reduction in threshold when `streak >= 4` (Default: 0.02).
- `STRATEGY_VOLATILITY_PENALTY`: Increase in threshold when volatility is low (Default: 0.05).
- `STRATEGY_MIN_RETURN_RATIO`: Minimum return magnitude as a ratio of volatility (Default: 0.1).

## Troubleshooting
-   **Invalid State Transition**: Indicates a race condition was blocked by the FSM. The engine resets to `Idle`.
-   **InternalQueueFull**: Outbound command channel is saturated. Check if the WebSocket connection is hanging.
-   **Insufficient History**: Transformer model needs more ticks before it can predict. The engine will skip signals until history is filled.
-   **No Signal / Stuck Terminal**: The engine periodically logs a heartbeat every 30 ticks to the console (e.g., "monitoring market... waiting for signal") indicating it is still alive. If trades are not firing, check the `reason` field in the log or the `tick_audit.log` for skip reasons like `Short Trend`, `Low Volatility`, or `Below Threshold`.

