# Runbook

## Prerequisites

- Rust toolchain with Cargo
- A `.env` file in the repository root
- Valid Deriv token for the selected environment

## Required Environment Variables

- `DERIV_ENVIRONMENT`: `DEMO` or `REAL`
- `DERIV_DEMO_TOKEN`: token used when `DERIV_ENVIRONMENT=DEMO`
- `DERIV_REAL_TOKEN`: token used when `DERIV_ENVIRONMENT=REAL`
- `DERIV_APP_ID`: Deriv application ID
- `DERIV_TRADING_ENABLED`: `true` or `false` (defaults to `false` for safety)

## Optional Runtime Variables

- `DERIV_SYMBOL`
- `DERIV_CONTRACT_TYPE`
- `DERIV_CURRENCY`
- `DERIV_STAKE`
- `DERIV_DURATION_TICKS`
- `DERIV_THRESHOLD`
- `DERIV_RECONNECT_BACKOFF_MS`
- `DERIV_MIN_API_INTERVAL_MS`
- `DERIV_COOLDOWN_TICKS`
- `DERIV_MAX_TICK_LATENCY_MS`
- `DERIV_INBOUND_QUEUE_CAPACITY`
- `DERIV_OUTBOUND_QUEUE_CAPACITY`
- `LOG_LEVEL`

## Local Commands

- `cargo fmt`: format the codebase
- `cargo check --offline`: compile-check the crate using the locked dependency set
- `cargo test --offline`: run unit and integration tests using the locked dependency set
- `cargo run`: start the trading system with `.env` configuration
- `python3 consolidate_project_sources.py`: generate a timestamped audit snapshot of source, docs, config, and notebook contents
- `make consolidate`: optional shorthand for the same audit snapshot command

## Audit Snapshot Tool

The repository includes `consolidate_project_sources.py` as a standalone audit utility. It:

- consolidates source, docs, config, and notebook content into a single timestamped text file
- excludes build artifacts, databases, model binaries, logs, and other non-source outputs
- includes `notebooks/train_transformer.ipynb` content for audit purposes and annotates it with the related local training script at `scripts/train.py`

## Startup Behavior

On startup the system:

1. Loads `.env`
2. Connects to Deriv over WebSocket
3. Authorizes using the token for the selected environment
4. Subscribes to ticks for the configured symbol
5. Begins deterministic tick processing and audit logging

## Runtime Outputs

- Structured logs are emitted to stdout.
- Tick audit lines are appended to `tick_audit.log`.

## Live Validation
1. Set `DERIV_ENVIRONMENT=DEMO`.
2. Set `DERIV_TRADING_ENABLED=false`.
3. Run the bot: `cargo run`.
4. Observe the logs. You should see "signal detected", followed by "proposal requested", and then "trading disabled; skipping buy".
5. Once you are confident in the signals and timing, set `DERIV_TRADING_ENABLED=true` to allow demo trades.

## Expected Event Flow for a Trade
1. **Tick Processing**: Bot receives ticks and updates internal state.
2. **Strategy Evaluation**: Bot detects a signal based on the `DERIV_THRESHOLD`.
3. **Proposal Request**: Bot sends a `proposal` request to Deriv to get a quote.
4. **Proposal Received**: Bot receives a `proposal` response with an `id`.
5. **Buy Execution**: Bot sends a `buy` request using the proposal `id`.
6. **Buy Confirmation**: Bot receives a `buy` response confirming the contract.
7. **Open Contract Monitoring**: Bot subscribes to updates for the open contract.
8. **Trade Closed**: Bot receives a final `proposal_open_contract` update when the trade is sold.
9. **Cooldown**: Bot enters a cooldown period if configured or after losses.

## Operational Readiness

### Monitoring Expectations
*   **Disconnects**: The system will automatically attempt to reconnect. Monitor logs for `websocket disconnected` and `connected`.
*   **API Errors**: `req_id` is now included in all API error logs to correlate with specific requests.
*   **Dropped Audit Lines**: If the `tick_audit.log` channel is full, a warning will be logged. Ensure high I/O availability.
*   **Cooldown Events**: Monitor for `entered cooldown` warnings, which indicate the risk manager has triggered a circuit breaker.

### Operator Guidance

#### Emergency Kill Switch
*   **Immediate Halt**: Send `SIGINT` (Ctrl+C) to the process. The system will stop sending new requests immediately.
*   **Safety Lock**: Ensure `DERIV_TRADING_ENABLED=false` in `.env` to prevent any accidental order placement during maintenance.

#### Recovery and Safe Restart
1.  **Stop the bot**: `Ctrl+C`.
2.  **Verify State**: Check Deriv dashboard for any open contracts. The bot does not persist state across restarts yet; orphaned contracts must be managed manually if the bot was in `InPosition` state.
3.  **Check Logs**: Review `tick_audit.log` for the last few trades to understand the reason for failure or cooldown.
4.  **Restart**: Run `cargo run` after ensuring the environment is stable.

#### Audit Completeness
*   Every tick processed is logged to `tick_audit.log` with:
    *   Timestamp and Price.
    *   FSM State and Strategy Decision.
    *   Processing Latency (ms).
    *   Probability Model output.
