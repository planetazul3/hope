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

## Current Verification Limitation

If the local environment cannot fetch missing Cargo dependencies, `cargo check` and `cargo test` may fail before compilation starts. In that case, restore dependency resolution first and rerun verification before shipping changes.
