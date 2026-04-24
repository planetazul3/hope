# Architecture

## Overview

`hope` is a WebSocket-first, event-driven trading system for Deriv synthetic indices. The system is designed so the execution engine remains deterministic even if the probability model changes.

## Runtime Flow

1. `src/websocket_client.rs` maintains a single persistent Deriv WebSocket connection.
2. Tick messages are routed into the engine as typed events.
3. `src/tick_processor.rs` computes tick direction and streak while maintaining a fixed-size ring buffer of 64 entries.
4. `src/fsm.rs` enforces explicit trade lifecycle transitions: `Idle`, `Evaluating`, `OrderPending`, `InPosition`, and `Cooldown`.
5. `src/strategy.rs` evaluates the deterministic placeholder model and emits signals only during `Evaluating`.
6. `src/execution.rs` enforces one API call per tick, minimum API spacing, and latency-based trade skipping.
7. `src/risk.rs` tracks consecutive losses and moves the system into cooldown after three losses.
8. `src/tick_logger.rs` writes per-tick audit records without blocking the trading loop.

## Core Boundaries

- System layer: transport, tick processing, FSM, execution control, risk, and logging.
- Model layer: probability estimation only. The current placeholder model always returns `0.6`.

## Connection Model

- Legacy Deriv WebSocket endpoint: `wss://ws.derivws.com/websockets/v3?app_id=...`
- Authorization is performed on the socket using the configured token.
- Reconnect and resubscription are handled by the WebSocket client.
- This legacy API choice is intentional; see ADR 0005 before changing transport or auth flow.

## Operational Constraints

- No REST calls in the trading loop.
- No per-request reconnects.
- No concurrent overlapping trades.
- No implicit state transitions.
- No dynamic growth in the tick history buffer.
