# ADR 0001: WebSocket-First Event-Driven Engine

- Status: Accepted
- Date: 2026-04-24

## Context

The trading system requires real-time tick handling, deterministic execution flow, and stable session state with Deriv. Reconnecting per request or relying on polling would increase latency, reduce auditability, and create state drift.

## Decision

Use a single persistent Deriv WebSocket connection as the primary runtime transport. Route inbound messages into an internal event-driven engine that processes ticks, trade updates, and API errors as typed events.

## Consequences

- Tick handling and trade updates share one connection lifecycle.
- Reconnect and resubscription logic must be explicit.
- The engine can remain transport-aware without mixing transport concerns into strategy logic.
- REST is excluded from the trading loop.
