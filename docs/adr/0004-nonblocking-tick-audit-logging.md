# ADR 0004: Nonblocking Tick Audit Logging

- Status: Accepted
- Date: 2026-04-24

## Context

Per-tick logging is mandatory for observability, but synchronous file I/O inside the tick loop would directly increase decision latency and risk missed timing targets.

## Decision

Emit per-tick audit records through a bounded, nonblocking handoff to a dedicated logging thread. If the logging channel is full, drop the audit line and emit a warning rather than stalling tick processing.

## Consequences

- Trading-loop latency remains prioritized over perfect log completeness.
- Audit output is still structured and append-only.
- Backpressure in logging becomes visible through warnings instead of hidden latency growth.
