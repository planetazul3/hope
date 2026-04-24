# Blueprint

## Purpose

This blueprint defines the intended production shape of `hope`. It is the stable reference for system boundaries, development priorities, and what “done” looks like beyond the current implementation.

## Product Goal

Build a deterministic, auditable, low-latency trading system for Deriv synthetic indices that can safely process ticks, make bounded decisions, execute through a strict FSM, and remain correct even when the model changes.

## System Blueprint

### 1. Market Connectivity

- Single persistent Deriv WebSocket connection
- Explicit authorization and subscription lifecycle
- Automatic reconnect and deterministic resubscription
- Typed routing for ticks, trade updates, and errors

### 2. Deterministic Core

- Fixed-size tick memory
- Strict FSM for the full trade lifecycle
- One decision per tick
- One API call per tick maximum
- Explicit rate limiting and latency guards

### 3. Strategy Boundary

- Strategy consumes normalized tick state
- Model remains a replaceable probability provider
- System invariants never depend on probabilistic behavior

### 4. Execution and Risk

- Proposal, buy, and open-contract handling remain transport-correct
- No overlapping trades unless explicitly enabled by design
- Consecutive-loss protection and cooldown remain mandatory
- Safe reset behavior on API errors or state desync

### 5. Observability and Operations

- Nonblocking structured logs
- Per-tick audit trail
- Operator-facing runbook and configuration docs
- Clear recovery steps for disconnects, API failures, and invalid state

## Definition of Production Readiness

The system is only production-ready when it has:

- validated live-demo behavior against the documented Deriv flow
- deterministic state handling across reconnects
- enough tests to cover the FSM, execution gating, and risk controls
- complete operational docs for startup, monitoring, and failure handling
- architecture and integration decisions recorded in ADRs
