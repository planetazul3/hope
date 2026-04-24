# Development Roadmap

## Usage

This roadmap is the active development tracker for `hope`. Before starting work, identify the current stage, continue from the next unfinished or partially finished item, and update this file when a stage materially changes.

Status values:

- `Done`
- `In Progress`
- `Planned`

## Current Stage Summary

- Current active stage: `Stage 7 - Neural Inference Evolution` (Complete)
- Overall state: Production-grade sovereign trading engine with Gaussian and Transformer-based probability models.

## Stage 1 - Deterministic Core Foundation

Status: `Done`

- Single persistent WebSocket client implemented
- Fixed-size tick processor implemented
- Strict FSM implemented
- Placeholder deterministic model integrated
- Execution rate limiting and one-call-per-tick guard implemented
- Cooldown-based risk manager implemented
- Nonblocking tick audit logging implemented

## Stage 2 - Documentation and Architecture Controls

Status: `Done`

- Contributor guide added
- Docs baseline added
- ADR workflow added
- Official Deriv reference linked
- Legacy Deriv API choice documented and protected by ADR 0005

## Stage 3 - Live Integration Hardening

Status: `Done`

- Validate real message flow for `authorize`, `proposal`, `buy`, and `proposal_open_contract`
- Confirm proposal-to-buy correlation behavior under live/demo traffic
- Add a safe trading enablement flag so startup does not imply live order placement by default
- Harden state handling for stale proposals, delayed buy responses, and reconnect with an open contract
- Extend runbook with live-demo validation steps and expected event flow

## Stage 4 - Test Expansion and Failure Simulation

Status: `Done`

- Add engine-level tests for `OrderPending`, `InPosition`, and `Cooldown` transitions
- Add tests for API error recovery and safe reset behavior
- Add tests for reconnect and resubscription with tracked contracts
- Add tests for latency-based trade skipping and rate-limit enforcement across ticks

## Stage 5 - Operational Readiness

Status: `Done`

- Add request correlation identifiers across proposal, buy, and contract updates
- Define monitoring expectations for disconnects, dropped audit lines, and cooldown events
- Add operator guidance for kill switch, recovery, and safe restart procedures
- Review log schema and operational outputs for audit completeness

## Stage 6 - Strategy Evolution

Status: `Done`

- Replace the constant probability placeholder only after Stage 3 and Stage 4 are stable
- Keep model replacement isolated from transport, FSM, execution, and risk code
- Record any meaningful model integration or feature-store decision in a new ADR

## Stage 7 - Neural Inference Evolution

Status: `Done`

- Extend feature extraction in `TickProcessor` to support sequence-based models (return magnitude, reversal timing)
- Document neural inference architecture and requirements in ADR 0007
- Integrate a lightweight neural runtime (e.g., `tract` or `onnxruntime`) for Transformer inference
- Implement `TransformerModel` satisfying the `ProbabilityModel` trait
- Validate inference latency and resource usage under live tick load
- Hardened Transformer inference with safe indexing and robust ONNX export (Stage 7 Complete)
