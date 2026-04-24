# ADR 0002: Strict FSM for Trade Lifecycle

- Status: Accepted
- Date: 2026-04-24

## Context

Trading systems fail dangerously when state changes are implicit or inferred from scattered conditions. This repository needs deterministic, auditable transitions that prevent invalid order flow.

## Decision

Represent trade lifecycle control with an explicit FSM containing `Idle`, `Evaluating`, `OrderPending`, `InPosition`, and `Cooldown`. All decision-making and execution gating must pass through this FSM, and invalid transitions must be rejected.

## Consequences

- Execution is only legal from defined states.
- State desynchronization can be detected and reset safely.
- Future features such as exits, cancels, or portfolio sync must integrate through explicit transitions instead of ad hoc flags.
