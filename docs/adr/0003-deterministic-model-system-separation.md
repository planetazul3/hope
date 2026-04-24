# ADR 0003: Deterministic Model/System Separation

- Status: Accepted
- Date: 2026-04-24

## Context

The system must continue to behave correctly even if the probability model changes. Mixing model behavior with transport, risk, or order control would make the engine harder to reason about and audit.

## Decision

Separate the deterministic system layer from the probabilistic model layer. The system owns connection management, tick processing, FSM transitions, execution control, risk limits, and logging. The model only produces probability estimates consumed by strategy evaluation.

## Consequences

- Model replacement does not require redesigning the execution engine.
- Operational safeguards remain deterministic and testable.
- Strategy logic can evolve without weakening system invariants.
