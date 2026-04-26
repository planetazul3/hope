# ADR 0002: Strict FSM for Trade Lifecycle

## Status
Accepted (Updated 2026-04-24)

## Context
The trading engine requires predictable state transitions to handle asynchronous WebSocket events without race conditions. Initially, an `Evaluating` state was included to represent the strategy processing time, but this introduced unnecessary ephemeral churn since strategy evaluation is now synchronous and extremely fast.

## Decision
We will use a simplified Finite State Machine (FSM) with the following states:

1.  **Idle**: The baseline state. Strategy evaluation occurs here.
2.  **OrderPending**: A signal has been generated, and we are waiting for a `proposal` quote or `buy` confirmation.
3.  **InPosition**: A contract is active. All new signals are ignored.
4.  **Cooldown**: A temporary state after a loss (or sequence of losses) where no evaluation occurs.
5.  **Recovery**: A transient safeguard state entered when an API error occurs while `OrderPending` or when the WebSocket disconnects mid-trade.

### Valid Transitions
-   `Idle` -> `OrderPending` (Signal generated)
-   `Idle` -> `Cooldown` (Manual or risk-based override)
-   `Idle` -> `Recovery` (Defensive guard from engine)
-   `OrderPending` -> `InPosition` (Buy confirmation received)
-   `OrderPending` -> `Idle` (Timeout or API error)
-   `OrderPending` -> `Recovery` (API error on active request)
-   `InPosition` -> `Idle` (Contract won/sold)
-   `InPosition` -> `Cooldown` (Contract lost/sold with risk trigger)
-   `InPosition` -> `Recovery` (Disconnect while holding contract)
-   `Recovery` -> `Idle` (Clean reset after state verification)
-   `Recovery` -> `InPosition` (Contract confirmed still open on reconnect)
-   `Cooldown` -> `Idle` (Cooldown timer expired)

## Consequences
-   **No Ephemeral Churn**: Removing `Evaluating` reduces the number of state transitions per tick.
-   **Atomic Signals**: Signals transition the system directly from `Idle` to `OrderPending`.
-   **Idempotency**: The FSM allows (and ignores) transitions to the current state to prevent runtime errors during rapid market movements.
