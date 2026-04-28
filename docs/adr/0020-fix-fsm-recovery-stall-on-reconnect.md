# ADR 0020: Fix FSM Recovery Stall on Reconnect

## Context
When the trading engine loses its WebSocket connection while in an `OrderPending` state (awaiting a proposal or a buy confirmation), the Finite State Machine (FSM) transitions to the `Recovery` state. Upon reconnection, the engine resubscribes to all "tracked" contracts (those whose outcomes are not yet known).

If a contract was resolved (e.g., reached its expiry and was sold) during the period the engine was disconnected, the Deriv API immediately sends a `proposal_open_contract` update with `is_sold: 1` as soon as the subscription is active.

In the previous implementation, the `Recovery` state logic in `src/engine.rs` would see that `active_contract_id` was `None` (as it is reset during recovery initialization) and would buffer the `is_sold: 1` event as an "early close event," expecting a `BuyAccepted` message to eventually arrive and claim it. However, because the contract was already closed before the engine re-authorized, no `BuyAccepted` or `Proposal` message for that specific lifecycle would ever be received. This resulted in the FSM stalling in `Recovery` or `OrderPending` until a 15-second safety timeout forced a hard reset.

## Decision
We have modified the `TradeUpdate::OpenContract` handler in `src/engine.rs` to explicitly handle the `Recovery` state.

If the engine is in `Recovery` and receives an `OpenContract` update with `is_sold: 1` for a contract ID that exists in the `tracked_contracts` set, it now bypasses the event buffer and processes the contract closure immediately using `process_contract_closure`.

This allows the engine to:
1. Update the Risk Manager with the actual profit/loss of the contract.
2. Remove the contract from the tracking set.
3. Transition the FSM back to `Idle` (or `Cooldown`) immediately.

## Consequences
- **Improved Resilience**: The engine recovers from disconnects significantly faster when trades resolve during the downtime.
- **Data Integrity**: Risk management statistics remain accurate even if the engine misses the exact moment of contract closure.
- **State Determinism**: Prevents a known deadlock where the FSM waits for messages (`BuyAccepted`) that the upstream API will never send for already-resolved contracts.
- **Safety**: The logic for `OrderPending` buffering during normal live operation remains untouched, ensuring that "normal" race conditions (where an update arrives before the confirmation) are still handled correctly.
