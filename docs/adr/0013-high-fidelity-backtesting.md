# ADR 0013: High-Fidelity Backtesting Methodology

## Status
Accepted

## Context
The initial backtesting implementation was overly simplistic: it closed trades on the immediate next tick and ignored system cooldowns. This led to inflated trade counts and unrealistic win-rate/profitability projections that did not reflect the real-world behavior of the `Engine` and the Deriv contract lifecycle.

## Decision
We have refactored the backtest binary (`src/bin/backtest.rs`) to implement high-fidelity simulation:
1.  **State-Aware Simulation**: The backtest now utilizes the `TradingFsm` and `RiskManager` structs from the core library, ensuring state transitions match the production engine.
2.  **Contract Duration**: Trades are held for exactly `DERIV_DURATION_TICKS` as specified in the configuration, rather than closing on the next tick.
3.  **System Cooldown**: The `Cooldown` state and `DERIV_COOLDOWN_TICKS` are now respected, preventing the strategy from entering new trades while the system is in protection mode.
4.  **Realistic Payouts**: Profit/loss calculations use a standard 95% payout for wins and 100% loss for the stake, providing a conservative estimate of profitability.

## Consequences
*   **Realistic Projections**: Backtest results now provide a much more accurate representation of how a strategy will perform in the live environment.
*   **Lower Trade Frequency**: Observed trade counts in backtests are significantly lower (approximately 50% reduction in typical runs) as they account for duration and cooldown occupancy.
*   **Architectural Alignment**: The backtest serves as a secondary validation of the `src/lib.rs` modules, ensuring that changes to the FSM or RiskManager are immediately reflected in simulation.
