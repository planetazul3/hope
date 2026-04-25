# ADR 0006: Gaussian Probability Model

## Status
Accepted (Updated 2026-04-24)

## Context
The strategy requires a fast, responsive estimation of the probability that the next tick will continue the current trend. A Gaussian (Normal) distribution model was selected for its analytical simplicity and lack of training overhead.

## Decision
The system implements a Gaussian model where the probability of an upward move is calculated based on:
1.  **Drift (Mean)**: The average price change over a configurable `VOLATILITY_WINDOW` (default: 10 ticks).
2.  **Volatility (Std Dev)**: The standard deviation of price changes over the same window.
3.  **Time Horizon**: The probability is projected for the next 1 tick.

### Incremental Calculation (O(1))
To maintain zero-allocation performance, the `TickProcessor` maintains running sums of returns and squared returns. As a new tick arrives:
-   The oldest return (at `VOLATILITY_WINDOW`) is subtracted from the sums.
-   The newest return is added to the sums.
-   Mean and variance are calculated directly from these sums without looping over the history buffer.

## Consequences
-   **High Responsiveness**: By using a 10-tick window, the model adapts quickly to sudden reversals.
-   **Zero-Allocation**: No dynamic memory is used during probability calculation.
-   **Deterministic**: Given the same tick sequence, the model always yields the same probability, enabling reliable backtesting.

### Signal Reliability Guards
To reduce false positives in low-information regimes, the following filters are applied before signal generation:
-   **Signal-to-Noise Ratio (SNR)**: If `abs(drift / volatility) < 0.05`, the probability is forced to `0.5`. This avoids high-confidence signals when the trend is not statistically significant.
-   **Trend Length**: Signals are only allowed if the current directional run (`ticks_since_reversal`) is at least `min_trend_length` (default: 5). This filters out frequent whipsaw moves.
-   **Return Magnitude**: Price movement must exceed `volatility * 0.1` to be considered meaningful, reducing signals triggered by tiny price churn.
-   **Volatility Epsilon**: A small constant (`1e-8`) is added to the volatility denominator to prevent numerical instability and division by zero.
