# ADR 0006: Gaussian Probability Model

- Status: Accepted
- Date: 2026-04-24

## Context

The system previously used a `ConstantModel` that returned a fixed probability of 0.6. This was a placeholder for development. To evolve the strategy, we need a model that responds to market dynamics while remaining isolated from the core execution logic (as per ADR 0003).

## Decision

Implement a `GaussianModel` that estimates the probability of a price move continuation based on recent volatility and estimated drift.

The model assumes a local Brownian motion process $dP_t = \mu dt + \sigma dW_t$.
The probability that the price will be higher after $T$ ticks is estimated as:
$$P(P_{t+T} > P_t) = \Phi\left(\frac{\mu \sqrt{T}}{\sigma}\right)$$

where:
- $\mu$ is the estimated drift (average return per tick over a window).
- $\sigma$ is the estimated volatility (standard deviation of returns per tick over a window).
- $T$ is the contract duration in ticks.
- $\Phi$ is the Cumulative Distribution Function (CDF) of the standard normal distribution.

## Implementation Details

- `TickProcessor` will track the sum and sum of squares of returns to allow $O(1)$ calculation of variance over the sliding window.
- `GaussianModel` will consume these statistics from the `TickSnapshot`.
- The `statrs` crate or a simple approximation of the normal CDF will be used. (I'll check if `statrs` is available, otherwise I'll use a standard approximation).

## Consequences

- The model provides a more realistic probability estimate than the constant placeholder.
- The engine can now react to volatility changes (e.g., skipping trades when volatility is too high or drift is too low).
- The mathematical foundation is documented and auditable.
