# Hope Project Documentation

`hope` is a production-grade trading system for Deriv synthetic indices, designed for low latency, deterministic execution, and auditable performance.

## Documentation Index

### Core Architecture
-   **[Architecture](architecture.md)**: System overview, runtime flow, and core boundaries.
-   **[Blueprint](blueprint.md)**: The target state and design principles of the engine.
-   **[Roadmap](roadmap.md)**: Current development stage and completed milestones.

### Operations
-   **[Runbook](runbook.md)**: Command reference, setup instructions, and troubleshooting.

### Architectural Decision Records (ADR)
Detailed justifications for technical choices:
-   [ADR 0001: WebSocket-first Event-Driven Engine](adr/0001-websocket-first-event-driven-engine.md)
-   [ADR 0002: Strict FSM for Trade Lifecycle](adr/0002-strict-fsm-for-trade-lifecycle.md)
-   [ADR 0003: Deterministic Model-System Separation](adr/0003-deterministic-model-system-separation.md)
-   [ADR 0004: Nonblocking Tick Audit Logging](adr/0004-nonblocking-tick-audit-logging.md)
-   [ADR 0005: Legacy Deriv WebSocket API Usage](adr/0005-stay-on-legacy-deriv-websocket-api-until-explicitly-replaced.md)
-   [ADR 0006: Gaussian Probability Model](adr/0006-gaussian-probability-model.md)
-   [ADR 0007: Neural Inference Integration](adr/0007-neural-inference-integration.md)
-   [ADR 0008: ML Training Pipeline Optimizations](adr/0008-ml-training-pipeline-optimizations.md)
-   [ADR 0009: Advanced ML Training and Strategy Enhancements](adr/0009-advanced-ml-training-and-strategy-enhancements.md)
-   [ADR 0010: Structured Logging and Environment-Aware Training](adr/0010-structured-logging-and-environment-aware-training.md)
-   [ADR 0011: Gated TCN with Squeeze-and-Excitation](adr/0011-gated-tcn-architecture.md)
-   [ADR 0012: Noise-Resilient Training with DWT and Contrastive Pre-training](adr/0012-noise-resilient-training.md)
-   [ADR 0013: High-Fidelity Backtesting Methodology](adr/0013-high-fidelity-backtesting.md)
-   [ADR 0014: Cloud-Only Training Enforcement](adr/0014-cloud-only-training-enforcement.md)

### Reference
-   [Deriv API Integration Notes](reference/deriv-api.md)

## Repository Structure

-   `src/lib.rs`: Core trading logic library.
-   `src/main.rs`: Live trading engine binary.
-   `src/bin/backtest.rs`: Strategy simulation binary.
-   `scripts/`: Python tools for data collection, export, and model training.
-   `notebooks/`: Research and model experimentation.
-   `data/`: SQLite storage for market ticks.
