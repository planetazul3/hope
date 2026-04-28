# Architecture Decision Records

This directory stores Architecture Decision Records (ADRs) for `hope`.

## Conventions

- File naming: `NNNN-short-kebab-case-title.md`
- Status values: `Accepted`, `Superseded`, or `Deprecated`
- Each ADR should describe context, decision, and consequences
- New architectural decisions should add a new ADR instead of rewriting history

## Index

- [0001 - WebSocket-First Event-Driven Engine](./0001-websocket-first-event-driven-engine.md)
- [0002 - Strict FSM for Trade Lifecycle](./0002-strict-fsm-for-trade-lifecycle.md)
- [0003 - Deterministic Model/System Separation](./0003-deterministic-model-system-separation.md)
- [0004 - Nonblocking Tick Audit Logging](./0004-nonblocking-tick-audit-logging.md)
- [0005 - Stay on Legacy Deriv WebSocket API Until Explicitly Replaced](./0005-stay-on-legacy-deriv-websocket-api-until-explicitly-replaced.md)
- [0006 - Gaussian Probability Model](./0006-gaussian-probability-model.md)
- [0007 - Neural Inference Integration](./0007-neural-inference-integration.md)
- [0008 - ML Training Pipeline Optimizations](./0008-ml-training-pipeline-optimizations.md)
- [0009 - Advanced ML Training and Strategy Enhancements](./0009-advanced-ml-training-and-strategy-enhancements.md)
- [0010 - Structured Logging and Environment-Aware Training](./0010-structured-logging-and-environment-aware-training.md)
- [0011 - Gated TCN with Squeeze-and-Excitation](./0011-gated-tcn-architecture.md) (Superseded by 0017)
- [0012 - Noise-Resilient Training with DWT and Contrastive Pre-training](./0012-noise-resilient-training.md)
- [0013 - High-Fidelity Backtesting Methodology](./0013-high-fidelity-backtesting.md)
- [0014 - Cloud-Only Training Enforcement](./0014-cloud-only-training-enforcement.md)
- [0015 - Professional-Grade Data Engineering Infrastructure](./0015-professional-grade-data-engineering-infrastructure.md)
- [0016 - Reliability and Performance Hardening](./0016-reliability-and-performance-hardening.md)
- [0017 - Production Hardening via Canonical Transformer, TS2Vec, and Daubechies Wavelets](./0017-canonical-transformer-ts2vec.md)
- [ADR 0018: Persistent Engine State and Buffered Ingestion](0018-persistent-engine-state-and-buffered-ingestion.md)
- [ADR 0019: Numerical Integrity and Math Parity Hardening](0019-numerical-integrity-and-math-parity-hardening.md)
- [ADR 0020: Fix FSM Recovery Stall on Reconnect](0020-fix-fsm-recovery-stall-on-reconnect.md)

