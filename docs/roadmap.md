# Development Roadmap

## Usage

This roadmap is the active development tracker for `hope`. Before starting work, identify the current stage, continue from the next unfinished or partially finished item, and update this file when a stage materially changes.

Status values:

- `Done`
- `In Progress`
- `Planned`

## Current Stage Summary

- Current active stage: `Stage 9 - ML Pipeline Observability and Portability` (Complete)
- Overall state: Production-ready modular engine with zero-allocation hot paths, verified backtesting capabilities, and a standardized, observable ML pipeline.

## Stage 1 - Deterministic Core Foundation

Status: `Done`

## Stage 2 - Documentation and Architecture Controls

Status: `Done`

## Stage 3 - Live Integration Hardening

Status: `Done`

## Stage 4 - Test Expansion and Failure Simulation

Status: `Done`

## Stage 5 - Operational Readiness

Status: `Done`

## Stage 6 - Strategy Evolution

Status: `Done`

## Stage 7 - Neural Inference Evolution

Status: `Done`

## Stage 8 - Simulation and Performance Hardening

Status: `Done`

- Rebuild project as a library (`src/lib.rs`) to enable cross-binary logic sharing
- Implement high-performance backtesting binary (`src/bin/backtest.rs`)
- Optimize statistical calculations (Drift/Volatility) to O(1) complexity via incremental sums
- Implement zero-allocation history access (`last_n_into`) and inference buffers
- Harden API reliability with `PermitGuard` logic
- Implement secure logging (filtered error payloads and restrictive file permissions)
- Enhance session auditing with live balance tracking and Win Rate metrics

## Stage 9 - ML Pipeline Observability and Portability

Status: `Done`

- Implement structured logging (`training.log`) for full training traceability
- Standardize cross-platform notebook execution (Google Colab / Kaggle / Local)
- Implement configurable model pooling (`mean` vs `last`) for architecture flexibility
- Add post-export ONNX validation and integrity checks
- Enhance UX with real-time `tqdm` progress visualization and hardware auditing
