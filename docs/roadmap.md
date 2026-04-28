# Development Roadmap

## Usage

This roadmap is the active development tracker for `hope`. Before starting work, identify the current stage, continue from the next unfinished or partially finished item, and update this file when a stage materially changes.

Status values:

- `Done`
- `In Progress`
- `Planned`

## Current Stage Summary

- Current active stage: `Maintenance and Scaling`
- Previous stage: `Stage 13 - Live Trading Deployment and Anomaly Monitoring` (Done)
- Overall state: Production-ready modular engine with zero-allocation hot paths, verified backtesting capabilities, and a standardized, observable ML pipeline backed by Canonical Causal Transformer and TS2Vec.

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
- Implement configurable model pooling (`mean` vs `last` vs `cls`) for architecture flexibility
- Add post-export ONNX validation and integrity checks
- Enhance UX with real-time `tqdm` progress visualization and hardware auditing
- Implement Transformer robustness: [CLS] token, Kaiming initialization, and Z-score standardization

## Stage 10 - Noise-Resilient Strategy Evolution

Status: `In Progress`

- [x] Transition ML backbone from Transformer to Gated TCN with Squeeze-and-Excitation
- [x] Implement Level-1 Haar Wavelet (DWT) feature extraction for time-frequency localization
- [x] Implement two-phase training curriculum: Contrastive Pre-training (SSL) and Supervised Fine-tuning
- [x] Synchronize Python and Rust feature calculation for 8-dimensional DWT inputs
- [x] Implement Multi-task learning (Auxiliary Volatility Head) for latent space regularization
- [x] Implement High-Fidelity Backtesting simulation in `src/bin/backtest.rs`
- [x] Implement INT8 Quantization for ONNX models to optimize L1/L2 cache residency
- [x] Implement high-fidelity data engineering pipeline: multi-symbol collection, incremental exports, and automated data validation
- [/] Explore regime-specific scaling and contrastive pre-training on multi-instrument data

## Stage 11 - ML Quality and Cloud Enforcement

Status: `Done`

- [x] Cloud-only training enforcement with runtime guards in all training entry points
- [x] Global reproducibility seeding (torch, numpy, random, cudnn deterministic)
- [x] Persistent cloud checkpoint saving during Phase 2 fine-tuning
- [x] Decision threshold sweep for empirical boundary selection and `.env` recommendation
- [x] DataLoader parallel data loading with `num_workers=2` and `pin_memory=True`
- [x] Comprehensive ML utility unit tests for `hope_ml.common` public API

## Stage 12 - Production Hardening

Status: `Done`

- [x] Integrate Canonical Causal Transformer with `[CLS]` token pooling and TS2Vec contrastive pre-training
- [x] Implement Daubechies (db2/db3) wavelet coefficients as $O(N)$ FIR filter banks
- [x] Zero-Allocation Rust paths for ONNX inference (`tract-onnx` and `StrategyEngine`)
- [x] Implement professional-grade UX logging and progress bars (`tqdm`)
- [x] Dynamic INT8 model quantization and Ed25519 cryptographic signing enforcement

## Stage 13 - Live Trading Deployment and Anomaly Monitoring

Status: `Done`
- [x] Implement Bayesian Optimization pipeline using Optuna for strategy hyperparameter tuning
- [x] Integrate Optuna and Plotly for advanced performance visualization
- [x] Harden backtester logging to include all strategy optimization targets
- [x] Implement real-time anomaly detection for live trading drift
- [x] Deploy live trading engine with production-ready guards
