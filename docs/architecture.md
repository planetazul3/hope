# Architecture

## System Architecture: The Dual-Engine Approach

`hope` is built on a **Dual-Engine Architecture** that separates research from execution:
- **Python (The Brain)**: Handles data engineering, ML research, model training, and feature discovery. It provides the high-level intelligence and historical memory.
- **Rust (The Body)**: Handles real-time transport, tick processing, the Finite State Machine (FSM), and trade execution. It provides low-latency, deterministic, and safe execution.

The two engines are synchronized through **Feature Parity** (identical mathematical implementations in both languages) and the **ONNX Model Interchange** format.

## The Engineering Gears

### 1. Data Engineering Gear (Python)
- **`scripts/tick_collector.py`**: A robust, asynchronous ingestion service. It supports multi-symbol discovery (`--mode list`), gap-aware historical collection (`--mode history`), and seamless live transitions (`--mode both`). It uses a class-based design with exponential backoff and graceful shutdown.
- **`scripts/export_db.py`**: A high-performance transformation tool. It enables incremental CSV/Parquet exports from SQLite, performs automated integrity validation (gap/duplicate detection), and generates summary statistics for dataset auditing.

### 2. Runtime Execution Gear (Rust)
- **Transport**: `src/websocket_client.rs` maintains a persistent Deriv WebSocket connection with automatic reconnection.
- **Tick Processing**: `src/tick_processor.rs` maintains a fixed-size ring buffer, computing volatility and drift in O(1) time.
- **State Management**: `src/fsm.rs` enforces trade lifecycle transitions: `Idle`, `OrderPending`, `InPosition`, and `Cooldown`.
- **Strategy Engine**: `src/strategy.rs` evaluates probability signals from the model against execution thresholds.
- **Execution & Risk**: `src/execution.rs` and `src/risk.rs` provide rate-limiting guards and consecutive-loss protection.

### 3. Machine Learning Gear (PyTorch/ONNX)
- **Architecture**: Canonical Causal Transformer Encoder ($L=32$, $O(L^2)$ mathematical optimality) with a prepended `[CLS]` token.
- **Training**: Two-phase curriculum starting with Contrastive Pre-training (TS2Vec: Hierarchical Contrastive Learning with latent-space timestamp masking and random cropping using InfoNCE loss) and then Supervised Fine-tuning with Focal Loss and Volatility Huber Loss. Training is performed exclusively in cloud GPU environments.
- **Interchange**: Models are exported to ONNX (static graph: 1x32x8) and dynamically quantized to INT8 (`QuantType.QInt8`) for low-latency CPU inference via the `tract` engine in Rust.

## Data & Storage Layer

- **Database**: SQLite with **Write-Ahead Logging (WAL)** and `synchronous=NORMAL` to support high-frequency concurrent writes from the collector and reads from the exporter.
- **Schema**: Multi-symbol aware (`symbol`, `epoch`, `quote`) with unique constraints and indexing for fast range queries.
- **Interchange Format**: Headerless CSV for high-speed parsing in both Rust and Python pipelines.

## Operational Constraints

- **Zero Allocation Hot Path**: The `tract-onnx` execution environment operates on a strictly zero-allocation hot path using pre-allocated `ndarray` buffers during the Tick -> Strategy -> Signal flow.
- **FSM-Strict**: "One Trade at a Time" enforcement and mandatory cooldown periods.
- **Complexity**: O(1) complexity for all per-tick statistical calculations.
- **Security**: Asynchronous audit logging with restrictive file permissions (`0o600`).
