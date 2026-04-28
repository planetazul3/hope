# Gemini Instructions

This repository uses `AGENTS.md` as the canonical project instruction file.

Follow all guidance in:

<!-- Imported from: ./AGENTS.md -->
# Repository Guidelines

## Project Structure & Module Organization
This repository is a Rust library crate with multiple binaries. 
- `src/lib.rs`: Core trading logic and module exports.
- `src/main.rs`: Entrypoint for the live trading engine.
- `src/bin/backtest.rs`: Entrypoint for strategy simulation.
Modules include `websocket_client.rs`, `engine.rs`, `fsm.rs`, `tick_processor.rs`, and `strategy.rs`.

Project documentation lives under `docs/`, and architectural decisions live under `docs/adr/`.

## Build, Test, and Development Commands
Use Cargo and the provided Makefile for local workflows.

- `make run` starts the live trading engine.
- `make backtest` runs strategy simulation on `data/ticks.csv`.
- `make export` exports ticks from SQLite to CSV for backtesting and training.
- `make collect` collects historical ticks from Deriv API.
- `make verify` runs format, check, and tests.
- `make consolidate` generates an audit snapshot for AI analysis.
- `cargo fmt` applies standard Rust formatting.
- `cargo test --offline` runs unit and integration tests.

Run `cargo fmt && cargo check --offline && cargo test --offline` before opening a pull request. If offline resolution stops working because dependencies changed, restore dependency resolution first, regenerate `Cargo.lock`, and then update the documented commands if needed.

## Coding Style & Naming Conventions
Target Rust 2021 and follow `rustfmt` defaults with 4-space indentation. Use `snake_case` for functions, modules, and variables, and `PascalCase` for structs and enums. Prefer small, focused types like `WebSocketClientConfig` and `WebSocketEvent`; keep fallible paths returning `anyhow::Result` where that pattern is already in use.

### Machine Learning Workflow
1. **Data**: Collect ticks with `make collect`. Use `--mode backfill` or `--mode both` for gapless collection.
2. **Export**: Convert to CSV with `make export`. Use `--incremental` for fast updates and `--validate` to ensure data integrity.
3. **Training**: Execute `notebooks/colab_training.ipynb` or `notebooks/kaggle_training.ipynb` in their respective cloud GPU environments. Invoking any Python training script directly on the local machine is prohibited.
4. **Deploy**: Ensure `model.onnx` and `model.onnx.sig` are in the project root.
5. **Config**: Set `TRANSFORMER_SEQUENCE_LENGTH=32` and `MODEL_PUBLIC_KEY` in `.env`.

## Data Engineering Standards
Tick collection and export tools must maintain high fidelity, performance, and professional UX:
- **Storage**: SQLite databases must use a multi-symbol schema (`symbol`, `epoch`, `quote`) with `UNIQUE(symbol, epoch)`. Enable `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` for concurrent I/O.
- **Architecture**: 
    - **Separation of Concerns**: Use a class-based, asynchronous approach (e.g., `TickCollector`, `DerivClient`) with discrete execution modes.
    - **Lifecycle Management**: Implement robust `asyncio` signal handling for `SIGINT` and `SIGTERM` to ensure graceful resource cleanup.
    - **Memory Efficiency**: Use chunked processing (via `pandas` or generator patterns) to handle datasets larger than available RAM.
- **Collection**: 
    - Support `history` (backward), `backfill` (forward from last saved), and `live` (WebSocket subscription) modes.
    - Implement `both` mode to perform historical backfill followed by a seamless transition to a live stream.
    - Include a `list` mode for real-time symbol discovery from the Deriv API.
- **Resilience**: Implement automatic reconnection logic and exponential backoff for WebSocket failures.
- **Observability**:
    - **Gap Analysis**: Perform real-time analysis of historical batches to detect and warn about non-sequential data (gaps > 2.1s).
    - **Progress Tracking**: Use `tqdm` for long-running I/O operations and context-aware logging with the `logging` module.
- **Export**: Use `pandas` for high-performance I/O. Support streaming Gzip and Parquet formats.
- **Verification**: Exports must optionally support `--validate` (gap/duplicate detection) and `--stats` (summary statistics and price distribution histograms).
- **Incremental**: Support `--incremental` export by using low-level file seeking (O(1) complexity) to detect the last epoch in target CSVs.

## ML Engineering Standards
All model training and export workflows must adhere to these standards:
- **Architecture**: Use Canonical Causal Transformer Encoder ($L=32$, $O(L^2)$ complexity) with a prepended [CLS] token and Multi-Task Learning (Direction + Volatility heads).
- **Optimization**: Training loops must use a two-phase curriculum starting with TS2Vec (Hierarchical Contrastive Learning with InfoNCE) and then Supervised Fine-tuning with Focal Loss and Volatility Huber Loss, using ReduceLROnPlateau and warmup scheduler.
- **Features**: Preprocessing must include 8 features: 5 base features, 2 Daubechies (db2/db3) FIR filter bank coefficients instead of Haar wavelets, and 1 short-to-long-term volatility ratio.
- **Batching**: Use PyTorch `DataLoader` and `TensorDataset`.
- **Integrity**: Always load the best recorded state dictionary before exporting to ONNX.
- **Export**: ONNX models must use static graphs (batch size 1, sequence length 32) and dynamic INT8 Quantization (`QuantType.QInt8`).
- **Path Agnosticism**: Training scripts must implement automatic fallback detection for cloud environments (Kaggle Dataset inputs, `/kaggle/working/hope/`, `/content/hope/`, and Google Drive mounts) to ensure robust execution across different clone locations.
- **Balance**: Loss functions must use Focal Loss with Label Smoothing.

## Testing Guidelines
Prefer unit tests close to the code they cover with `#[cfg(test)] mod tests`. Name tests by behavior, for example `reconnects_after_socket_error`. Add integration tests under `tests/` only when validating crate-level behavior across modules. New parsing, reconnect, or error-handling logic should ship with tests.

## Documentation & ADR Maintenance
Documentation is part of the change, not follow-up work. Update files in `docs/` whenever a change affects behavior, architecture, configuration, operations, or developer workflow. Add a new ADR in `docs/adr/` when introducing, changing, or reversing an architectural decision; update the ADR index when you do. Do not leave `AGENTS.md`, docs, or ADRs stale after code changes.

For Deriv-specific work, treat the official docs as mandatory input. Check `https://developers.deriv.com/llms.txt` first, then use the local read-only notes under `docs/reference/` to keep repository context aligned. If a Deriv API change affects message shapes, auth flow, or execution behavior, update the relevant docs and ADRs in the same change.
The current runtime intentionally uses the legacy Deriv WebSocket API and token-based socket authorization. Do not migrate to a newer Deriv transport or auth flow unless the change is explicit, documented, and backed by a new ADR.

## Cross-Agent Instruction Files
Keep `AGENTS.md` as the canonical instruction source for the repository. `GEMINI.md` and `.github/copilot-instructions.md` must stay aligned with it so Gemini CLI and GitHub Copilot / VS Code agents follow the same project rules. When changing repository guidance, update the corresponding agent instruction files in the same change.

## Blueprint & Roadmap Discipline
Before starting substantial work, check `docs/blueprint.md` for the target system shape and `docs/roadmap.md` for the current development stage. Identify which stage is active, continue from the next incomplete item, and keep improvements aligned with that stage unless the user explicitly reprioritizes the work.

When a change completes a roadmap item, advances the current stage, or changes development priorities, update `docs/roadmap.md` in the same change. If implementation diverges from the blueprint in a meaningful way, update `docs/blueprint.md` and add or amend an ADR when the divergence is architectural.

## Commit & Pull Request Guidelines
Match the existing history: short, imperative commit subjects such as `Harden websocket event handling`. Pull requests should summarize the behavior change, list verification commands run, and call out any runtime or configuration impact. Include logs or terminal snippets when changing connection lifecycle behavior.

## Configuration Tips
Runtime configuration is loaded from `.env`. Logging is controlled with `LOG_LEVEL`, and trading defaults are loaded through `AppConfig` in `src/config.rs`. Document any changes to tokens, endpoint, app ID, symbol, contract type, thresholds, timing, or queue sizing in the relevant docs and PR description.

<!-- End of import from: ./AGENTS.md -->

## Project Modernization (April 2026)
This project is undergoing a comprehensive modernization to align all dependencies (Python and Rust) with their latest stable versions as of April 2026.

- **High-level Goal**: Standardize on Python 3.12+ and Rust 2021 Edition across all components.
- **Detailed Contexts**:
    - [Python & ML Modernization](./scripts/GEMINI.md) — NumPy 2.x compatibility and ML pipeline updates.

## Gemini-Specific Note

After updating this file or any imported instruction file, reload Gemini CLI memory so the latest repository instructions are active.

Repository workflow additions remain defined in `AGENTS.md`, including the audit snapshot tool:

- `python3 consolidate_project_sources.py`
- `make consolidate`