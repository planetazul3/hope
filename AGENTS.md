# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Rust binary crate. Application entrypoint code lives in `src/main.rs`, and the runtime is split across focused modules in `src/` such as `websocket_client.rs`, `engine.rs`, `fsm.rs`, and `tick_processor.rs`. Project documentation lives under `docs/`, and architectural decisions live under `docs/adr/`. Keep new modules and docs organized by responsibility rather than by file size alone. If integration coverage becomes necessary, place it in a top-level `tests/` directory.

## Build, Test, and Development Commands
Use Cargo for all local workflows.

- `cargo run` starts the trading system with the current `.env` configuration.
- `cargo check --offline` validates compilation quickly without producing a release binary.
- `cargo fmt` applies standard Rust formatting.
- `cargo test --offline` runs unit and integration tests; it may be minimal until more tests are added.
- `python3 consolidate_project_sources.py` generates a timestamped audit snapshot of source, docs, config, and notebooks.
- `make consolidate` runs the same audit snapshot command through the optional Makefile entry point.

Run `cargo fmt && cargo check --offline && cargo test --offline` before opening a pull request. If offline resolution stops working because dependencies changed, restore dependency resolution first, regenerate `Cargo.lock`, and then update the documented commands if needed.

## Coding Style & Naming Conventions
Target Rust 2021 and follow `rustfmt` defaults with 4-space indentation. Use `snake_case` for functions, modules, and variables, and `PascalCase` for structs and enums. Prefer small, focused types like `WebSocketClientConfig` and `WebSocketEvent`; keep fallible paths returning `anyhow::Result` where that pattern is already in use.

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
