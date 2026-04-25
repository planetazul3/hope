# Copilot Repository Instructions

Use [AGENTS.md](../AGENTS.md) as the canonical source of repository guidance.

Key rules for this repository:

- Work in English for code, docs, configs, and responses.
- Check `docs/blueprint.md` and `docs/roadmap.md` before substantial work, identify the current stage, and continue from the next incomplete item unless the user explicitly reprioritizes.
- Keep `docs/`, ADRs, and roadmap updates in the same change when behavior, architecture, config, operations, or development stage changes.
- Treat Deriv integration docs as mandatory input. Check `https://developers.deriv.com/llms.txt` first, then align changes with `docs/reference/deriv-api.md`.
- The runtime intentionally uses the legacy Deriv WebSocket API and token-based socket authorization. Do not change transport or auth flow without an explicit ADR-backed migration.
- Prefer deterministic, auditable changes. Do not invent Deriv API fields or undocumented behavior.
- Run `cargo fmt`, `cargo check --offline`, and `cargo test --offline` when applicable before claiming completion.
- Adhere to the **ML Engineering Standards** (Architecture, Optimization, Features, Batching, Integrity, Export, Balance) defined in `AGENTS.md`.
- Use `python3 consolidate_project_sources.py` or `make consolidate` when generating an auditable project snapshot that includes notebooks.
