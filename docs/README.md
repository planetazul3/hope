# Documentation

This directory contains the project documentation for `hope`, a Rust-based trading system for Deriv synthetic indices.

## Contents

- `blueprint.md`: Target production design and system boundaries.
- `roadmap.md`: Current staged development plan and progress tracker.
- `architecture.md`: Current system structure, data flow, and component responsibilities.
- `runbook.md`: Local setup, runtime configuration, and operator-facing execution notes.
- `adr/README.md`: ADR index and conventions.
- `reference/README.md`: Read-only local notes for external systems and official vendor docs.

## Documentation Rules

- Update docs when code changes behavior, configuration, operational flow, or architecture.
- Update `roadmap.md` when development stage or progress status materially changes.
- Keep docs concrete and repository-specific; avoid aspirational or undocumented behavior.
- Record architectural decisions in `docs/adr/` when introducing, changing, or reversing core technical choices.

## Current Focus

The current documentation covers the system blueprint, staged roadmap, event-driven trading engine, deterministic FSM-centered execution flow, runtime configuration from `.env`, and the initial architectural decisions that shape the repository.
