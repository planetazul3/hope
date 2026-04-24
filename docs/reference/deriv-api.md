# Deriv API Reference Note

- Status: Read-only reference note
- Upstream source: `https://developers.deriv.com/llms.txt`
- Retrieved for this repository: 2026-04-24

## Purpose

This file is a local development reference for the Deriv integration used by `hope`. It is not a replacement for the official documentation. Before changing transport, authorization, trading requests, or message parsing, check the upstream Deriv docs first.

## Official Reference

- Primary reference: `https://developers.deriv.com/llms.txt`
- API docs index: `https://developers.deriv.com/docs`
- Legacy WebSocket docs used by the current runtime flow: `https://api.deriv.com/docs/core-concepts/authorization-authentication/`

## Current Repository Alignment

The current implementation uses the legacy WebSocket flow on `wss://ws.derivws.com/websockets/v3?app_id=...` and authorizes the socket with a token from `.env`. This matches the legacy Deriv documentation pattern used by the code in `src/websocket_client.rs`.

This legacy API choice is intentional and protected by `docs/adr/0005-stay-on-legacy-deriv-websocket-api-until-explicitly-replaced.md`. Do not migrate to the newer Deriv auth/transport model implicitly.

The repository currently relies on documented WebSocket message types for:

- `authorize`
- `tick` / `ticks`
- `proposal`
- `buy`
- `proposal_open_contract`
- `error`

## Development Rules for Deriv-Specific Changes

- Do not invent request or response fields.
- Do not assume undocumented message ordering.
- Do not switch transport or authorization flow without updating the relevant ADR.
- Do not change Deriv-facing payloads without updating this note and the affected docs when the upstream contract changes.
- If the official docs and the current implementation diverge, document the mismatch explicitly before changing code.
