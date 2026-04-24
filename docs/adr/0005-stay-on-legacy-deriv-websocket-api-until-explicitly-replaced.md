# ADR 0005: Stay on Legacy Deriv WebSocket API Until Explicitly Replaced

- Status: Accepted
- Date: 2026-04-24

## Context

The current runtime uses the legacy Deriv WebSocket flow on `wss://ws.derivws.com/websockets/v3?app_id=...` with socket-level `authorize` using a token from `.env`. The newer Deriv platform uses a different documentation set and a different authenticated connection bootstrap model. Accidentally mixing the two would risk broken auth, mismatched payloads, and silent transport drift.

## Decision

Keep the trading runtime on the legacy Deriv WebSocket API until an explicit migration decision is made and recorded in a new ADR. Treat the legacy API choice as intentional, not temporary by omission.

## Consequences

- Do not replace the current auth flow with OTP/bootstrap auth without a dedicated migration change.
- Do not swap endpoints, message shapes, or request conventions based on newer docs alone.
- API-facing changes must verify compatibility with both the current implementation and the legacy reference note in `docs/reference/deriv-api.md`.
- A future migration must update code, docs, ADRs, and operational guidance together.
