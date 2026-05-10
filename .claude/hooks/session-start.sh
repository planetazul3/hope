#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if command -v python3 >/dev/null 2>&1; then
  python3 -m pip install -r requirements.txt
  python3 -m pip install pytest
fi

if command -v cargo >/dev/null 2>&1; then
  cargo fetch --locked
  cargo build --locked --tests --offline
fi
