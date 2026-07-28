#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run scripts/check_stack_health.py --timeout 180 --interval 5
uv run scripts/seed.py
# Real newsgroup posts are ~7x longer than the hand-written sentences this corpus
# replaced, so MiniLM has proportionally more tokens to embed across 1500 documents.
uv run scripts/verify_demo.py --timeout 480 --interval 5
