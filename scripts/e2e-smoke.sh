#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run scripts/check_stack_health.py --timeout 180 --interval 5
uv run scripts/seed.py
# Budget covers the slowest stage, which is now training rather than ingest: a real LoRA
# run over the drift window, an ONNX export, and re-embedding the whole corpus with the
# activated encoder. Raised rather than trading away assertions.
uv run scripts/verify_demo.py --timeout 1800 --interval 5
