#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run scripts/check_stack_health.py --timeout 180 --interval 5
uv run scripts/seed.py
uv run scripts/verify_demo.py --timeout 240 --interval 5
