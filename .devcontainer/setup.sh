#!/usr/bin/env bash
# Runs once when the Codespace is created.
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
corepack enable && corepack prepare pnpm@9.15.9 --activate

pnpm install --frozen-lockfile
uv sync --all-packages --dev

# Compose reads .env; the example carries working local defaults.
[ -f .env ] || cp .env.example .env

cat <<'EOF'

Continuum is ready.

  docker compose --env-file .env -f infra/docker-compose.yml up --build -d --wait
  uv run scripts/seed.py
  uv run scripts/verify_demo.py --timeout 1800 --interval 5

The dashboard is on port 3000 once the stack is healthy. First build pulls torch for the
trainer image and bakes the embedding model in, so expect it to take a while.

EOF
