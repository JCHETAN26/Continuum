import importlib.util
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import yaml
from continuum_shared.db_url import strip_prisma_only_params, to_psycopg_url


def test_prisma_only_params_are_stripped_for_psycopg() -> None:
    stripped = strip_prisma_only_params(
        "postgresql://u:p@postgres:5432/continuum"
        "?schema=public&connection_limit=10&pool_timeout=30&sslmode=require"
    )
    query = parse_qs(urlsplit(stripped).query)

    # libpq rejects the connection outright when handed parameters it does not know.
    assert "schema" not in query
    assert "connection_limit" not in query
    assert "pool_timeout" not in query
    # Genuine libpq parameters must survive.
    assert query["sslmode"] == ["require"]
    assert urlsplit(stripped).path == "/continuum"


def test_to_psycopg_url_selects_the_driver_and_keeps_credentials() -> None:
    url = to_psycopg_url(
        "postgresql://u:p@postgres:5432/continuum?schema=public&connection_limit=10"
    )

    assert url.startswith("postgresql+psycopg://u:p@postgres:5432/continuum")
    assert "connection_limit" not in url


def test_compose_database_url_survives_the_alembic_conversion() -> None:
    """The migrations service runs Alembic against exactly this URL."""
    compose = yaml.safe_load(Path("infra/docker-compose.yml").read_text())

    url = to_psycopg_url(compose["x-app-env"]["DATABASE_URL"])

    assert url == "postgresql+psycopg://continuum:continuum@postgres:5432/continuum"


def test_ops_migration_is_reversible() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "shared"
        / "alembic"
        / "versions"
        / "20260728_0001_ops_runtime_tables.py"
    )
    spec = importlib.util.spec_from_file_location("ops_runtime_tables", migration_path)
    assert spec and spec.loader
    migration = importlib.util.module_from_spec(spec)
    sys.modules["ops_runtime_tables"] = migration
    spec.loader.exec_module(migration)

    assert migration.revision == "20260728_0001"
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
