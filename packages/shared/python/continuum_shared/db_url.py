"""Connection-string helpers shared by the Prisma clients and Alembic.

Both read the same ``DATABASE_URL``, but they do not accept the same query parameters.
Prisma sizes its connection pool from the URL; libpq rejects a connection outright when
handed a parameter it does not recognise. Anything Prisma-specific has to come off the
URL before psycopg sees it.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

PRISMA_ONLY_PARAMS = frozenset(
    {
        "schema",
        "connection_limit",
        "pool_timeout",
        "pgbouncer",
        "socket_timeout",
        "statement_cache_size",
    }
)


def strip_prisma_only_params(url: str) -> str:
    """Remove Prisma-specific query parameters, preserving genuine libpq ones."""
    parts = urlsplit(url)
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query) if key not in PRISMA_ONLY_PARAMS]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def to_psycopg_url(url: str) -> str:
    """Convert a Prisma-flavoured Postgres URL into one SQLAlchemy can open."""
    cleaned = strip_prisma_only_params(url)
    if cleaned.startswith("postgresql://"):
        return cleaned.replace("postgresql://", "postgresql+psycopg://", 1)
    return cleaned
