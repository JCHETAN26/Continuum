from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from continuum_shared.db_url import to_psycopg_url
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = None


def database_url() -> str:
    url = os.environ.get("DATABASE_URL") or read_dotenv_value("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required for Alembic migrations.")
    return to_psycopg_url(url)


def read_dotenv_value(key: str) -> str | None:
    dotenv = Path(".env")
    if not dotenv.exists():
        dotenv = Path(".env.example")
    if not dotenv.exists():
        return None
    for line in dotenv.read_text().splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        if name == key:
            return value.strip().strip("'").strip('"')
    return None


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
