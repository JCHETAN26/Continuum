#!/bin/bash
# Creates the shadow database Prisma needs for `migrate dev`.
#
# Prisma diffs the schema against a scratch database to author migrations. It can create one
# itself only with elevated privileges, so provisioning it here keeps the application role
# unprivileged and makes `pnpm db:migrate:create` work on a fresh volume.
#
# Runs once, on first initialisation of the data directory.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
	CREATE DATABASE ${POSTGRES_DB}_shadow OWNER $POSTGRES_USER;
EOSQL

# pgvector is installed per-database. The application database gets the extension from the
# initial migration; the shadow database needs it up front or the diff fails on the
# vector(384) columns.
for db in "${POSTGRES_DB}_shadow"; do
	psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
		CREATE EXTENSION IF NOT EXISTS vector;
	EOSQL
done

echo "shadow database ${POSTGRES_DB}_shadow ready"
