# ADR-005: Production Ops Polish

## Status

Accepted

## Context

Continuum now runs a multi-service feedback loop: ingest, embedding, drift, linguistic drift,
training, serving, and dashboard. The system needs operational guarantees that are independent
of the demo path: reversible schema changes, structured observability, data-retention bounds,
secret-safe API authentication, and an automatic rollback policy when an activated model harms
serving reliability.

## Decision

- Keep Prisma responsible for the core product schema and add Alembic for operational tables
  that do not need generated Prisma models.
- Run Alembic through a compose `migrations` one-shot service before app containers serve.
- Emit JSON logs with correlation IDs from every FastAPI service and instrument FastAPI with
  OpenTelemetry stdout spans for local development.
- Prefer `API_KEY_BCRYPT_HASH` for deployments, with constant-time plaintext comparison kept
  only for local compatibility.
- Run retention as a simple scheduled worker in Compose instead of adding pg_cron to the local
  Postgres image.
- Keep rollback decisions in serving memory for low latency, while persisting request metrics
  and rollback audit events best-effort to operational tables.

## Consequences

The core schema remains stable and Prisma-generated clients stay small. Alembic migrations are
reversible and can be tested independently in CI. Local operations remain easy to run with
Compose, and production deployments can replace the scheduler, trace exporter, or secret source
without changing service code.
