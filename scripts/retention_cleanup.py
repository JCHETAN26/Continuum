from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import structlog
from continuum_shared.config import settings
from continuum_shared.prisma import Prisma

logger = structlog.get_logger()


@dataclass(frozen=True)
class RetentionPolicy:
    embeddings_days: int = 90
    drift_windows_days: int = 30
    training_jobs_days: int = 365


RETENTION_SQL = {
    "embeddings": "DELETE FROM embeddings WHERE created_at < now() - ($1::int * interval '1 day')",
    "drift_windows": (
        "DELETE FROM drift_windows WHERE created_at < now() - ($1::int * interval '1 day')"
    ),
    "linguistic_windows": (
        "DELETE FROM linguistic_windows WHERE created_at < now() - ($1::int * interval '1 day')"
    ),
    "training_jobs": (
        "DELETE FROM training_jobs WHERE queued_at < now() - ($1::int * interval '1 day')"
    ),
}


async def run_retention_cleanup(db: Prisma, policy: RetentionPolicy) -> dict[str, int]:
    deleted = {
        "embeddings": await db.execute_raw(RETENTION_SQL["embeddings"], policy.embeddings_days),
        "drift_windows": await db.execute_raw(
            RETENTION_SQL["drift_windows"], policy.drift_windows_days
        ),
        "linguistic_windows": await db.execute_raw(
            RETENTION_SQL["linguistic_windows"], policy.drift_windows_days
        ),
        "training_jobs": await db.execute_raw(
            RETENTION_SQL["training_jobs"], policy.training_jobs_days
        ),
    }
    logger.info("Retention cleanup completed", **deleted)
    return deleted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up old Continuum operational data.")
    parser.add_argument("--embeddings-days", type=int, default=settings.retention_embeddings_days)
    parser.add_argument(
        "--drift-windows-days",
        type=int,
        default=settings.retention_drift_windows_days,
    )
    parser.add_argument(
        "--training-jobs-days",
        type=int,
        default=settings.retention_training_jobs_days,
    )
    args = parser.parse_args()

    db = Prisma()
    await db.connect()
    try:
        await run_retention_cleanup(
            db,
            RetentionPolicy(
                embeddings_days=args.embeddings_days,
                drift_windows_days=args.drift_windows_days,
                training_jobs_days=args.training_jobs_days,
            ),
        )
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
