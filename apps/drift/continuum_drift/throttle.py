from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from continuum_shared.config import settings
from continuum_shared.prisma import Prisma
from continuum_shared.prisma.enums import TrainingTrigger

Priority = Literal["dual_signal_high", "embedding_drift", "linguistic_drift"]


@dataclass(frozen=True)
class ThrottleDecision:
    accepted: bool
    reason: str
    priority: Priority | None = None


@dataclass(frozen=True)
class TriggerThrottler:
    min_documents: int = 100
    # Cosine distance between window centroids, the same scale the drift service alerts
    # on. Keep this within reach of drift_threshold: set far above it, the embedding
    # signal can never clear the gate and only linguistic drift can trigger training.
    min_embedding_drift: float = 0.08
    min_linguistic_drift: float = 0.60
    cooldown_hours: float = 6.0
    max_daily_trains: int = 3

    @classmethod
    def from_settings(cls) -> TriggerThrottler:
        return cls(
            min_documents=settings.drift_trigger_min_documents,
            min_embedding_drift=settings.drift_trigger_min_embedding_drift,
            min_linguistic_drift=settings.drift_trigger_min_linguistic_drift,
            cooldown_hours=settings.drift_trigger_cooldown_hours,
            max_daily_trains=settings.drift_trigger_max_daily_trains,
        )

    async def decide(
        self,
        db: Prisma,
        *,
        document_count: int,
        embedding_drift: float,
        linguistic_drift: float | None,
        now: datetime | None = None,
    ) -> ThrottleDecision:
        now = now or datetime.now(UTC)
        linguistic_score = linguistic_drift or 0.0

        if document_count < self.min_documents:
            return ThrottleDecision(False, "insufficient_documents")

        embedding_high = embedding_drift >= self.min_embedding_drift
        linguistic_high = linguistic_score >= self.min_linguistic_drift
        if not embedding_high and not linguistic_high:
            return ThrottleDecision(False, "below_drift_thresholds")

        last_train_time = await self.last_train_time(db)
        if last_train_time is not None and self.cooldown_hours > 0:
            cooldown_start = now - timedelta(hours=self.cooldown_hours)
            if last_train_time >= cooldown_start:
                return ThrottleDecision(False, "cooldown_active")

        trains_today = await self.trains_today(db, now)
        if trains_today >= self.max_daily_trains:
            return ThrottleDecision(False, "daily_cap_reached")

        if embedding_high and linguistic_high:
            return ThrottleDecision(True, "accepted", "dual_signal_high")
        if embedding_high:
            return ThrottleDecision(True, "accepted", "embedding_drift")
        return ThrottleDecision(True, "accepted", "linguistic_drift")

    async def last_train_time(self, db: Prisma) -> datetime | None:
        job = await db.trainingjob.find_first(
            where={"trigger": TrainingTrigger.DRIFT_ALERT},
            order={"queuedAt": "desc"},
        )
        return job.queuedAt if job else None

    async def trains_today(self, db: Prisma, now: datetime) -> int:
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = await db.query_raw(
            """
            SELECT COUNT(*)::int AS count
            FROM training_jobs
            WHERE trigger = 'DRIFT_ALERT'
              AND queued_at >= $1::timestamptz
              AND queued_at < $2::timestamptz
            """,
            start_of_day,
            now,
        )
        return rows[0]["count"] if rows else 0
