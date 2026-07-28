from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import getenv


@dataclass(frozen=True)
class RequestMetric:
    model_version: str
    status_code: int
    latency_ms: float
    observed_at: datetime


@dataclass(frozen=True)
class RollbackDecision:
    should_rollback: bool
    model_version: str | None
    previous_version: str | None
    error_rate: float
    request_count: int
    reason: str


class ModelRollbackPolicy:
    def __init__(
        self,
        *,
        error_rate_threshold: float = 0.05,
        window: timedelta = timedelta(minutes=5),
        min_requests: int = 100,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.error_rate_threshold = error_rate_threshold
        self.window = window
        self.min_requests = min_requests
        self._now = now or (lambda: datetime.now(UTC))
        self._metrics: dict[str, deque[RequestMetric]] = defaultdict(deque)
        self._previous_active: str | None = None

    @classmethod
    def from_env(cls) -> ModelRollbackPolicy:
        return cls(
            error_rate_threshold=float(getenv("ROLLBACK_ERROR_RATE_THRESHOLD", "0.05")),
            window=timedelta(seconds=int(getenv("ROLLBACK_WINDOW_SECONDS", "300"))),
            min_requests=int(getenv("ROLLBACK_MIN_REQUESTS", "100")),
        )

    def note_activation(self, previous_version: str | None, active_version: str) -> None:
        if previous_version and previous_version != active_version:
            self._previous_active = previous_version

    def record(
        self,
        model_version: str,
        *,
        status_code: int,
        latency_ms: float,
        observed_at: datetime | None = None,
    ) -> None:
        metric = RequestMetric(
            model_version=model_version,
            status_code=status_code,
            latency_ms=latency_ms,
            observed_at=observed_at or self._now(),
        )
        bucket = self._metrics[model_version]
        bucket.append(metric)
        self._prune(model_version)

    def evaluate(self, active_version: str | None) -> RollbackDecision:
        if not active_version:
            return RollbackDecision(False, None, self._previous_active, 0.0, 0, "no_active_model")

        self._prune(active_version)
        metrics = list(self._metrics[active_version])
        request_count = len(metrics)
        if request_count < self.min_requests:
            return RollbackDecision(
                False,
                active_version,
                self._previous_active,
                0.0,
                request_count,
                "insufficient_requests",
            )

        errors = sum(1 for metric in metrics if metric.status_code >= 500)
        error_rate = errors / request_count if request_count else 0.0
        if error_rate <= self.error_rate_threshold:
            return RollbackDecision(
                False,
                active_version,
                self._previous_active,
                error_rate,
                request_count,
                "error_rate_ok",
            )

        if not self._previous_active or self._previous_active == active_version:
            return RollbackDecision(
                False,
                active_version,
                self._previous_active,
                error_rate,
                request_count,
                "no_previous_model",
            )

        return RollbackDecision(
            True,
            active_version,
            self._previous_active,
            error_rate,
            request_count,
            "error_rate_exceeded",
        )

    async def rollback_if_needed(
        self,
        active_version: str | None,
        rollback: Callable[[str, str], Awaitable[None]],
    ) -> RollbackDecision:
        decision = self.evaluate(active_version)
        if decision.should_rollback and decision.model_version and decision.previous_version:
            await rollback(decision.model_version, decision.previous_version)
            self._metrics[decision.model_version].clear()
        return decision

    def _prune(self, model_version: str) -> None:
        cutoff = self._now() - self.window
        bucket = self._metrics[model_version]
        while bucket and bucket[0].observed_at < cutoff:
            bucket.popleft()
