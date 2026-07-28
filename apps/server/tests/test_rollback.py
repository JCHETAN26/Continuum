from datetime import UTC, datetime, timedelta

import pytest
from continuum_server.rollback import ModelRollbackPolicy


def test_rollback_policy_waits_for_minimum_request_count() -> None:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    policy = ModelRollbackPolicy(min_requests=3, now=lambda: now)
    policy.note_activation("baseline", "candidate")
    policy.record("candidate", status_code=503, latency_ms=10)
    policy.record("candidate", status_code=503, latency_ms=11)

    decision = policy.evaluate("candidate")

    assert not decision.should_rollback
    assert decision.reason == "insufficient_requests"


def test_rollback_policy_triggers_when_error_rate_exceeds_threshold() -> None:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    policy = ModelRollbackPolicy(error_rate_threshold=0.05, min_requests=100, now=lambda: now)
    policy.note_activation("baseline", "candidate")
    for _ in range(94):
        policy.record("candidate", status_code=200, latency_ms=8)
    for _ in range(6):
        policy.record("candidate", status_code=503, latency_ms=12)

    decision = policy.evaluate("candidate")

    assert decision.should_rollback
    assert decision.previous_version == "baseline"
    assert decision.error_rate == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_rollback_policy_executes_callback_once() -> None:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    policy = ModelRollbackPolicy(error_rate_threshold=0.05, min_requests=1, now=lambda: now)
    policy.note_activation("baseline", "candidate")
    policy.record("candidate", status_code=503, latency_ms=12)
    calls: list[tuple[str, str]] = []

    async def rollback(failed: str, previous: str) -> None:
        calls.append((failed, previous))

    decision = await policy.rollback_if_needed("candidate", rollback)

    assert decision.should_rollback
    assert calls == [("candidate", "baseline")]
    assert policy.evaluate("candidate").reason == "insufficient_requests"


def test_rollback_policy_prunes_old_metrics() -> None:
    current = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    policy = ModelRollbackPolicy(min_requests=1, window=timedelta(minutes=5), now=lambda: current)
    policy.note_activation("baseline", "candidate")
    policy.record(
        "candidate",
        status_code=503,
        latency_ms=12,
        observed_at=current - timedelta(minutes=10),
    )

    decision = policy.evaluate("candidate")

    assert decision.request_count == 0
    assert not decision.should_rollback
