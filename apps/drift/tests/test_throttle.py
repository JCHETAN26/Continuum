from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from continuum_drift.throttle import TriggerThrottler


def db_with_training_history(*, last_train_time=None, trains_today=0):
    db = MagicMock()
    if last_train_time is None:
        db.trainingjob.find_first = AsyncMock(return_value=None)
    else:
        job = MagicMock()
        job.queuedAt = last_train_time
        db.trainingjob.find_first = AsyncMock(return_value=job)
    db.query_raw = AsyncMock(return_value=[{"count": trains_today}])
    return db


@pytest.mark.asyncio
async def test_rejects_insufficient_documents_before_database_checks():
    throttler = TriggerThrottler(min_documents=100)
    db = db_with_training_history()

    decision = await throttler.decide(
        db,
        document_count=99,
        embedding_drift=0.99,
        linguistic_drift=0.99,
    )

    assert decision.accepted is False
    assert decision.reason == "insufficient_documents"
    db.trainingjob.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_rejects_when_both_signals_are_low():
    throttler = TriggerThrottler(
        min_documents=10, min_embedding_drift=0.75, min_linguistic_drift=0.6
    )
    db = db_with_training_history()

    decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.5,
        linguistic_drift=0.4,
    )

    assert decision.accepted is False
    assert decision.reason == "below_drift_thresholds"


@pytest.mark.asyncio
async def test_accepts_a_realistic_embedding_spike_without_a_linguistic_signal():
    """Regression: the defaults must be reachable by real drift scores.

    Centroid cosine distance reads around 0.10 on a domain shift, and the linguistic
    window is computed on its own schedule, so it is routinely absent when a drift alert
    is evaluated. With min_embedding_drift left at a normalised-scale 0.75 this returned
    below_drift_thresholds and no training job was ever created from embedding drift.
    """
    throttler = TriggerThrottler()
    db = db_with_training_history()

    decision = await throttler.decide(
        db,
        document_count=1500,
        embedding_drift=0.0988,
        linguistic_drift=None,
    )

    assert decision.accepted is True
    assert decision.priority == "embedding_drift"


@pytest.mark.asyncio
async def test_rejects_when_cooldown_active():
    now = datetime.now(UTC)
    throttler = TriggerThrottler(min_documents=10, cooldown_hours=6)
    db = db_with_training_history(last_train_time=now - timedelta(hours=1))

    decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.8,
        linguistic_drift=0.0,
        now=now,
    )

    assert decision.accepted is False
    assert decision.reason == "cooldown_active"


@pytest.mark.asyncio
async def test_rejects_when_daily_cap_reached():
    throttler = TriggerThrottler(min_documents=10, cooldown_hours=0, max_daily_trains=3)
    db = db_with_training_history(trains_today=3)

    decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.8,
        linguistic_drift=0.0,
    )

    assert decision.accepted is False
    assert decision.reason == "daily_cap_reached"


@pytest.mark.asyncio
async def test_accepts_dual_signal_with_priority():
    throttler = TriggerThrottler(min_documents=10, cooldown_hours=0, max_daily_trains=3)
    db = db_with_training_history(trains_today=0)

    decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.9,
        linguistic_drift=0.8,
    )

    assert decision.accepted is True
    assert decision.priority == "dual_signal_high"


@pytest.mark.asyncio
async def test_accepts_linguistic_signal_without_embedding_signal():
    # Thresholds are explicit so the case stays "linguistic only" regardless of what the
    # defaults are tuned to; embedding drift has to sit genuinely below its own gate.
    throttler = TriggerThrottler(
        min_documents=10,
        min_embedding_drift=0.08,
        min_linguistic_drift=0.6,
        cooldown_hours=0,
        max_daily_trains=3,
    )
    db = db_with_training_history(trains_today=0)

    decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.02,
        linguistic_drift=0.8,
    )

    assert decision.accepted is True
    assert decision.priority == "linguistic_drift"
