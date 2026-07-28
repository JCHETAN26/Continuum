import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from continuum_drift.throttle import TriggerThrottler

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_THROTTLE_INTEGRATION") != "1",
    reason="Set RUN_THROTTLE_INTEGRATION=1 to run the throttling integration contract.",
)
@pytest.mark.asyncio
async def test_second_immediate_trigger_rejected_by_cooldown():
    now = datetime.now(UTC)
    throttler = TriggerThrottler(min_documents=10, cooldown_hours=6, max_daily_trains=3)
    db = MagicMock()
    db.trainingjob.find_first = AsyncMock(return_value=None)
    db.query_raw = AsyncMock(return_value=[{"count": 1}])
    first_decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.9,
        linguistic_drift=0.7,
        now=now,
    )

    job = MagicMock()
    job.queuedAt = now
    db.trainingjob.find_first = AsyncMock(return_value=job)
    second_decision = await throttler.decide(
        db,
        document_count=100,
        embedding_drift=0.9,
        linguistic_drift=0.7,
        now=now + timedelta(seconds=1),
    )

    assert first_decision.accepted is True
    assert second_decision.accepted is False
    assert second_decision.reason == "cooldown_active"
