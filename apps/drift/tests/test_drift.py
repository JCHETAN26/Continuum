from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from continuum_shared.config import settings

# Mock dependencies before import
with patch("continuum_drift.worker.Prisma"):
    from continuum_drift.api import DriftSummaryResponse, DriftWindowResponse, compute_projection
    from continuum_drift.worker import (
        align_window_end,
        compute_centroid,
        get_or_create_baseline,
        process_window,
    )


@pytest.mark.asyncio
async def test_compute_centroid():
    """Postgres returns one already-averaged row; the worker only parses it.

    The averaging moved into the database, so the window no longer arrives here as one
    text blob per document. tests/integration/test_centroid_avg.py checks against a real
    pgvector that AVG() agrees with the mean this used to compute in Python.
    """
    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(return_value=[{"centroid": "[2.0,2.0,2.0]", "document_count": 2}])

    start = datetime.now(UTC)
    end = start + timedelta(minutes=5)

    result = await compute_centroid(mock_db, start, end)

    assert result is not None
    centroid, count = result
    assert count == 2
    assert np.allclose(centroid, np.array([2.0, 2.0, 2.0]))
    # One row regardless of window size, rather than one per document.
    statement = " ".join(mock_db.query_raw.call_args.args[0].split())
    assert "AVG(vector)" in statement


@pytest.mark.asyncio
async def test_compute_centroid_returns_none_for_an_empty_window():
    """AVG over no rows is NULL, and a null centroid is not the origin."""
    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(return_value=[{"centroid": None, "document_count": 0}])

    result = await compute_centroid(mock_db, datetime.now(UTC), datetime.now(UTC))

    assert result is None


@pytest.mark.asyncio
async def test_process_window_creates_alert(monkeypatch):
    mock_db = MagicMock()

    # Simulate existing window = None
    mock_db.driftwindow = MagicMock()
    mock_db.driftwindow.find_unique = AsyncMock(return_value=None)

    # Baseline id and centroid
    async def mock_get_baseline(*args, **kwargs):
        return ("baseline_id", np.array([1.0, 1.0, 1.0]))

    monkeypatch.setattr("continuum_drift.worker.get_or_create_baseline", mock_get_baseline)

    # Compute centroid returns a very different vector
    async def mock_compute(*args, **kwargs):
        return np.array([-1.0, -1.0, -1.0]), 3

    monkeypatch.setattr("continuum_drift.worker.compute_centroid", mock_compute)

    mock_db.execute_raw = AsyncMock()

    mock_producer = MagicMock()

    async def mock_linguistic_score(*args, **kwargs):
        return 0.8

    monkeypatch.setattr(
        "continuum_drift.worker.latest_linguistic_drift_score", mock_linguistic_score
    )
    monkeypatch.setattr(
        "continuum_drift.worker.TriggerThrottler.from_settings",
        lambda: MagicMock(
            decide=AsyncMock(
                return_value=MagicMock(
                    accepted=True,
                    reason="accepted",
                    priority="dual_signal_high",
                )
            )
        ),
    )

    await process_window(mock_db, mock_producer, "FIVE_MIN", timedelta(minutes=5))

    # Should insert into drift_windows
    mock_db.execute_raw.assert_called_once()

    # Should produce an alert since distance is high
    mock_producer.produce.assert_called_once()

    args, kwargs = mock_producer.produce.call_args
    assert args[0] == "drift-alerts"
    assert b"FIVE_MIN" in kwargs["key"]
    assert b"drift_score" in kwargs["value"]


def test_compute_projection_uses_pca_for_small_samples():
    points, method = compute_projection([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert method == "pca"
    assert len(points) == 2
    assert len(points[0]) == 2


def test_compute_projection_uses_tsne_for_demo_samples():
    vectors = [[float(index == dimension) for dimension in range(6)] for index in range(6)]

    points, method = compute_projection(vectors)

    assert method == "tsne"
    assert len(points) == 6


@pytest.mark.asyncio
async def test_drift_event_payload_contains_windows_summary_and_projection(monkeypatch):
    from continuum_drift import api

    now = datetime.now(UTC)

    async def mock_status():
        return [
            DriftWindowResponse(
                id="window-id",
                windowSize="FIVE_MIN",
                windowStart=now,
                windowEnd=now + timedelta(minutes=2),
                documentCount=42,
                driftScore=0.67,
                wassersteinDistance=0.67,
                threshold=0.35,
                breached=True,
                baselineId="baseline-id",
                createdAt=now,
            )
        ]

    async def mock_summary():
        return DriftSummaryResponse(
            documentCount=100,
            embeddingCount=95,
            latestDriftScore=0.67,
            breached=True,
            threshold=0.35,
        )

    async def mock_projection():
        return {
            "method": "pca",
            "points": [{"x": 0.1, "y": -0.2, "source": "medical", "label": "doc"}],
        }

    monkeypatch.setattr(api, "get_drift_status", mock_status)
    monkeypatch.setattr(api, "get_drift_summary", mock_summary)
    monkeypatch.setattr(api, "get_embedding_projection", mock_projection)

    payload = await api.get_drift_event_payload()

    assert payload["windows"][0]["id"] == "window-id"
    assert payload["summary"]["breached"] is True
    assert payload["projection"]["method"] == "pca"


def test_short_windows_tumble_instead_of_overlapping():
    """Consecutive short windows must not share documents.

    Windows used to be two minutes wide and advance once a minute, so half of every window
    was the previous one. A distribution that changed arrived averaged with the one it
    replaced, which cost a window before the score crossed the threshold.
    """
    duration = timedelta(seconds=40)
    origin = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    ends = [
        align_window_end(origin + timedelta(seconds=offset), duration)
        for offset in (0, 13, 39, 40, 41, 79, 80)
    ]

    assert ends[0] == ends[1] == ends[2] == datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    assert ends[3] == ends[4] == ends[5] == datetime(2026, 8, 2, 12, 0, 40, tzinfo=UTC)
    assert ends[6] == datetime(2026, 8, 2, 12, 1, 20, tzinfo=UTC)

    # Distinct boundaries, so [start, end) of one window is [end, ...) of the next.
    assert ends[3] - ends[0] == duration


def test_hour_window_still_slides_every_minute():
    """Tumbling the hour window would emit one row an hour and starve a short run."""
    moment = datetime(2026, 8, 2, 12, 34, 56, tzinfo=UTC)

    assert align_window_end(moment, timedelta(minutes=1)) == moment.replace(second=0, microsecond=0)


@pytest.mark.asyncio
async def test_baseline_ignores_a_window_too_small_to_describe_a_distribution():
    """A run starting just before a boundary leaves a sliver in the first window.

    Taking it as the baseline would measure every later window against noise, which short
    tumbling windows make much easier to hit than two-minute sliding ones did.
    """
    mock_db = MagicMock()
    mock_db.driftwindow.find_first = AsyncMock(return_value=None)
    mock_db.query_raw = AsyncMock(return_value=[{"vec_str": "[1.0,0.0,0.0]"}])
    mock_db.execute_raw = AsyncMock()

    await get_or_create_baseline(mock_db)

    where = mock_db.driftwindow.find_first.call_args.kwargs["where"]
    assert where["documentCount"] == {"gte": settings.drift_trigger_min_documents}
