from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Mock dependencies before import
with patch("continuum_drift.worker.Prisma"):
    from continuum_drift.api import DriftSummaryResponse, DriftWindowResponse, compute_projection
    from continuum_drift.worker import compute_centroid, process_window


@pytest.mark.asyncio
async def test_compute_centroid():
    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(
        return_value=[{"vec_str": "[1.0, 2.0, 3.0]"}, {"vec_str": "[3.0, 2.0, 1.0]"}]
    )

    start = datetime.now(UTC)
    end = start + timedelta(minutes=5)

    result = await compute_centroid(mock_db, start, end)

    assert result is not None
    centroid, count = result
    assert count == 2
    assert np.allclose(centroid, np.array([2.0, 2.0, 2.0]))


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
