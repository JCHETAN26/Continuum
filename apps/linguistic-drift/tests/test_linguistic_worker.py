from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from continuum_linguistic_drift.schemas import LinguisticDriftReport
from continuum_linguistic_drift.worker import (
    fetch_documents,
    insert_linguistic_window,
    process_linguistic_window,
)


@pytest.mark.asyncio
async def test_fetch_documents_builds_window_query():
    db = MagicMock()
    db.query_raw = AsyncMock(
        return_value=[
            {
                "id": "doc-id",
                "text": "Acme Search cache incident",
                "source": "software",
                "occurred_at": datetime.now(UTC),
            }
        ]
    )

    docs = await fetch_documents(
        db,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        limit=50,
    )

    assert docs[0].id == "doc-id"
    query = db.query_raw.call_args.args[0]
    assert "occurred_at >=" in query
    assert "occurred_at <" in query


@pytest.mark.asyncio
async def test_insert_linguistic_window_persists_report():
    db = MagicMock()
    db.execute_raw = AsyncMock()
    db.query_raw = AsyncMock(return_value=[{"id": "persisted-window"}])
    report = LinguisticDriftReport(
        document_count=3,
        entity_kl_divergence=0.5,
        topic_wasserstein=0.6,
        vocab_chi2_pvalue=0.01,
        composite_score=0.72,
        threshold=0.65,
        breached=True,
        new_entities=[],
        emerging_topics=[],
        emerging_terms=[],
    )

    window_id = await insert_linguistic_window(
        db, datetime.now(UTC) - timedelta(minutes=2), datetime.now(UTC), report
    )

    assert window_id == "persisted-window"
    db.execute_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_linguistic_window_emits_alert(monkeypatch):
    db = MagicMock()
    producer = MagicMock()
    now = datetime.now(UTC)
    baseline_docs = [
        {
            "id": str(index),
            "text": f"Acme Search cache query {index}",
            "source": "software",
            "occurred_at": now,
        }
        for index in range(10)
    ]
    window_docs = [
        {
            "id": "w1",
            "text": "Cardiology Clinic insulin medication",
            "source": "medical",
            "occurred_at": now,
        },
        {
            "id": "w2",
            "text": "Cardiology Clinic patient diagnosis",
            "source": "medical",
            "occurred_at": now,
        },
    ]
    db.query_raw = AsyncMock(side_effect=[baseline_docs, window_docs, [{"id": "drift-id"}], []])
    db.execute_raw = AsyncMock()

    async def mock_insert(*args, **kwargs):
        return "linguistic-window-id"

    monkeypatch.setattr("continuum_linguistic_drift.worker.insert_linguistic_window", mock_insert)

    class FakeAnalyzer:
        def analyze(self, baseline, window):
            assert len(baseline) == 10
            assert len(window) == 2
            return LinguisticDriftReport(
                document_count=2,
                entity_kl_divergence=0.7,
                topic_wasserstein=0.7,
                vocab_chi2_pvalue=0.01,
                composite_score=0.8,
                threshold=0.65,
                breached=True,
                new_entities=[],
                emerging_topics=[],
                emerging_terms=[],
            )

    await process_linguistic_window(db, producer, FakeAnalyzer(), duration=timedelta(minutes=2))

    producer.produce.assert_called_once()
    assert producer.produce.call_args.args[0] == "linguistic-drift-alerts"
