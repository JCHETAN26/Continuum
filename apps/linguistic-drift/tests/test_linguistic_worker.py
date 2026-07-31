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


class FakeDb:
    """Answers the worker's queries against an in-memory, newest-first document list."""

    def __init__(self, rows, *, already_analysed=False):
        self.rows = rows
        self.already_analysed = already_analysed
        self.execute_raw = AsyncMock()

    async def query_raw(self, query, *args):
        if "COUNT(*)" in query:
            return [{"total": len(self.rows)}]
        if "linguistic_windows" in query:
            return [{"present": 1}] if self.already_analysed else []
        if "OFFSET" in query:
            limit, offset = args
            return self.rows[offset : offset + limit]

        has_start = "occurred_at >=" in query
        has_end = "occurred_at <" in query
        if has_start and has_end:
            start, end, limit = args
            selected = [row for row in self.rows if start <= row["occurred_at"] < end]
        elif has_end:
            end, limit = args
            selected = [row for row in self.rows if row["occurred_at"] < end]
        else:
            (limit,) = args
            selected = list(self.rows)
        return selected[:limit]


def burst_rows(count, *, ending_at, span_seconds=60):
    """Documents arriving as one burst, newest first, like the seeded demo corpus."""
    step = timedelta(seconds=span_seconds / count)
    return [
        {
            "id": f"doc-{index}",
            "text": f"document {index} about hardware",
            "source": "mac_hardware" if index < count // 2 else "pc_hardware",
            "occurred_at": ending_at - (step * index),
        }
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_live_stream_uses_the_wall_clock_window():
    from continuum_linguistic_drift.worker import resolve_analysis_window

    now = datetime.now(UTC)
    rows = burst_rows(40, ending_at=now - timedelta(seconds=5), span_seconds=600)
    resolved = await resolve_analysis_window(
        FakeDb(rows), duration=timedelta(minutes=2), baseline_limit=1000, window_limit=1000
    )

    assert resolved is not None
    assert resolved.event_time is False


@pytest.mark.asyncio
async def test_idle_stream_falls_back_to_a_rank_split():
    """The seeded corpus arrives in about a minute, then nothing.

    Two minutes later the trailing wall-clock window is empty and stays empty, which left
    the analyser skipping every poll and the end-to-end check timing out.
    """
    from continuum_linguistic_drift.worker import resolve_analysis_window

    now = datetime.now(UTC)
    rows = burst_rows(1200, ending_at=now - timedelta(minutes=30))
    resolved = await resolve_analysis_window(
        FakeDb(rows), duration=timedelta(minutes=2), baseline_limit=1000, window_limit=1000
    )

    assert resolved is not None
    assert resolved.event_time is True
    assert len(resolved.window) == 600
    assert len(resolved.baseline) == 600
    # Bounds describe the documents analysed, not the clock.
    assert resolved.window_start == rows[599]["occurred_at"]
    assert resolved.window_start < resolved.window_end
    assert resolved.window_end < now
    # The split lands on the drifted half of the seeded corpus.
    assert {row.source for row in resolved.window} == {"mac_hardware"}


@pytest.mark.asyncio
async def test_too_few_documents_yields_no_window():
    from continuum_linguistic_drift.worker import resolve_analysis_window

    rows = burst_rows(6, ending_at=datetime.now(UTC) - timedelta(minutes=30))
    resolved = await resolve_analysis_window(
        FakeDb(rows), duration=timedelta(minutes=2), baseline_limit=1000, window_limit=1000
    )

    assert resolved is None


@pytest.mark.asyncio
async def test_idle_window_is_analysed_only_once():
    """spaCy over the whole corpus every thirty seconds, forever, is not free."""
    now = datetime.now(UTC)
    rows = burst_rows(1200, ending_at=now - timedelta(minutes=30))
    db = FakeDb(rows, already_analysed=True)
    analyzer = MagicMock()

    result = await process_linguistic_window(
        db, MagicMock(), analyzer, duration=timedelta(minutes=2)
    )

    assert result is None
    analyzer.analyze.assert_not_called()
