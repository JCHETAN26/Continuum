from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from confluent_kafka import Producer
from continuum_shared.config import settings
from continuum_shared.prisma import Prisma

from continuum_linguistic_drift.analyzer import LinguisticDriftAnalyzer
from continuum_linguistic_drift.schemas import DocumentForAnalysis, LinguisticDriftReport

logger = structlog.get_logger()


def get_kafka_producer():
    conf = {
        "bootstrap.servers": settings.kafka_brokers,
        "client.id": f"{settings.kafka_client_id}-linguistic-drift-worker",
    }
    return Producer(conf)


async def fetch_documents(
    db: Prisma, start_time: datetime | None, end_time: datetime | None, *, limit: int
) -> list[DocumentForAnalysis]:
    clauses = []
    args: list[Any] = []
    if start_time is not None:
        args.append(start_time)
        clauses.append(f"occurred_at >= ${len(args)}::timestamptz")
    if end_time is not None:
        args.append(end_time)
        clauses.append(f"occurred_at < ${len(args)}::timestamptz")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    rows = await db.query_raw(
        f"""
        SELECT id::text, text, source, occurred_at
        FROM documents
        {where}
        ORDER BY occurred_at DESC
        LIMIT ${len(args)}
        """,
        *args,
    )
    return [
        DocumentForAnalysis(
            id=row["id"],
            text=row["text"],
            source=row["source"],
            occurred_at=row["occurred_at"],
        )
        for row in rows
    ]


async def latest_semantic_drift_window_id(db: Prisma) -> str | None:
    rows = await db.query_raw(
        """
        SELECT id::text
        FROM drift_windows
        WHERE breached = true
        ORDER BY window_start DESC
        LIMIT 1
        """
    )
    return rows[0]["id"] if rows else None


async def insert_linguistic_window(
    db: Prisma, window_start: datetime, window_end: datetime, report: LinguisticDriftReport
) -> str:
    window_id = str(uuid.uuid4())
    await db.execute_raw(
        """
        INSERT INTO linguistic_windows (
            id,
            window_start,
            window_end,
            document_count,
            entity_kl_divergence,
            topic_wasserstein,
            vocab_chi2_pvalue,
            composite_score,
            threshold,
            breached,
            new_entities,
            emerging_topics,
            emerging_terms,
            created_at
        )
        VALUES (
            $1::uuid,
            $2::timestamptz,
            $3::timestamptz,
            $4,
            $5,
            $6,
            $7,
            $8,
            $9,
            $10,
            $11::jsonb,
            $12::jsonb,
            $13::jsonb,
            NOW()
        )
        ON CONFLICT (window_start, window_end) DO UPDATE SET
            document_count = EXCLUDED.document_count,
            entity_kl_divergence = EXCLUDED.entity_kl_divergence,
            topic_wasserstein = EXCLUDED.topic_wasserstein,
            vocab_chi2_pvalue = EXCLUDED.vocab_chi2_pvalue,
            composite_score = EXCLUDED.composite_score,
            threshold = EXCLUDED.threshold,
            breached = EXCLUDED.breached,
            new_entities = EXCLUDED.new_entities,
            emerging_topics = EXCLUDED.emerging_topics,
            emerging_terms = EXCLUDED.emerging_terms
        """,
        window_id,
        window_start,
        window_end,
        report.document_count,
        report.entity_kl_divergence,
        report.topic_wasserstein,
        report.vocab_chi2_pvalue,
        report.composite_score,
        report.threshold,
        report.breached,
        json.dumps([entity.model_dump() for entity in report.new_entities]),
        json.dumps([topic.model_dump() for topic in report.emerging_topics]),
        json.dumps([term.model_dump() for term in report.emerging_terms]),
    )
    rows = await db.query_raw(
        """
        SELECT id::text
        FROM linguistic_windows
        WHERE window_start = $1::timestamptz AND window_end = $2::timestamptz
        """,
        window_start,
        window_end,
    )
    return rows[0]["id"] if rows else window_id


async def link_to_training_job(
    db: Prisma, linguistic_window_id: str, *, drift_window_id: str | None = None
) -> str | None:
    rows = await db.query_raw(
        """
        SELECT id::text
        FROM training_jobs
        WHERE status IN ('QUEUED', 'RUNNING')
        ORDER BY queued_at DESC
        LIMIT 1
        """
    )
    if not rows:
        return None

    training_job_id = rows[0]["id"]
    await db.execute_raw(
        """
        INSERT INTO training_linguistic_signals (
            training_job_id,
            linguistic_window_id,
            drift_window_id,
            signal_weight
        )
        VALUES ($1::uuid, $2::uuid, $3::uuid, $4)
        ON CONFLICT (training_job_id, linguistic_window_id) DO UPDATE SET
            drift_window_id = EXCLUDED.drift_window_id,
            signal_weight = EXCLUDED.signal_weight
        """,
        training_job_id,
        linguistic_window_id,
        drift_window_id,
        1.0,
    )
    return training_job_id


async def process_linguistic_window(
    db: Prisma,
    producer: Producer,
    analyzer: LinguisticDriftAnalyzer,
    *,
    duration: timedelta,
    baseline_limit: int = 1000,
    window_limit: int = 1000,
) -> str | None:
    now = datetime.now(UTC)
    window_end = now.replace(second=0, microsecond=0)
    window_start = window_end - duration

    baseline_docs = await fetch_documents(db, None, window_start, limit=baseline_limit)
    window_docs = await fetch_documents(db, window_start, window_end, limit=window_limit)
    if len(baseline_docs) < 10 or len(window_docs) < 2:
        logger.info(
            "Not enough documents for linguistic drift",
            baseline_count=len(baseline_docs),
            window_count=len(window_docs),
        )
        return None

    report = analyzer.analyze(baseline_docs, window_docs)
    window_id = await insert_linguistic_window(db, window_start, window_end, report)
    logger.info(
        "Processed linguistic window",
        window_id=window_id,
        composite_score=report.composite_score,
        breached=report.breached,
    )

    if report.breached:
        drift_window_id = await latest_semantic_drift_window_id(db)
        training_job_id = await link_to_training_job(db, window_id, drift_window_id=drift_window_id)
        alert = {
            "window_id": window_id,
            "drift_window_id": drift_window_id,
            "training_job_id": training_job_id,
            "composite_score": report.composite_score,
            "threshold": report.threshold,
            "timestamp": now.isoformat(),
            "new_entities": [entity.model_dump() for entity in report.new_entities],
            "emerging_terms": [term.model_dump() for term in report.emerging_terms],
        }
        producer.produce(
            "linguistic-drift-alerts",
            key=window_id.encode("utf-8"),
            value=json.dumps(alert).encode("utf-8"),
        )
        producer.poll(0)
        logger.warning("Linguistic drift threshold breached.", alert=alert)

    return window_id


async def run_linguistic_drift_worker():
    db = Prisma()
    await db.connect()
    producer = get_kafka_producer()
    analyzer = LinguisticDriftAnalyzer(threshold=settings.linguistic_drift_threshold)
    duration = timedelta(minutes=settings.linguistic_drift_window_minutes)

    logger.info("Linguistic drift worker started.")
    try:
        while True:
            await process_linguistic_window(db, producer, analyzer, duration=duration)
            producer.flush(timeout=1.0)
            await asyncio.sleep(settings.linguistic_drift_poll_seconds)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        await db.disconnect()
