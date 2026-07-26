import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from confluent_kafka import Producer
from continuum_shared.config import settings
from fastapi import FastAPI, HTTPException

from continuum_ingest.api.schema import DocumentPayload

logger = structlog.get_logger()

producer_instance: Producer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global producer_instance
    conf = {
        'bootstrap.servers': settings.kafka_brokers,
        'client.id': settings.kafka_client_id,
        'acks': 'all',
        'enable.idempotence': True
    }
    producer_instance = Producer(conf)
    logger.info("Kafka producer initialized", brokers=settings.kafka_brokers)
    yield
    if producer_instance:
        producer_instance.flush()
        logger.info("Kafka producer flushed")

app = FastAPI(title="Continuum Ingest API", lifespan=lifespan)

@app.post("/v1/ingest/batch", status_code=202)
async def ingest_batch(payloads: list[DocumentPayload]) -> dict[str, Any]:
    if not producer_instance:
        raise HTTPException(status_code=500, detail="Producer not initialized")

    for payload in payloads:
        # Generate idempotency key
        key_str = f"{payload.source}:{payload.document_id}:{payload.timestamp.isoformat()}"
        idempotency_key = hashlib.sha256(key_str.encode()).hexdigest()
        content_hash = hashlib.sha256(payload.text.encode()).hexdigest()

        message = {
            "document_id": payload.document_id,
            "text": payload.text,
            "source": payload.source,
            "timestamp": payload.timestamp.isoformat(),
            "metadata": payload.metadata,
            "idempotency_key": idempotency_key,
            "content_hash": content_hash,
        }

        try:
            producer_instance.produce(
                topic="document-stream",
                key=idempotency_key.encode(),
                value=json.dumps(message).encode()
            )
        except Exception as e:
            logger.error("Failed to enqueue message", error=str(e), document_id=payload.document_id)
            raise HTTPException(status_code=500, detail="Failed to enqueue documents")

    producer_instance.poll(0)
    return {"status": "accepted", "count": len(payloads)}
