import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from confluent_kafka import Producer
from continuum_shared.config import settings
from continuum_shared.observability import get_tracer, instrument_fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from continuum_ingest.api.schema import DocumentPayload

logger = structlog.get_logger()
tracer = get_tracer("continuum-ingest")

producer_instance: Producer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global producer_instance
    conf = {
        "bootstrap.servers": settings.kafka_brokers,
        "client.id": settings.kafka_client_id,
        "acks": "all",
        "enable.idempotence": True,
    }
    producer_instance = Producer(conf)
    logger.info("Kafka producer initialized", brokers=settings.kafka_brokers)
    yield
    if producer_instance:
        producer_instance.flush()
        logger.info("Kafka producer flushed")


app = FastAPI(title="Continuum Ingest API", lifespan=lifespan)
instrument_fastapi(app, "continuum-ingest")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/v1/ingest/batch", status_code=202)
async def ingest_batch(payloads: list[DocumentPayload]) -> dict[str, Any]:
    if not producer_instance:
        raise HTTPException(status_code=500, detail="Producer not initialized")

    with tracer.start_as_current_span(
        "ingest.receive",
        attributes={"document_count": len(payloads)},
    ):
        for payload in payloads:
            await publish_document(payload)

    producer_instance.poll(0)
    return {"status": "accepted", "count": len(payloads)}


async def publish_document(payload: DocumentPayload) -> None:
    if not producer_instance:
        raise HTTPException(status_code=500, detail="Producer not initialized")

    with tracer.start_as_current_span(
        "ingest.publish",
        attributes={"document_id": payload.document_id, "source": payload.source},
    ):
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
                value=json.dumps(message).encode(),
            )
        except Exception as e:
            logger.error("Failed to enqueue message", error=str(e), document_id=payload.document_id)
            raise HTTPException(status_code=500, detail="Failed to enqueue documents")
