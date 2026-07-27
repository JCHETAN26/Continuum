import asyncio
import io
import json
from datetime import datetime

import structlog
from confluent_kafka import Consumer, KafkaError
from continuum_shared.config import settings
from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.errors import UniqueViolationError
from minio import Minio

logger = structlog.get_logger()


async def run_worker() -> None:
    db = Prisma()
    await db.connect()

    # settings.s3_endpoint is AnyUrl, we need string host:port
    s3_endpoint_str = (
        str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
    )
    minio_client = Minio(
        endpoint=s3_endpoint_str,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        secure=str(settings.s3_endpoint).startswith("https://"),
    )

    # Ensure bucket exists
    if not minio_client.bucket_exists(settings.s3_bucket_documents):
        minio_client.make_bucket(settings.s3_bucket_documents)

    conf = {
        "bootstrap.servers": settings.kafka_brokers,
        "group.id": "ingest-worker",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(conf)
    consumer.subscribe(["document-stream"])

    logger.info("Worker started, consuming from document-stream")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue

            msg_err = msg.error()
            if msg_err:
                if msg_err.code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error("Kafka error", error=msg_err)
                    break

            msg_val = msg.value()
            if not msg_val:
                continue

            payload = json.loads(msg_val.decode("utf-8"))
            idempotency_key = payload["idempotency_key"]

            existing = await db.document.find_unique(where={"idempotencyKey": idempotency_key})
            if existing:
                logger.debug("Document already exists, skipping", idempotency_key=idempotency_key)
                consumer.commit(asynchronous=False)
                continue

            object_key = f"{payload['source']}/{idempotency_key}.json"
            raw_bytes = json.dumps(payload).encode("utf-8")
            minio_client.put_object(
                bucket_name=settings.s3_bucket_documents,
                object_name=object_key,
                data=io.BytesIO(raw_bytes),
                length=len(raw_bytes),
                content_type="application/json",
            )

            try:
                occurred_at = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))

                await db.document.create(
                    data={
                        "externalId": payload["document_id"],
                        "idempotencyKey": idempotency_key,
                        "text": payload["text"],
                        "source": payload["source"],
                        "occurredAt": occurred_at,
                        "metadata": Json(payload["metadata"]) if payload.get("metadata") else None,
                        "contentHash": payload["content_hash"],
                        "objectKey": object_key,
                    }
                )
                logger.info("Ingested document", document_id=payload["document_id"])
            except UniqueViolationError:
                logger.debug(
                    "Document already exists (concurrent insert), skipping",
                    idempotency_key=idempotency_key,
                )
            except Exception as e:
                logger.error("Failed to insert document", error=str(e))
                # Do not commit so we can retry
                continue

            consumer.commit(asynchronous=False)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        await db.disconnect()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
