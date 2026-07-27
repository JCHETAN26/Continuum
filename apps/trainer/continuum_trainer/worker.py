import asyncio
import json
from datetime import datetime
from uuid import uuid4

import structlog
from confluent_kafka import Consumer, KafkaError
from continuum_shared.config import settings
from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.enums import ModelStatus, TrainingJobStatus, TrainingTrigger

from continuum_trainer.pipeline import _async_run_training_pipeline

logger = structlog.get_logger()


async def create_training_run(db: Prisma, drift_window_id: str | None) -> tuple[str, str] | None:
    active_job = await db.trainingjob.find_first(
        where={"status": {"in": [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]}},
        order={"queuedAt": "desc"},
    )
    if active_job:
        logger.info("Training already in progress, ignoring duplicate alert", job_id=active_job.id)
        return None

    version = f"{datetime.utcnow().strftime('%Y.%m.%d')}-{str(uuid4())[:8]}"
    model = await db.modelversion.create(
        data={
            "version": version,
            "baseModel": settings.embedding_model,
            "status": ModelStatus.DRAFT,
        }
    )
    job = await db.trainingjob.create(
        data={
            "status": TrainingJobStatus.QUEUED,
            "trigger": TrainingTrigger.DRIFT_ALERT,
            "driftWindowId": drift_window_id,
            "modelVersionId": model.id,
            "baseModel": model.baseModel,
            "hyperparameters": Json({"lora_rank": 8, "lora_alpha": 16, "demo": True}),
        }
    )
    return model.id, job.id


async def run_alert_worker() -> None:
    db = Prisma()
    await db.connect()

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_brokers,
            "group.id": "trainer-alert-worker",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe(["drift-alerts"])
    logger.info("Trainer worker started, consuming drift-alerts")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                await asyncio.sleep(0.2)
                continue

            msg_err = msg.error()
            if msg_err:
                if msg_err.code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka error", error=msg_err)
                await asyncio.sleep(1.0)
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            created = await create_training_run(db, payload.get("window_id"))
            if created:
                model_id, job_id = created
                logger.info("Drift alert created training run", model_id=model_id, job_id=job_id)
                await _async_run_training_pipeline(model_id, job_id)
            consumer.commit(asynchronous=False)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        await db.disconnect()


def main() -> None:
    asyncio.run(run_alert_worker())


if __name__ == "__main__":
    main()
