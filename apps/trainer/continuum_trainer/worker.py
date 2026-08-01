import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import structlog
from confluent_kafka import Consumer, KafkaError
from continuum_shared.config import settings
from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.enums import ModelStatus, TrainingJobStatus, TrainingTrigger

from continuum_trainer.pipeline import _async_run_training_pipeline

logger = structlog.get_logger()

PipelineRunner = Callable[[str, str], Awaitable[None]]


async def create_training_run(
    db: Prisma, drift_window_id: str | None, *, domain_tag: str | None = None
) -> tuple[str, str] | None:
    if drift_window_id:
        existing_window_job = await db.trainingjob.find_first(
            where={
                "trigger": TrainingTrigger.DRIFT_ALERT,
                "driftWindowId": drift_window_id,
            },
            order={"queuedAt": "desc"},
        )
        if existing_window_job:
            logger.info(
                "Training job already exists for drift window",
                drift_window_id=drift_window_id,
                job_id=existing_window_job.id,
            )
            return None

    active_job = await db.trainingjob.find_first(
        where={"status": {"in": [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]}},
        order={"queuedAt": "desc"},
    )
    if active_job:
        logger.info("Training already in progress, ignoring duplicate alert", job_id=active_job.id)
        return None

    if settings.retrain_cooldown_minutes > 0:
        cooldown_start = datetime.now(UTC) - timedelta(minutes=settings.retrain_cooldown_minutes)
        recent_job = await db.trainingjob.find_first(
            where={
                "trigger": TrainingTrigger.DRIFT_ALERT,
                "queuedAt": {"gte": cooldown_start},
            },
            order={"queuedAt": "desc"},
        )
        if recent_job:
            logger.info(
                "Retrain cooldown active, ignoring drift alert",
                job_id=recent_job.id,
                cooldown_minutes=settings.retrain_cooldown_minutes,
            )
            return None

    version = f"{datetime.now(UTC).strftime('%Y.%m.%d')}-{str(uuid4())[:8]}"
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
            "hyperparameters": Json(
                {
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "demo": settings.trainer_backend == "demo_adapter",
                    "domain_tag": domain_tag,
                    "cooldown_minutes": settings.retrain_cooldown_minutes,
                }
            ),
        }
    )
    return model.id, job.id


async def reclaim_abandoned_jobs(
    db: Prisma, *, runner: PipelineRunner = _async_run_training_pipeline
) -> list[str]:
    """Re-run training jobs left RUNNING by a worker that died mid-run.

    The worker is driven entirely by Kafka and never polls for QUEUED work, so a job whose
    process disappeared stayed RUNNING forever: restart:unless-stopped brought the container
    back, create_training_run saw a job already existed for that drift window and returned
    None, and nothing ever picked it up again. An end-to-end run sat on that for its full
    1800s timeout after the trainer was killed for exceeding its memory limit.

    A RUNNING job at startup can only be abandoned, because this container is a singleton
    and is the only thing that executes jobs. Attempts are bounded by maxAttempts, so a run
    that fails deterministically ends up FAILED rather than looping.
    """
    abandoned = await db.trainingjob.find_many(
        where={"status": TrainingJobStatus.RUNNING}, order={"queuedAt": "asc"}
    )
    resumed: list[str] = []
    for job in abandoned:
        if job.attempts >= job.maxAttempts:
            await db.trainingjob.update(
                where={"id": job.id},
                data={
                    "status": TrainingJobStatus.FAILED,
                    "error": "abandoned by a worker restart after exhausting attempts",
                    "finishedAt": datetime.now(UTC),
                },
            )
            logger.warning("Abandoned training job exhausted its attempts", job_id=job.id)
            continue

        logger.info(
            "Resuming a training job abandoned by a worker restart",
            job_id=job.id,
            attempts=job.attempts,
        )
        if job.modelVersionId:
            await run_training_with_retries(db, job.modelVersionId, job.id, runner=runner)
            resumed.append(job.id)
    return resumed


async def run_training_with_retries(
    db: Prisma,
    model_id: str,
    job_id: str,
    *,
    runner: PipelineRunner = _async_run_training_pipeline,
) -> None:
    while True:
        try:
            await runner(model_id, job_id)
            return
        except Exception as exc:
            job = await db.trainingjob.find_unique(where={"id": job_id})
            attempts = job.attempts if job else 1
            max_attempts = job.maxAttempts if job else 1
            if attempts >= max_attempts:
                await db.trainingjob.update(
                    where={"id": job_id},
                    data={
                        "status": TrainingJobStatus.FAILED,
                        "finishedAt": datetime.now(UTC),
                        "error": str(exc),
                    },
                )
                logger.exception("Training failed after max attempts", job_id=job_id)
                return

            delay = settings.training_retry_base_seconds * (2 ** max(0, attempts - 1))
            logger.warning(
                "Training failed, retrying after backoff",
                job_id=job_id,
                attempts=attempts,
                max_attempts=max_attempts,
                delay_seconds=delay,
                error=str(exc),
            )
            if delay > 0:
                await asyncio.sleep(delay)


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

    # Before taking new alerts, finish anything a previous process left mid-run. Otherwise
    # the dedupe check below sees a RUNNING job for that drift window and skips it forever.
    await reclaim_abandoned_jobs(db)
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
            created = await create_training_run(
                db,
                payload.get("window_id"),
                domain_tag=payload.get("domain_tag") or payload.get("source"),
            )
            if created:
                model_id, job_id = created
                logger.info("Drift alert created training run", model_id=model_id, job_id=job_id)
                await run_training_with_retries(db, model_id, job_id)
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
