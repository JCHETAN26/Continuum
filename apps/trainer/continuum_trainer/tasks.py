import structlog
from continuum_shared.config import settings
from redis import Redis
from rq import Queue

logger = structlog.get_logger()

# Setup Redis connection and queue
try:
    redis_conn = Redis.from_url(str(settings.redis_url))
    q = Queue("training", connection=redis_conn)
except Exception as e:
    logger.error("Failed to connect to Redis", error=str(e))
    q = None


def queue_training_job(model_id: str):
    """Enqueues a training job for a given model ID"""
    if q is None:
        logger.warning("Redis queue is not available, running job synchronously for testing.")
        from continuum_trainer.pipeline import run_training_pipeline

        run_training_pipeline(model_id)
        return

    logger.info("Enqueuing training job", model_id=model_id)
    # The actual function must be importable by the worker
    q.enqueue("continuum_trainer.pipeline.run_training_pipeline", model_id, job_timeout=3600)
