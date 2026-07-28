from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from continuum_shared.prisma.enums import TrainingJobStatus
from continuum_trainer.worker import create_training_run, run_training_with_retries


def model_mock() -> MagicMock:
    model = MagicMock()
    model.id = "model-id"
    model.baseModel = "sentence-transformers/all-MiniLM-L6-v2"
    return model


def job_mock(
    job_id: str = "job-id",
    *,
    attempts: int = 0,
    max_attempts: int = 3,
) -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.attempts = attempts
    job.maxAttempts = max_attempts
    job.queuedAt = datetime.now(UTC)
    return job


@pytest.mark.asyncio
async def test_create_training_run_ignores_duplicate_drift_window():
    db = MagicMock()
    db.trainingjob.find_first = AsyncMock(return_value=job_mock("existing-job"))
    db.modelversion.create = AsyncMock()
    db.trainingjob.create = AsyncMock()

    created = await create_training_run(db, "window-id")

    assert created is None
    db.modelversion.create.assert_not_called()
    db.trainingjob.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_training_run_respects_cooldown(monkeypatch):
    db = MagicMock()
    db.trainingjob.find_first = AsyncMock(
        side_effect=[
            None,
            None,
            job_mock("recent-job"),
        ]
    )
    db.modelversion.create = AsyncMock()
    db.trainingjob.create = AsyncMock()
    monkeypatch.setattr("continuum_trainer.worker.settings.retrain_cooldown_minutes", 15)

    created = await create_training_run(db, "new-window")

    assert created is None
    db.modelversion.create.assert_not_called()
    assert db.trainingjob.find_first.await_count == 3


@pytest.mark.asyncio
async def test_create_training_run_records_policy_in_hyperparameters(monkeypatch):
    db = MagicMock()
    db.trainingjob.find_first = AsyncMock(side_effect=[None, None, None])
    db.modelversion.create = AsyncMock(return_value=model_mock())
    db.trainingjob.create = AsyncMock(return_value=job_mock())
    monkeypatch.setattr("continuum_trainer.worker.settings.retrain_cooldown_minutes", 10)
    monkeypatch.setattr("continuum_trainer.worker.settings.trainer_backend", "peft")

    created = await create_training_run(db, "window-id", domain_tag="healthcare")

    assert created == ("model-id", "job-id")
    hyperparameters = db.trainingjob.create.call_args.kwargs["data"]["hyperparameters"]
    assert hyperparameters.data["domain_tag"] == "healthcare"
    assert hyperparameters.data["demo"] is False
    assert hyperparameters.data["cooldown_minutes"] == 10


@pytest.mark.asyncio
async def test_run_training_with_retries_succeeds_after_backoff(monkeypatch):
    db = MagicMock()
    db.trainingjob.find_unique = AsyncMock(
        side_effect=[
            job_mock(attempts=1, max_attempts=3),
            job_mock(attempts=2, max_attempts=3),
        ]
    )
    db.trainingjob.update = AsyncMock()
    runner = AsyncMock(side_effect=[RuntimeError("first"), RuntimeError("second"), None])
    sleep = AsyncMock()
    monkeypatch.setattr("continuum_trainer.worker.asyncio.sleep", sleep)
    monkeypatch.setattr("continuum_trainer.worker.settings.training_retry_base_seconds", 0.0)

    await run_training_with_retries(db, "model-id", "job-id", runner=runner)

    assert runner.await_count == 3
    db.trainingjob.update.assert_not_called()


@pytest.mark.asyncio
async def test_run_training_with_retries_marks_failed_after_max_attempts(monkeypatch):
    db = MagicMock()
    db.trainingjob.find_unique = AsyncMock(return_value=job_mock(attempts=3, max_attempts=3))
    db.trainingjob.update = AsyncMock()
    runner = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr("continuum_trainer.worker.settings.training_retry_base_seconds", 0.0)

    await run_training_with_retries(db, "model-id", "job-id", runner=runner)

    db.trainingjob.update.assert_awaited_once()
    data = db.trainingjob.update.call_args.kwargs["data"]
    assert data["status"] == TrainingJobStatus.FAILED
    assert data["error"] == "boom"
