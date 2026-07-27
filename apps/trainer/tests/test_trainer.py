from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

with patch("continuum_trainer.api.Prisma"), patch("continuum_trainer.pipeline.Prisma"):
    from continuum_shared.prisma.enums import ModelStatus
    from continuum_trainer.api import ModelVersionResponse, TrainingJobResponse, app
    from continuum_trainer.pipeline import _async_run_training_pipeline

client = TestClient(app)


@pytest.mark.asyncio
async def test_create_model():
    with (
        patch("continuum_trainer.api.db") as mock_db,
        patch("continuum_trainer.api.run_training_pipeline") as mock_pipeline,
    ):
        # Setup mock db response
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.version = "2026.07.26-1234"
        mock_model.baseModel = "sentence-transformers/all-MiniLM-L6-v2"
        mock_model.status = ModelStatus.DRAFT
        mock_model.artifactUri = None
        mock_model.artifactSha256 = None
        mock_model.metrics = None
        mock_model.baselineMetrics = None
        mock_model.improvementPct = None

        mock_job = MagicMock()
        mock_job.id = "job-id"

        mock_db.modelversion.create = AsyncMock(return_value=mock_model)
        mock_db.trainingjob.create = AsyncMock(return_value=mock_job)

        # Call API
        response = client.post("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id"
        assert data["status"] == "DRAFT"

        mock_pipeline.assert_called_once_with("test-id", "job-id")


@pytest.mark.asyncio
async def test_activate_model():
    with patch("continuum_trainer.api.db") as mock_db:
        # Setup mock db responses
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.version = "2026.07.26-1234"
        mock_model.status = ModelStatus.PASSED
        mock_model.baseModel = "sentence-transformers/all-MiniLM-L6-v2"
        mock_model.artifactUri = "s3://..."
        mock_model.artifactSha256 = "..."
        mock_model.metrics = None
        mock_model.baselineMetrics = None
        mock_model.improvementPct = None

        mock_db.modelversion.find_unique = AsyncMock(return_value=mock_model)

        # Mock transaction
        mock_tx = MagicMock()
        mock_tx.modelversion.find_many = AsyncMock(return_value=[])

        updated_model = MagicMock()
        updated_model.id = mock_model.id
        updated_model.version = mock_model.version
        updated_model.status = ModelStatus.ACTIVE
        updated_model.baseModel = mock_model.baseModel
        updated_model.artifactUri = mock_model.artifactUri
        updated_model.artifactSha256 = mock_model.artifactSha256
        updated_model.metrics = None
        updated_model.baselineMetrics = None
        updated_model.improvementPct = None

        mock_tx.modelversion.update = AsyncMock(return_value=updated_model)

        # Tx context manager
        mock_db.tx.return_value.__aenter__.return_value = mock_tx

        response = client.post(f"/v1/models/{mock_model.version}/activate")
        assert response.status_code == 200
        assert response.json()["status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_pipeline_execution():
    with (
        patch("continuum_trainer.pipeline.Prisma") as mock_prisma_cls,
        patch(
            "continuum_trainer.pipeline._run_corpus_adapter_training", new_callable=AsyncMock
        ) as mock_training,
        patch("continuum_trainer.pipeline.evaluate_model", new_callable=AsyncMock) as mock_eval,
        patch(
            "continuum_trainer.pipeline._export_trained_artifact", new_callable=AsyncMock
        ) as mock_export,
    ):
        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.disconnect = AsyncMock()

        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.version = "test-version"
        mock_model.baseModel = "base"
        mock_db.modelversion.find_unique = AsyncMock(return_value=mock_model)
        mock_db.modelversion.update = AsyncMock()

        mock_prisma_cls.return_value = mock_db

        mock_eval.return_value = True  # Passes eval
        mock_training.return_value = (
            {"sample_count": 500, "loss_history": [{"step": 10, "loss": 1.0}]},
            b"onnx-bytes",
            [{"source": "software", "vector": [1.0, 0.0]}],
        )
        mock_eval.return_value = (
            True,
            {"mrr": 0.72},
            {"mrr": 0.58},
            0.24,
        )
        mock_export.return_value = ("s3://continuum-models/test/model.json", "a" * 64, 512)

        await _async_run_training_pipeline("test-id")

        mock_training.assert_called_once()

        # Ensure eval was called
        mock_eval.assert_called_once_with(
            "test-version", b"onnx-bytes", [{"source": "software", "vector": [1.0, 0.0]}]
        )

        # Ensure final update set status to PASSED
        args, kwargs = mock_db.modelversion.update.call_args
        assert kwargs["data"]["status"] == ModelStatus.PASSED


@pytest.mark.asyncio
async def test_training_event_payload_contains_models_and_jobs(monkeypatch):
    from datetime import UTC, datetime

    from continuum_trainer import api

    now = datetime.now(UTC)

    async def mock_models():
        return [
            ModelVersionResponse(
                id="model-id",
                version="2026.07.26-demo",
                baseModel="sentence-transformers/all-MiniLM-L6-v2",
                status="PASSED",
                artifactUri="s3://continuum-models/demo/model.json",
                artifactSha256="a" * 64,
                metrics={"mrr": 0.72},
                baselineMetrics={"mrr": 0.58},
                improvementPct=0.24,
            )
        ]

    async def mock_jobs():
        return [
            TrainingJobResponse(
                id="job-id",
                status="SUCCEEDED",
                trigger="DRIFT_ALERT",
                modelVersionId="model-id",
                driftWindowId="window-id",
                sampleCount=500,
                lossHistory=[{"step": 1, "loss": 0.5}],
                queuedAt=now,
                startedAt=now,
                finishedAt=now,
            )
        ]

    monkeypatch.setattr(api, "list_models", mock_models)
    monkeypatch.setattr(api, "list_training_jobs", mock_jobs)

    payload = await api.get_training_event_payload()

    assert payload["models"][0]["version"] == "2026.07.26-demo"
    assert payload["models"][0]["metrics"]["mrr"] == 0.72
    assert payload["jobs"][0]["lossHistory"][0]["loss"] == 0.5
