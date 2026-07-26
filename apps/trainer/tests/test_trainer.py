import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Mock dependencies before import
with patch("continuum_trainer.api.Prisma"), patch("continuum_trainer.pipeline.Prisma"), patch("continuum_trainer.api.queue_training_job"):
    from continuum_trainer.api import app
    from continuum_trainer.pipeline import _async_run_training_pipeline
    from continuum_shared.prisma.enums import ModelStatus

client = TestClient(app)

@pytest.mark.asyncio
async def test_create_model():
    with patch("continuum_trainer.api.db") as mock_db, patch("continuum_trainer.api.queue_training_job") as mock_queue:
        
        # Setup mock db response
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.version = "2026.07.26-1234"
        mock_model.baseModel = "sentence-transformers/all-MiniLM-L6-v2"
        mock_model.status = ModelStatus.DRAFT
        mock_model.artifactUri = None
        mock_model.artifactSha256 = None
        
        mock_db.modelversion.create = AsyncMock(return_value=mock_model)
        
        # Call API
        response = client.post("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id"
        assert data["status"] == "DRAFT"
        
        # Verify queue was called
        mock_queue.assert_called_once_with("test-id")

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
        
        mock_tx.modelversion.update = AsyncMock(return_value=updated_model)
        
        # Tx context manager
        mock_db.tx.return_value.__aenter__.return_value = mock_tx
        
        response = client.post(f"/v1/models/{mock_model.version}/activate")
        assert response.status_code == 200
        assert response.json()["status"] == "ACTIVE"

@pytest.mark.asyncio
async def test_pipeline_execution():
    with patch("continuum_trainer.pipeline.Prisma") as MockPrisma, \
         patch("continuum_trainer.pipeline._run_hf_lora_training", new_callable=AsyncMock) as mock_hf, \
         patch("continuum_trainer.pipeline.evaluate_model", new_callable=AsyncMock) as mock_eval:
         
        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.disconnect = AsyncMock()
        
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.version = "test-version"
        mock_model.baseModel = "base"
        mock_db.modelversion.find_unique = AsyncMock(return_value=mock_model)
        mock_db.modelversion.update = AsyncMock()
        
        MockPrisma.return_value = mock_db
        
        mock_eval.return_value = True # Passes eval
        
        await _async_run_training_pipeline("test-id")
        
        # Ensure HF logic was called
        mock_hf.assert_called_once_with("base")
        
        # Ensure eval was called
        mock_eval.assert_called_once_with("test-version")
        
        # Ensure final update set status to PASSED
        args, kwargs = mock_db.modelversion.update.call_args
        assert kwargs["data"]["status"] == ModelStatus.PASSED
