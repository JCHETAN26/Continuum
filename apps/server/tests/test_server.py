import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Mock out DB before import
with patch("continuum_server.engine.Prisma"):
    from continuum_server.api import app
    from continuum_server.engine import engine
    from continuum_server.grpc_server import EmbedServiceServicer
    from continuum_server.grpc_gen import embed_pb2

client = TestClient(app)

@pytest.fixture
def mock_engine_state():
    engine.current_version = "test-version"
    engine.dimension = 384
    return engine

def test_health(mock_engine_state):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "active_version": "test-version"}

def test_embed_rest_unauthorized():
    response = client.post("/v1/embed", json={"texts": ["hello"]})
    assert response.status_code == 401

def test_embed_rest_success(mock_engine_state):
    headers = {"x-api-key": "continuum-secret-key"}
    response = client.post("/v1/embed", json={"texts": ["hello", "world"]}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model_version_used"] == "test-version"
    assert data["dimension"] == 384
    assert len(data["embeddings"]) == 2

def test_embed_rest_batch_too_large(mock_engine_state):
    headers = {"x-api-key": "continuum-secret-key"}
    texts = ["hello"] * 33
    response = client.post("/v1/embed", json={"texts": texts}, headers=headers)
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_embed_grpc_success(mock_engine_state):
    servicer = EmbedServiceServicer()
    request = embed_pb2.EmbedBatchRequest(texts=["hello", "world"])
    context = MagicMock()
    
    response = await servicer.EmbedBatch(request, context)
    
    assert response.model_version_used == "test-version"
    assert response.dimension == 384
    assert len(response.embeddings) == 2
