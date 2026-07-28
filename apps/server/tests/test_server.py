from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from continuum_shared.artifacts import (
    build_demo_artifact_manifest,
    encode_manifest,
    sha256_hex,
)
from continuum_shared.security import hash_api_key
from continuum_trainer.pipeline import EmbeddingAdapter, export_adapter_to_onnx
from fastapi.testclient import TestClient

# Mock out DB before import
with patch("continuum_server.engine.Prisma"):
    from continuum_server.api import app
    from continuum_server.engine import ModelEngine, engine
    from continuum_server.grpc_gen import embed_pb2
    from continuum_server.grpc_server import EmbedServiceServicer

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


def test_embed_rest_accepts_bcrypt_api_key(mock_engine_state, monkeypatch):
    monkeypatch.setenv("API_KEY_BCRYPT_HASH", hash_api_key("hashed-secret"))
    response = client.post(
        "/v1/embed",
        json={"texts": ["hello"]},
        headers={"x-api-key": "hashed-secret"},
    )

    assert response.status_code == 200


def test_list_rollbacks(mock_engine_state, monkeypatch):
    class FakeDb:
        async def query_raw(self, query: str):
            return [
                {
                    "id": "event-1",
                    "failed_version": "candidate",
                    "restored_version": "baseline",
                    "error_rate": 0.06,
                    "request_count": 120,
                    "created_at": "2026-07-28T00:00:00Z",
                }
            ]

    monkeypatch.setattr(engine, "db", FakeDb())

    response = client.get("/v1/rollbacks")

    assert response.status_code == 200
    assert response.json()[0]["failedVersion"] == "candidate"
    assert response.json()[0]["restoredVersion"] == "baseline"


def test_embed_rest_baseline_override(mock_engine_state):
    headers = {"x-api-key": "continuum-secret-key", "x-model": "baseline"}
    response = client.post("/v1/embed", json={"texts": ["hello"]}, headers=headers)

    assert response.status_code == 200
    assert response.json()["model_version_used"] == "baseline"


def test_embed_rest_inactive_model_rejected(mock_engine_state):
    headers = {"x-api-key": "continuum-secret-key", "x-model": "inactive"}
    response = client.post("/v1/embed", json={"texts": ["hello"]}, headers=headers)

    assert response.status_code == 503


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


@pytest.mark.asyncio
async def test_embed_grpc_baseline_override(mock_engine_state):
    servicer = EmbedServiceServicer()
    request = embed_pb2.EmbedBatchRequest(texts=["hello"], model_version="baseline")
    context = MagicMock()

    response = await servicer.EmbedBatch(request, context)

    assert response.model_version_used == "baseline"


@pytest.mark.asyncio
async def test_load_artifact_manifest_verifies_checksum():
    manifest = build_demo_artifact_manifest(
        version="candidate",
        base_model="continuum/hash-embedding-demo",
        embedding_dim=384,
        metrics={"mrr": 0.72},
        baseline_metrics={"mrr": 0.58},
        improvement_pct=0.24,
    )
    data = encode_manifest(manifest)
    response = MagicMock()
    response.read.return_value = data

    minio_client = MagicMock()
    minio_client.get_object.return_value = response

    with patch("continuum_server.engine.Minio", return_value=minio_client):
        loaded = await ModelEngine()._load_artifact_manifest(
            "s3://continuum-models/candidate/model-manifest.json", sha256_hex(data)
        )

    assert loaded == manifest
    minio_client.get_object.assert_called_once_with(
        "continuum-models", "candidate/model-manifest.json"
    )


@pytest.mark.asyncio
async def test_load_artifact_manifest_rejects_bad_checksum():
    response = MagicMock()
    response.read.return_value = b'{"embedding_dim":384}'

    minio_client = MagicMock()
    minio_client.get_object.return_value = response

    with patch("continuum_server.engine.Minio", return_value=minio_client):
        with pytest.raises(ValueError, match="checksum mismatch"):
            await ModelEngine()._load_artifact_manifest(
                "s3://continuum-models/candidate/model-manifest.json", "0" * 64
            )


@pytest.mark.asyncio
async def test_embed_batch_runs_loaded_onnx_session():
    onnx_artifact = export_adapter_to_onnx(EmbeddingAdapter(384), 384)
    manifest = {
        "embedding_dim": 384,
        "onnx": {
            "uri": "s3://continuum-models/candidate/adapter.onnx",
            "sha256": sha256_hex(onnx_artifact),
            "input_name": "input",
            "output_name": "embeddings",
        },
    }

    engine_under_test = ModelEngine()
    engine_under_test.current_version = "candidate"
    with patch.object(
        engine_under_test,
        "_download_s3_object",
        new_callable=AsyncMock,
        return_value=onnx_artifact,
    ):
        session, input_name, output_name = await engine_under_test._load_onnx_session(manifest)
        engine_under_test.session = session
        engine_under_test.onnx_input_name = input_name
        engine_under_test.onnx_output_name = output_name

    embeddings, version, dimension = await engine_under_test.embed_batch(["cardiology follow up"])

    assert version == "candidate"
    assert dimension == 384
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
