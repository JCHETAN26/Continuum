from unittest.mock import patch

import pytest
from continuum_ingest.api.main import app
from fastapi.testclient import TestClient


# We need to mock the Producer to avoid connecting to real Kafka during simple unit tests
@pytest.fixture
def client():
    with patch("continuum_ingest.api.main.Producer"):
        with TestClient(app) as test_client:
            yield test_client


def test_ingest_batch_success(client):
    response = client.post(
        "/v1/ingest/batch",
        json=[
            {
                "document_id": "doc-123",
                "text": "This is a test document",
                "source": "unit-test",
                "timestamp": "2026-07-26T21:30:00Z",
                "metadata": {"author": "AI"},
            }
        ],
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "count": 1}


def test_ingest_batch_validation_error(client):
    response = client.post(
        "/v1/ingest/batch",
        json=[
            {
                # Missing text
                "document_id": "doc-123",
                "source": "unit-test",
                "timestamp": "2026-07-26T21:30:00Z",
            }
        ],
    )
    assert response.status_code == 422
