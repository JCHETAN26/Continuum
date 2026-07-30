"""The worker's claim decides what gets re-encoded after an activation.

Vectors from two different encoders are not comparable, so activating an adapted model
invalidates the whole corpus. That re-encoding is the worker's job: doing it inside the
training pipeline blocked the job for the length of a full re-index and could not resume.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from continuum_embedding.main import (
    CLAIM_QUERY,
    CLAIM_UNEMBEDDED_QUERY,
    UPSERT_QUERY,
    encode,
    resolve_model,
)
from continuum_shared.active_model import LoadedModel


def normalise(sql: str) -> str:
    return " ".join(sql.split())


def test_claim_selects_documents_encoded_by_another_model():
    statement = normalise(CLAIM_QUERY)

    assert "e.id IS NULL" in statement
    assert "e.model_version_id IS DISTINCT FROM $1::uuid" in statement
    # Replicas must take disjoint batches without coordinating.
    assert "FOR UPDATE OF d SKIP LOCKED" in statement


def test_claim_without_an_active_model_only_takes_unembedded_documents():
    """With nothing to compare against, re-encoding would loop forever."""
    statement = normalise(CLAIM_UNEMBEDDED_QUERY)

    assert "NOT EXISTS" in statement
    assert "model_version_id" not in statement
    assert "FOR UPDATE OF d SKIP LOCKED" in statement


def test_upsert_replaces_the_previous_vector_and_stamps_the_model():
    statement = normalise(UPSERT_QUERY)

    # DO NOTHING would leave the corpus on the old encoder forever.
    assert "ON CONFLICT (document_id) DO UPDATE" in statement
    assert "model_version_id = EXCLUDED.model_version_id" in statement


def test_encode_uses_the_adapted_encoder_when_one_is_active():
    model = LoadedModel(model_id="m1", version="v2", kind="encoder", session=object())

    with (
        patch("continuum_embedding.main.encode_with_session") as mock_encode,
        patch("continuum_embedding.main.get_tokenizer"),
        patch("continuum_embedding.main.embed_texts") as mock_base,
    ):
        mock_encode.return_value = np.ones((2, 384), dtype=np.float32)
        vectors = encode(["a", "b"], model)

    mock_encode.assert_called_once()
    mock_base.assert_not_called()
    assert len(vectors) == 2


def test_encode_falls_back_to_the_base_model_for_a_projection_artifact():
    """A projection post-processes base vectors; it cannot consume tokens."""
    model = LoadedModel(model_id="m1", version="v2", kind="projection", session=object())

    with (
        patch("continuum_embedding.main.encode_with_session") as mock_encode,
        patch("continuum_embedding.main.embed_texts") as mock_base,
    ):
        mock_base.return_value = [[0.0] * 384]
        encode(["a"], model)

    mock_base.assert_called_once()
    mock_encode.assert_not_called()


@pytest.mark.asyncio
async def test_model_artifact_is_reloaded_only_when_the_version_changes():
    """Downloading and building a session on every poll would dominate the loop."""
    current = LoadedModel(model_id="m1", version="v2", kind="encoder", session=object())

    with patch("continuum_embedding.main.load_active_model", new_callable=AsyncMock) as loader:
        loader.return_value = LoadedModel(
            model_id="m1", version="v2", kind="encoder", session=object()
        )
        resolved = await resolve_model(SimpleNamespace(), current)

    assert resolved is current


@pytest.mark.asyncio
async def test_a_broken_artifact_does_not_stop_the_worker():
    """New documents still need embedding even if the active artifact cannot be read."""
    current = LoadedModel(model_id="m1", version="v2", kind="encoder", session=object())

    with patch("continuum_embedding.main.load_active_model", new_callable=AsyncMock) as loader:
        loader.side_effect = ValueError("checksum mismatch")
        resolved = await resolve_model(SimpleNamespace(), current)

    assert resolved is current
