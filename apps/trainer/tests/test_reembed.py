"""Activating an adapted encoder is an index migration.

Every stored vector came from the previous model, and cosine distance between vectors
produced by two different encoders is meaningless. Left alone, retrieval silently degrades
and the drift detector reads the discontinuity as drift that never happened.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest
from continuum_trainer.pipeline import reembed_corpus_with_encoder

MODEL_ID = "22222222-2222-2222-2222-222222222222"


def fake_db(documents: list[dict]):
    db = AsyncMock()
    db.query_raw = AsyncMock(return_value=documents)
    db.execute_raw = AsyncMock(return_value=1)
    return db


@pytest.mark.asyncio
async def test_every_document_is_reembedded_and_tagged(monkeypatch):
    documents = [{"id": f"doc-{index}", "text": f"document {index}"} for index in range(5)]
    monkeypatch.setattr(
        "continuum_trainer.pipeline.run_onnx_encoder",
        lambda artifact, texts: np.ones((len(texts), 384), dtype=np.float32),
    )
    db = fake_db(documents)

    count = await reembed_corpus_with_encoder(db, model_id=MODEL_ID, onnx_artifact=b"onnx")

    assert count == 5
    assert db.execute_raw.await_count == 5
    # Each row records which model produced it, so a later activation can tell them apart.
    for call in db.execute_raw.await_args_list:
        assert call.args[2] == MODEL_ID


@pytest.mark.asyncio
async def test_existing_vectors_are_replaced_not_skipped(monkeypatch):
    """ON CONFLICT DO NOTHING would leave the whole corpus on the previous model."""
    monkeypatch.setattr(
        "continuum_trainer.pipeline.run_onnx_encoder",
        lambda artifact, texts: np.ones((len(texts), 384), dtype=np.float32),
    )
    db = fake_db([{"id": "doc-1", "text": "a document"}])

    await reembed_corpus_with_encoder(db, model_id=MODEL_ID, onnx_artifact=b"onnx")

    statement = db.execute_raw.await_args_list[0].args[0]
    assert "ON CONFLICT (document_id) DO UPDATE" in statement
    assert "DO NOTHING" not in statement


@pytest.mark.asyncio
async def test_batches_are_encoded_together(monkeypatch):
    """Encoding one document per call would make a re-index unusably slow."""
    batch_sizes: list[int] = []

    def record(artifact, texts):
        batch_sizes.append(len(texts))
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr("continuum_trainer.pipeline.run_onnx_encoder", record)
    documents = [{"id": f"doc-{index}", "text": "text"} for index in range(250)]

    await reembed_corpus_with_encoder(
        fake_db(documents), model_id=MODEL_ID, onnx_artifact=b"onnx", batch_size=100
    )

    assert batch_sizes == [100, 100, 50]


@pytest.mark.asyncio
async def test_empty_corpus_is_a_no_op(monkeypatch):
    db = fake_db([])

    assert await reembed_corpus_with_encoder(db, model_id=MODEL_ID, onnx_artifact=b"x") == 0
    db.execute_raw.assert_not_awaited()
