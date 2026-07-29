"""Activating an adapted encoder is an index migration.

Every stored vector came from the previous model, and cosine distance between vectors
produced by two different encoders is meaningless. Left alone, retrieval silently degrades
and the drift detector reads the discontinuity as drift that never happened.
"""

from __future__ import annotations

from types import SimpleNamespace
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


@pytest.mark.asyncio
async def test_evaluation_set_is_drawn_per_source():
    """A single-domain evaluation set cannot measure retrieval at all.

    Relevance is same-source, so a set containing one source has no negatives and every
    metric returns 0.0. Ordering purely by embedding time allowed exactly that: drifted
    documents arrive last and embed last, so the most recent rows can be one source
    entirely, and one run in three reported baseline and candidate MRR both 0.0000.
    """
    from continuum_trainer.pipeline import load_training_examples

    captured: list = []

    async def query_raw(sql: str, *args):
        captured.append((sql, args))
        return [
            {"source": "pc_hardware", "text": "isa irq", "vec_str": "[0.1,0.2]"},
            {"source": "mac_hardware", "text": "quadra vram", "vec_str": "[0.3,0.4]"},
            {"source": "pc_hardware", "text": "bios beep", "vec_str": "[0.5,0.6]"},
            {"source": "mac_hardware", "text": "scsi chain", "vec_str": "[0.7,0.8]"},
        ]

    examples = await load_training_examples(SimpleNamespace(query_raw=query_raw), per_source=250)

    statement = " ".join(captured[0][0].split())
    assert "PARTITION BY d.source" in statement
    assert captured[0][1] == (250,)
    assert {example["source"] for example in examples} == {"pc_hardware", "mac_hardware"}


def test_demo_backend_reports_no_loss_trajectory():
    """The demo adapter solves in closed form, so it has no optimisation steps.

    It used to emit a descending curve by walking a hardcoded margin schedule
    ([0.8, 0.6, 0.4, 0.25, 0.15]) against a model that never changed between "steps". The
    curve fell because the constants fell, and the dashboard rendered it as training
    telemetry for a backend that trains nothing.
    """
    import inspect

    from continuum_trainer import pipeline

    source = inspect.getsource(pipeline)
    assert "estimate_loss_history" not in source
    # The margin schedule that manufactured the curve must not come back either.
    assert "0.25, 0.15" not in source
