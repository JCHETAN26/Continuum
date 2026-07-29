"""Semantic contract for the embedding model.

These assert meaning rather than shape. The previous lexical-hash embedder satisfied every
structural property (384 dims, unit norm, deterministic) while encoding no semantics at
all: its apparent domain separation came from a hardcoded word list that happened to match
the seed corpus. Shape assertions alone cannot tell the two apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from continuum_shared.embeddings import (
    EMBEDDING_DIMENSION,
    embed_text,
    embed_texts,
    mean_pool_and_normalize,
    vector_literal,
)


def cosine(left: list[float], right: list[float]) -> float:
    return float(np.array(left) @ np.array(right))


def test_paraphrases_are_closer_than_unrelated_text():
    """The property a lexical hash cannot satisfy: similarity without shared tokens."""
    mat, cat, ops = embed_texts(
        [
            "The cat sits on the mat",
            "A cat is sitting on a mat",
            "Kubernetes rolling restart of the node pool",
        ]
    )

    assert cosine(mat, cat) > 0.8
    assert cosine(mat, ops) < 0.3


def test_synonymous_sentences_match_without_word_overlap():
    physician, doctor = embed_texts(
        [
            "The physician prescribed medication for the ailment.",
            "A doctor gave the patient drugs to treat the illness.",
        ]
    )

    # Only "the" is shared, so any token-overlap scheme scores these near zero.
    assert cosine(physician, doctor) > 0.5


def test_vectors_are_unit_norm_and_correctly_sized():
    vectors = embed_texts(["first document", "second document"])

    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    for vector in vectors:
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic_across_calls():
    """Drift compares centroids over time, so identical text must embed identically."""
    first = embed_text("Administering 50mg of Losartan for hypertension management.")
    second = embed_text("Administering 50mg of Losartan for hypertension management.")

    assert first == second


def test_batching_does_not_change_vectors():
    """Padding is per batch, so masking has to make batch composition irrelevant."""
    text = "Performing an echocardiogram to assess cardiac function."
    alone = embed_text(text)
    batched = embed_texts(["a much longer document about unrelated infrastructure work", text])[1]

    assert cosine(alone, batched) == pytest.approx(1.0, abs=1e-4)


def test_rejects_a_dimension_the_model_cannot_produce():
    with pytest.raises(ValueError, match="384"):
        embed_texts(["text"], dimension=128)


def test_empty_batch_returns_no_vectors():
    assert embed_texts([]) == []


def test_mean_pooling_ignores_masked_padding_positions():
    states = np.array([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)

    pooled = mean_pool_and_normalize(states, mask)

    # Mean of the two real tokens is [2, 0]; the padded position must not contribute.
    assert pooled[0][0] == pytest.approx(1.0, abs=1e-6)
    assert pooled[0][1] == pytest.approx(0.0, abs=1e-6)


def test_vector_literal_formats_for_pgvector():
    assert vector_literal([1.0, -0.5]).startswith("[1.00000000,-0.50000000")


def test_large_batches_are_chunked_not_sent_as_one_call():
    """A single big ONNX call allocates batch x heads x seq x seq and exhausts memory.

    1000 documents at 256 tokens is roughly 3.1 GB of attention in fp32, past the
    trainer's limit. Evaluation encoded that in one call and hung for 26 minutes.
    """
    from continuum_shared import embeddings as module

    calls: list[int] = []
    real_run = module.get_runtime().session.run

    def counting_run(outputs, feeds):
        calls.append(len(feeds["input_ids"]))
        return real_run(outputs, feeds)

    runtime = module.get_runtime()
    original = runtime.session.run
    runtime.session.run = counting_run
    try:
        embed_texts([f"document {index}" for index in range(70)])
    finally:
        runtime.session.run = original

    assert len(calls) == 3
    assert max(calls) <= module.ENCODE_BATCH_SIZE
