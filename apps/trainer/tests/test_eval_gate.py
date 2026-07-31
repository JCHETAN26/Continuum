"""The activation gate must be a real quality bar, not a rounding error.

It sat at a hardcoded 0.005 — half a percent. An adapter that merely reshuffles near-tied
neighbours clears that without retrieving anything better, and the pipeline then promotes
it to ACTIVE and serves it.
"""

from __future__ import annotations

import numpy as np
import pytest
from continuum_shared.config import settings
from continuum_trainer.eval import evaluate_model, score_retrieval


def test_default_gate_is_a_meaningful_quality_bar():
    assert settings.activation_min_improvement == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_marginal_improvement_does_not_activate(monkeypatch):
    """A candidate a hair better than baseline must not be promoted."""
    monkeypatch.setattr(settings, "activation_min_improvement", 0.10)
    examples = [
        {"source": "software", "vector": [1.0, 0.0, 0.0]},
        {"source": "software", "vector": [0.99, 0.14, 0.0]},
        {"source": "medical", "vector": [0.0, 1.0, 0.0]},
        {"source": "medical", "vector": [0.0, 0.99, 0.14]},
    ]
    # Return the baseline vectors untouched: zero improvement.
    monkeypatch.setattr(
        "continuum_trainer.eval.run_onnx_adapter",
        lambda artifact, vectors: np.array(vectors, dtype=np.float32),
    )

    passed, _, _, improvement = await evaluate_model("v1", b"", examples)

    assert improvement == pytest.approx(0.0, abs=1e-9)
    assert passed is False


@pytest.mark.asyncio
async def test_gate_threshold_is_configurable(monkeypatch):
    """The same candidate must pass a loose bar and fail a strict one."""
    # Baseline interleaves the domains, so retrieval starts poor.
    examples = [
        {"source": "software", "vector": [1.0, 0.0]},
        {"source": "medical", "vector": [0.92, 0.39]},
        {"source": "software", "vector": [0.71, 0.71]},
        {"source": "medical", "vector": [0.0, 1.0]},
    ]
    # The "adapter" separates the domains cleanly: a large, unambiguous improvement.
    separated = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        "continuum_trainer.eval.run_onnx_adapter", lambda artifact, vectors: separated
    )

    monkeypatch.setattr(settings, "activation_min_improvement", 0.01)
    permissive, _, _, improvement = await evaluate_model("v1", b"", examples)

    monkeypatch.setattr(settings, "activation_min_improvement", 0.99)
    strict, _, _, _ = await evaluate_model("v1", b"", examples)

    assert improvement > 0.01
    assert permissive is True
    assert strict is False


def test_score_retrieval_reports_mrr():
    examples = [
        {"source": "software", "vector": [1.0, 0.0]},
        {"source": "software", "vector": [0.99, 0.1]},
        {"source": "medical", "vector": [0.0, 1.0]},
        {"source": "medical", "vector": [0.1, 0.99]},
    ]

    metrics = score_retrieval(examples, np.array([e["vector"] for e in examples], dtype=np.float32))

    assert metrics["mrr"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_evaluation_set_is_capped(monkeypatch):
    """Each document is encoded twice, so the eval set dominates pipeline runtime.

    Passing a thousand documents through two encoders left the run evaluating for 26
    minutes and timing out after training had already succeeded.
    """
    from continuum_trainer import eval as eval_module

    seen: list[int] = []

    def record(texts):
        seen.append(len(texts))
        # Distinct vectors so the retrieval scorer has something to rank.
        return np.eye(len(texts), 8, dtype=np.float32)

    monkeypatch.setattr("continuum_trainer.eval.embed_texts", record)
    monkeypatch.setattr(
        "continuum_trainer.eval.run_onnx_encoder",
        lambda artifact, texts: np.eye(len(texts), 8, dtype=np.float32),
    )

    body = " ".join(f"word{n}" for n in range(60))
    examples = [
        {
            "source": "pc_hardware" if index % 2 else "mac_hardware",
            "text": f"document {index} opening line about hardware {body}",
        }
        for index in range(1000)
    ]
    await eval_module.evaluate_encoder_model("v1", b"onnx", examples)

    # Queries and documents are embedded separately, so each capped pair costs two calls.
    assert seen == [eval_module.MAX_EVALUATION_PAIRS, eval_module.MAX_EVALUATION_PAIRS]
