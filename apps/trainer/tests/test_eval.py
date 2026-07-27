import numpy as np
import pytest
from continuum_trainer.eval import evaluate_model, score_retrieval
from continuum_trainer.pipeline import EmbeddingAdapter, export_adapter_to_onnx


def test_score_retrieval_prefers_same_source():
    examples = [
        {"source": "software", "vector": [1.0, 0.0]},
        {"source": "software", "vector": [0.9, 0.1]},
        {"source": "healthcare", "vector": [0.0, 1.0]},
        {"source": "healthcare", "vector": [0.1, 0.9]},
    ]
    vectors = np.array([example["vector"] for example in examples], dtype=np.float32)

    metrics = score_retrieval(examples, vectors)

    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mean_margin"] > 0


@pytest.mark.asyncio
async def test_evaluate_model_scores_onnx_adapter():
    examples = [
        {"source": "software", "vector": [1.0, 0.0]},
        {"source": "software", "vector": [0.9, 0.1]},
        {"source": "healthcare", "vector": [0.0, 1.0]},
        {"source": "healthcare", "vector": [0.1, 0.9]},
    ]
    artifact = export_adapter_to_onnx(EmbeddingAdapter(2), 2)

    passed, metrics, baseline_metrics, improvement = await evaluate_model(
        "candidate", artifact, examples
    )

    assert passed is False
    assert metrics["mrr"] == baseline_metrics["mrr"]
    assert improvement == 0
