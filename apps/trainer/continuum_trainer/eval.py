from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnxruntime as ort
import structlog

logger = structlog.get_logger()


async def evaluate_model(
    version: str, onnx_artifact: bytes, examples: list[dict]
) -> tuple[bool, dict[str, float], dict[str, float], float]:
    """
    Evaluate the trained adapter against baseline embeddings with retrieval metrics.

    Queries and candidate documents come from the same corpus examples. A relevant match
    is any other document from the same source/domain.
    """

    logger.info("Running retrieval evaluation", version=version, examples=len(examples))
    if len(examples) < 4 or len({example["source"] for example in examples}) < 2:
        baseline_metrics = {"mrr": 0.0, "recall_at_5": 0.0, "mean_margin": 0.0, "quality": 0.0}
        metrics = baseline_metrics.copy()
        return False, metrics, baseline_metrics, 0.0

    baseline_vectors = np.array([example["vector"] for example in examples], dtype=np.float32)
    adapted_vectors = run_onnx_adapter(onnx_artifact, baseline_vectors)

    baseline_metrics = score_retrieval(examples, baseline_vectors)
    metrics = score_retrieval(examples, adapted_vectors)
    improvement = (metrics["quality"] - baseline_metrics["quality"]) / max(
        baseline_metrics["quality"], 1e-6
    )
    passed = improvement > 0.02

    logger.info(
        "Evaluation completed",
        version=version,
        baseline=baseline_metrics,
        candidate=metrics,
        improvement=f"{improvement * 100:.2f}%",
        passed=passed,
    )
    return passed, metrics, baseline_metrics, improvement


def run_onnx_adapter(onnx_artifact: bytes, vectors: np.ndarray) -> np.ndarray:
    with NamedTemporaryFile(suffix=".onnx", delete=False) as artifact_file:
        artifact_file.write(onnx_artifact)
        artifact_path = artifact_file.name

    try:
        session = ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
        output = session.run(["embeddings"], {"input": vectors.astype(np.float32)})[0]
        return output.astype(np.float32)
    finally:
        Path(artifact_path).unlink(missing_ok=True)


def score_retrieval(examples: list[dict], vectors: np.ndarray) -> dict[str, float]:
    normalized = normalize_rows(vectors)
    scores = normalized @ normalized.T

    reciprocal_ranks = []
    recalls_at_5 = []
    margins = []
    for query_index, query_example in enumerate(examples):
        same_source = [
            index
            for index, candidate in enumerate(examples)
            if index != query_index and candidate["source"] == query_example["source"]
        ]
        different_source = [
            index
            for index, candidate in enumerate(examples)
            if candidate["source"] != query_example["source"]
        ]
        if not same_source or not different_source:
            continue

        ranked = [
            index
            for index in np.argsort(scores[query_index])[::-1].tolist()
            if index != query_index
        ]
        first_relevant_rank = min(ranked.index(index) + 1 for index in same_source)
        reciprocal_ranks.append(1.0 / first_relevant_rank)
        recalls_at_5.append(float(any(index in same_source for index in ranked[:5])))
        positive_score = scores[query_index, same_source].mean()
        negative_score = scores[query_index, different_source].mean()
        margins.append(float(positive_score - negative_score))

    if not reciprocal_ranks:
        return {"mrr": 0.0, "recall_at_5": 0.0, "mean_margin": 0.0, "quality": 0.0}

    mrr = float(np.mean(reciprocal_ranks))
    recall_at_5 = float(np.mean(recalls_at_5))
    mean_margin = float(np.mean(margins))
    quality = (0.65 * mrr) + (0.2 * recall_at_5) + (0.15 * ((mean_margin + 1.0) / 2.0))
    return {
        "mrr": round(mrr, 6),
        "recall_at_5": round(recall_at_5, 6),
        "mean_margin": round(mean_margin, 6),
        "quality": round(quality, 6),
    }


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
