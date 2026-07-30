from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import onnxruntime as ort
import structlog
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts, encode_with_session, get_tokenizer

logger = structlog.get_logger()

MAX_EVALUATION_EXAMPLES = 400


async def evaluate_model(
    version: str, onnx_artifact: bytes, examples: list[dict[str, Any]]
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
    passed = improvement > settings.activation_min_improvement

    logger.info(
        "Evaluation completed",
        version=version,
        baseline=baseline_metrics,
        candidate=metrics,
        improvement=f"{improvement * 100:.2f}%",
        passed=passed,
    )
    return passed, metrics, baseline_metrics, improvement


async def evaluate_encoder_model(
    version: str, onnx_artifact: bytes, examples: list[dict[str, Any]]
) -> tuple[bool, dict[str, float], dict[str, float], float]:
    """Score a LoRA-adapted encoder against the base model on the same documents.

    Both sides embed the same raw text, so the comparison isolates the adapter. The
    projection path cannot be reused here: it scores a matrix applied to base vectors,
    while this model produces its own vectors from tokens.
    """
    # Every document is encoded twice here, once by the base model and once by the
    # candidate, so the evaluation set is the dominant cost of the whole pipeline. A few
    # hundred documents already give a stable MRR across two domains; a thousand mostly
    # buys runtime.
    if len(examples) > MAX_EVALUATION_EXAMPLES:
        examples = examples[:MAX_EVALUATION_EXAMPLES]

    logger.info("Running encoder evaluation", version=version, examples=len(examples))
    if len(examples) < 4 or len({example["source"] for example in examples}) < 2:
        empty = {"mrr": 0.0, "recall_at_5": 0.0, "mean_margin": 0.0, "quality": 0.0}
        return False, empty, empty.copy(), 0.0

    texts = [str(example["text"]) for example in examples]
    baseline_vectors = np.array(embed_texts(texts), dtype=np.float32)
    adapted_vectors = run_onnx_encoder(onnx_artifact, texts)

    baseline_metrics = score_retrieval(examples, baseline_vectors)
    metrics = score_retrieval(examples, adapted_vectors)
    improvement = (metrics["quality"] - baseline_metrics["quality"]) / max(
        baseline_metrics["quality"], 1e-6
    )
    passed = improvement > settings.activation_min_improvement

    logger.info(
        "Encoder evaluation completed",
        version=version,
        baseline=baseline_metrics,
        candidate=metrics,
        baseline_mrr=baseline_metrics["mrr"],
        candidate_mrr=metrics["mrr"],
        improvement=f"{improvement * 100:.2f}%",
        gate=f"{settings.activation_min_improvement * 100:.2f}%",
        passed=passed,
    )
    return passed, metrics, baseline_metrics, improvement


def run_onnx_encoder(onnx_artifact: bytes, texts: list[str]) -> np.ndarray:
    with NamedTemporaryFile(suffix=".onnx", delete=False) as artifact_file:
        artifact_file.write(onnx_artifact)
        artifact_path = artifact_file.name

    try:
        session = ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
        return np.asarray(encode_with_session(session, get_tokenizer(), texts), dtype=np.float32)
    finally:
        Path(artifact_path).unlink(missing_ok=True)


def run_onnx_adapter(onnx_artifact: bytes, vectors: np.ndarray) -> np.ndarray:
    with NamedTemporaryFile(suffix=".onnx", delete=False) as artifact_file:
        artifact_file.write(onnx_artifact)
        artifact_path = artifact_file.name

    try:
        session = ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
        output = session.run(["embeddings"], {"input": vectors.astype(np.float32)})[0]
        return np.asarray(output, dtype=np.float32)
    finally:
        Path(artifact_path).unlink(missing_ok=True)


def score_retrieval(examples: list[dict[str, Any]], vectors: np.ndarray) -> dict[str, float]:
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
    return np.asarray(vectors / norms, dtype=np.float32)
