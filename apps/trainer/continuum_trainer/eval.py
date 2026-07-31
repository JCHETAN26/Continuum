from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import onnxruntime as ort
import structlog
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts, encode_with_session, get_tokenizer
from continuum_shared.pairs import split_query_and_document
from continuum_shared.retrieval_metrics import score_ranking

logger = structlog.get_logger()

# Each pair costs four encodes: query and document, base and candidate.
MAX_EVALUATION_PAIRS = 300
MIN_EVALUATION_PAIRS = 8


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
    """Score a LoRA-adapted encoder against the base model on held-out retrieval.

    Each query must retrieve its own document out of every candidate. The previous gate
    counted any document from the same newsgroup as relevant, which half the candidates
    satisfied: it reported MRR near 0.88 for both models, left almost no headroom, and four
    training runs moved it by between -0.21% and +1.25%. Nothing could be learned that the
    measurement could see. On this task the base model scores around 0.5, so an improvement
    has somewhere to show up.

    It is also the task the adapter now trains on, so the gate and the objective agree.
    """
    splits = [
        (split, str(example["source"]))
        for example in examples
        if (split := split_query_and_document(str(example["text"]))) is not None
    ]
    pairs = [split for split, _ in splits]
    sources = [source for _, source in splits]
    if len(pairs) < MIN_EVALUATION_PAIRS:
        empty = {
            "mrr": 0.0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "quality": 0.0,
        }
        logger.warning(
            "Too few documents long enough to evaluate on", version=version, pairs=len(pairs)
        )
        return False, empty, empty.copy(), 0.0

    pairs = pairs[:MAX_EVALUATION_PAIRS]
    sources = sources[:MAX_EVALUATION_PAIRS]
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]

    logger.info("Running encoder evaluation", version=version, pairs=len(pairs))

    baseline = score_pair_retrieval(
        np.array(embed_texts(queries), dtype=np.float32),
        np.array(embed_texts(documents), dtype=np.float32),
        sources,
    )
    candidate = score_pair_retrieval(
        run_onnx_encoder(onnx_artifact, queries),
        run_onnx_encoder(onnx_artifact, documents),
        sources,
    )

    improvement = (candidate["quality"] - baseline["quality"]) / max(baseline["quality"], 1e-6)
    passed = improvement > settings.activation_min_improvement

    logger.info(
        "Encoder evaluation completed",
        version=version,
        baseline=baseline,
        candidate=candidate,
        baseline_mrr=baseline["mrr"],
        candidate_mrr=candidate["mrr"],
        baseline_ndcg=baseline["ndcg_at_10"],
        candidate_ndcg=candidate["ndcg_at_10"],
        improvement=f"{improvement * 100:.2f}%",
        gate=f"{settings.activation_min_improvement * 100:.2f}%",
        passed=passed,
    )
    return passed, candidate, baseline, improvement


def score_pair_retrieval(
    queries: np.ndarray, documents: np.ndarray, sources: list[str]
) -> dict[str, float]:
    """Rank every document for each query; exactly one is correct."""
    similarities = normalize_rows(queries) @ normalize_rows(documents).T
    metrics = score_ranking(similarities, sources)
    # Weighted toward MRR, which is the metric the gate exists to protect. NDCG is
    # reported but deliberately kept out of the promotion decision for now, so adding it
    # cannot change which models ship.
    metrics["quality"] = round((0.7 * metrics["mrr"]) + (0.3 * metrics["recall_at_5"]), 6)
    return metrics


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
