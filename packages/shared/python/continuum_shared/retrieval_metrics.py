"""Ranking metrics, shared by the promotion gate and the standalone benchmark.

Kept in one place for the same reason the pair construction is: a model gated on one
definition of quality and reported on another is being measured twice against two
different standards, and only one of them decides what ships.
"""

from __future__ import annotations

import math

import numpy as np

# Graded relevance. Exact is the document a query was cut from; related is any other
# document from the same domain. The grades matter more than their absolute values, but
# 2 and 1 keep the ideal ranking's gain interpretable.
GAIN_EXACT = 2.0
GAIN_RELATED = 1.0


def reciprocal_rank(order: np.ndarray, gold: int) -> float:
    """1/rank of the one correct document."""
    return 1.0 / (int(np.where(order == gold)[0][0]) + 1)


def ndcg_at_k(order: np.ndarray, gains: np.ndarray, k: int = 10) -> float:
    """Graded NDCG over the top k results.

    With a single relevant document this would be a monotone transform of the rank and so
    would duplicate MRR. Grading related documents above unrelated ones is what makes it
    measure something separate: whether a model that misses the exact document still keeps
    the right domain near the top. A model losing its grip on a drifted domain should lose
    that coherence, not only its exact-match precision.
    """
    ranked = gains[order[:k]]
    discounts = np.array([1.0 / math.log2(rank + 2) for rank in range(len(ranked))])
    dcg = float(np.sum(ranked * discounts))

    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def build_gain_vector(sources: list[str], gold: int) -> np.ndarray:
    """Gains for one query: its own document, then its domain, then everything else."""
    gains = np.zeros(len(sources), dtype=np.float64)
    for index, source in enumerate(sources):
        if source == sources[gold]:
            gains[index] = GAIN_RELATED
    gains[gold] = GAIN_EXACT
    return gains


def score_ranking(similarities: np.ndarray, sources: list[str], k: int = 10) -> dict[str, float]:
    """Score a query-by-document similarity matrix where query i matches document i."""
    reciprocals, hits_at_1, hits_at_5, ndcgs = [], [], [], []
    for index in range(similarities.shape[0]):
        order = np.argsort(similarities[index])[::-1]
        rank = int(np.where(order == index)[0][0]) + 1
        reciprocals.append(1.0 / rank)
        hits_at_1.append(1.0 if rank == 1 else 0.0)
        hits_at_5.append(1.0 if rank <= 5 else 0.0)
        ndcgs.append(ndcg_at_k(order, build_gain_vector(sources, index), k))

    return {
        "mrr": round(float(np.mean(reciprocals)), 6),
        "recall_at_1": round(float(np.mean(hits_at_1)), 6),
        "recall_at_5": round(float(np.mean(hits_at_5)), 6),
        "ndcg_at_10": round(float(np.mean(ndcgs)), 6),
    }
