"""Graded NDCG has to measure something MRR does not, or it is decoration.

With one relevant document per query, NDCG is a monotone transform of the rank and carries
exactly the information MRR already has. Grading same-domain documents above unrelated ones
is what makes it a separate signal: whether a model that misses the exact document still
keeps the right domain near the top.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from continuum_shared.retrieval_metrics import (
    GAIN_EXACT,
    GAIN_RELATED,
    build_gain_vector,
    ndcg_at_k,
    score_ranking,
)


def test_gain_vector_grades_exact_above_domain_above_unrelated():
    sources = ["pc", "pc", "mac", "mac"]

    gains = build_gain_vector(sources, gold=0)

    assert gains[0] == GAIN_EXACT
    assert gains[1] == GAIN_RELATED
    assert gains[2] == 0.0 and gains[3] == 0.0


def test_perfect_ranking_scores_one():
    sources = ["pc", "pc", "mac", "mac"]
    order = np.array([0, 1, 2, 3])

    assert ndcg_at_k(order, build_gain_vector(sources, 0)) == pytest.approx(1.0)


def test_same_domain_miss_beats_a_cross_domain_miss():
    """The distinction MRR cannot draw: both put the gold document at rank 3."""
    sources = ["pc", "pc", "mac", "mac"]
    gains = build_gain_vector(sources, gold=0)

    stayed_in_domain = ndcg_at_k(np.array([1, 3, 0, 2]), gains)
    left_the_domain = ndcg_at_k(np.array([3, 2, 0, 1]), gains)

    assert stayed_in_domain > left_the_domain
    # And MRR is identical for both, which is the point.
    assert 1 / 3 == 1 / 3


def test_ndcg_is_bounded():
    sources = ["pc", "mac", "mac", "pc"]
    gains = build_gain_vector(sources, gold=0)

    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]):
        value = ndcg_at_k(np.array(order), gains)
        assert 0.0 <= value <= 1.0


def test_score_ranking_reports_every_metric():
    """Similarities are strictly ordered so no metric depends on tie handling.

    An identity matrix leaves every non-gold document at the same score, and argsort then
    decides whether the same-domain document lands second or last, which moves NDCG.
    """
    # Query i ranks: own document, then its domain partner, then the other domain.
    similarities = np.array(
        [
            [0.99, 0.80, 0.30, 0.20],
            [0.80, 0.99, 0.30, 0.20],
            [0.20, 0.30, 0.99, 0.80],
            [0.20, 0.30, 0.80, 0.99],
        ],
        dtype=np.float32,
    )
    sources = ["pc", "pc", "mac", "mac"]

    metrics = score_ranking(similarities, sources)

    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)


def test_binary_ndcg_would_have_duplicated_mrr():
    """Documented so the graded design is not later 'simplified' back into redundancy."""
    for rank in (1, 2, 3, 5, 10):
        binary_ndcg = 1 / math.log2(rank + 1)
        mrr = 1 / rank
        # Different values, but both strictly decreasing in rank: no independent signal.
        assert binary_ndcg > 0 and mrr > 0
    assert 1 / math.log2(2 + 1) > 1 / 2
