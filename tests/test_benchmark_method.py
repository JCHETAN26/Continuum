"""The benchmark's scoring has to be right or its numbers are decoration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
spec = importlib.util.spec_from_file_location("continuum_benchmark", BENCH)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
sys.modules["continuum_benchmark"] = benchmark
spec.loader.exec_module(benchmark)


def test_perfect_retrieval_scores_one():
    vectors = np.eye(4, dtype=np.float32)
    mrr, recall_1, recall_5 = benchmark.score(vectors, vectors, [0, 1, 2, 3])

    assert mrr == 1.0
    assert recall_1 == 1.0
    assert recall_5 == 1.0


def test_second_place_scores_one_half():
    """MRR is the reciprocal of the rank of the one correct document."""
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    # The wrong document is a closer match than the right one.
    documents = np.array([[0.7, 0.7], [0.99, 0.14]], dtype=np.float32)

    mrr, recall_1, recall_5 = benchmark.score(queries, documents, [0])

    assert mrr == 0.5
    assert recall_1 == 0.0
    assert recall_5 == 1.0


def test_a_result_outside_the_top_five_scores_zero_recall():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    # Strictly decreasing similarity, so the gold document's rank is unambiguous. Equal
    # similarities would leave the ordering to argsort's tie handling.
    documents = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.2],
            [0.96, 0.28],
            [0.94, 0.34],
            [0.92, 0.39],
            [0.90, 0.44],
            [0.20, 0.98],
        ],
        dtype=np.float32,
    )
    mrr, recall_1, recall_5 = benchmark.score(queries, documents, [6])

    assert recall_1 == 0.0
    assert recall_5 == 0.0
    assert mrr < 0.2


def test_one_hundred_queries_are_requested():
    """The checklist asks for fifty per domain, against the previous three in total."""
    assert benchmark.QUERIES_PER_DOMAIN == 50
