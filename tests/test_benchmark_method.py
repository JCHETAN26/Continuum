"""Configuration of the standalone retrieval benchmark.

The ranking maths moved to continuum_shared.retrieval_metrics, shared with the promotion
gate so the two cannot disagree, and is tested alongside it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
spec = importlib.util.spec_from_file_location("continuum_benchmark", BENCH)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
sys.modules["continuum_benchmark"] = benchmark
spec.loader.exec_module(benchmark)


def test_one_hundred_queries_are_requested():
    """The checklist asks for fifty per domain, against the three this replaced."""
    assert benchmark.QUERIES_PER_DOMAIN == 50


def test_scoring_is_shared_with_the_promotion_gate():
    """A model gated on one definition of quality and reported on another is measured
    against two standards, and only one of them decides what ships."""
    import inspect

    from continuum_shared.retrieval_metrics import score_ranking

    assert "score_ranking" in inspect.getsource(benchmark)
    assert callable(score_ranking)


def test_results_carry_ndcg():
    result = benchmark.DomainResult(
        domain="pc_hardware",
        queries=50,
        candidates=100,
        mrr=0.6051,
        recall_at_1=0.5,
        recall_at_5=0.7,
        ndcg_at_10=0.6612,
    )

    assert "NDCG@10=0.6612" in result.render()
