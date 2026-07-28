import importlib.util
from pathlib import Path

benchmark_path = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
spec = importlib.util.spec_from_file_location("continuum_benchmark", benchmark_path)
assert spec is not None
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)


def test_score_retrieval_reports_mrr_and_ranks():
    mrr, ranks, best_indices, scores_by_query = benchmark.score_retrieval(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]],
    )

    assert mrr == 0.75
    assert ranks == [1, 2]
    assert best_indices == [0, 2]
    assert len(scores_by_query) == 2
