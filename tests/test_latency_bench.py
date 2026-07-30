"""The percentile maths has to be right, or the reported numbers are fiction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench" / "latency.py"
spec = importlib.util.spec_from_file_location("continuum_latency", BENCH)
assert spec and spec.loader
latency = importlib.util.module_from_spec(spec)
sys.modules["continuum_latency"] = latency
spec.loader.exec_module(latency)


def test_percentiles_return_observed_values_not_interpolations():
    """statistics.quantiles interpolates, inventing a latency no request actually took."""
    samples = [float(value) for value in range(1, 101)]

    assert latency.percentile(samples, 0.50) == 50.0
    assert latency.percentile(samples, 0.95) == 95.0
    assert latency.percentile(samples, 0.99) == 99.0
    assert all(latency.percentile(samples, f) in samples for f in (0.5, 0.95, 0.99))


def test_percentile_of_a_single_sample_is_that_sample():
    assert latency.percentile([7.5], 0.99) == 7.5


def test_percentile_is_order_independent():
    ordered = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert latency.percentile(list(reversed(ordered)), 0.95) == latency.percentile(ordered, 0.95)


def test_tail_percentile_reflects_slow_requests():
    """p99 covers the slowest 1%, so it needs more than one outlier in 100 to move.

    A single slow request out of a hundred is the maximum, not the 99th percentile: 99 of
    the samples still sit at or below the fast value. Two outliers put one inside the tail.
    """
    one_outlier = [10.0] * 99 + [900.0]
    assert latency.percentile(one_outlier, 0.99) == 10.0
    assert max(one_outlier) == 900.0

    two_outliers = [10.0] * 98 + [900.0, 950.0]
    assert latency.percentile(two_outliers, 0.99) == 900.0
    assert latency.percentile(two_outliers, 0.50) == 10.0


def test_empty_sample_set_is_an_error_rather_than_a_zero():
    with pytest.raises(ValueError, match="no samples"):
        latency.percentile([], 0.5)


def test_report_renders_every_percentile():
    report = latency.LatencyReport(
        endpoint="http://localhost:8002/v1/embed",
        batch_size=32,
        requests=50,
        p50_ms=101.5,
        p95_ms=180.2,
        p99_ms=240.9,
        mean_ms=115.0,
        min_ms=90.0,
        max_ms=250.0,
    )

    rendered = report.render()
    assert "batch=32" in rendered
    assert "p50=" in rendered and "p95=" in rendered and "p99=" in rendered
    # 50 samples cannot support a p99: ceil(0.99 * 50) is rank 50, the slowest request.
    assert "(p99==max at this n)" in rendered


def test_render_omits_the_p99_caveat_once_the_sample_is_large_enough():
    report = latency.LatencyReport(
        endpoint="http://localhost:8002/v1/embed",
        batch_size=32,
        requests=200,
        p50_ms=101.5,
        p95_ms=180.2,
        p99_ms=240.9,
        mean_ms=115.0,
        min_ms=90.0,
        max_ms=250.0,
    )

    assert "(p99==max at this n)" not in report.render()
