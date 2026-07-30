"""Measure serving latency percentiles against the running embedding API.

The product spec asks for p50/p95/p99 per endpoint and sets a target of p99 under 50 ms at
batch 32 on CPU. This measures rather than asserts: the numbers it prints are whatever the
service does, and a target that turns out to be unreachable is a finding worth recording
rather than a threshold to quietly relax.

Payloads are drawn from the same 20 Newsgroups corpus the demo ingests, so token counts
reflect real documents. Latency on a transformer scales with sequence length, so measuring
with short synthetic strings would understate it by a wide margin.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus import load_baseline_documents  # noqa: E402

DEFAULT_URL = "http://localhost:8002/v1/embed"
DEFAULT_API_KEY = "continuum-secret-key"


@dataclass(frozen=True)
class LatencyReport:
    endpoint: str
    batch_size: int
    requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def render(self) -> str:
        # At small sample counts the nearest-rank p99 collapses onto the slowest request:
        # ceil(0.99 * 50) is 50. Flagging it keeps the output from presenting the maximum
        # as a percentile it cannot support.
        p99_note = "" if self.requests >= 100 else "  (p99==max at this n)"
        return (
            f"batch={self.batch_size:<3} n={self.requests:<4} "
            f"p50={self.p50_ms:8.2f}ms  p95={self.p95_ms:8.2f}ms  "
            f"p99={self.p99_ms:8.2f}ms  max={self.max_ms:8.2f}ms{p99_note}"
        )


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile.

    statistics.quantiles interpolates between neighbours, which invents a latency no
    request actually took. At these sample sizes the difference is visible.
    """
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    # Nearest-rank: ceil(fraction * N). Using round(x + 0.5) instead trips Python's
    # banker's rounding, which turns a p95 of rank 95.5 into 96 and reports a different
    # request's latency than the definition calls for.
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def measure(
    client: httpx.Client,
    url: str,
    api_key: str,
    texts: list[str],
    batch_size: int,
    requests: int,
    warmup: int,
) -> LatencyReport:
    payload_texts = [texts[index % len(texts)] for index in range(batch_size)]
    body = {"texts": payload_texts}
    headers = {"x-api-key": api_key, "x-model": "auto"}

    # The first calls pay for ONNX session warm-up and allocator growth, which is real but
    # one-off, and including it would put a startup cost into the tail percentiles.
    for _ in range(warmup):
        client.post(url, json=body, headers=headers, timeout=60.0).raise_for_status()

    samples: list[float] = []
    for _ in range(requests):
        started = time.perf_counter()
        response = client.post(url, json=body, headers=headers, timeout=60.0)
        response.raise_for_status()
        samples.append((time.perf_counter() - started) * 1000.0)

    return LatencyReport(
        endpoint=url,
        batch_size=batch_size,
        requests=requests,
        p50_ms=round(percentile(samples, 0.50), 3),
        p95_ms=round(percentile(samples, 0.95), 3),
        p99_ms=round(percentile(samples, 0.99), 3),
        mean_ms=round(statistics.fmean(samples), 3),
        min_ms=round(min(samples), 3),
        max_ms=round(max(samples), 3),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texts = load_baseline_documents(64, seed=20260727)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]

    reports: list[LatencyReport] = []
    with httpx.Client() as client:
        for batch_size in batch_sizes:
            report = measure(
                client,
                args.url,
                args.api_key,
                texts,
                batch_size,
                args.requests,
                args.warmup,
            )
            reports.append(report)
            print(report.render(), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(report) for report in reports], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
