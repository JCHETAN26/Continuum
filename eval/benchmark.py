"""Retrieval benchmark over held-out documents.

Independent of the evaluation the trainer runs on itself. The trainer scores a candidate
with the same code that decides whether to promote it, which is fine for a gate but is not
evidence anyone outside the pipeline should take at face value.

Method. Each query is the opening sentence of a held-out post; the document is the rest of
that post, and it is the only relevant result among every candidate. So the task is: given
an opening, find the post it came from, among hundreds of posts about the same subject.

That matters. Scoring relevance as "any document from the same newsgroup" makes the task
almost free, because half the candidates qualify — the trainer's own gate scores that way
and reports MRR around 0.88 as a result. One relevant document out of several hundred is a
task a retrieval model can actually be wrong about.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from continuum_shared.pairs import split_query_and_document  # noqa: E402
from continuum_shared.retrieval_metrics import score_ranking  # noqa: E402
from corpus import BASELINE_CATEGORIES, DRIFT_CATEGORIES, load_documents  # noqa: E402

SERVER_URL = "http://localhost:8002/v1/embed"
API_KEY = "continuum-secret-key"

QUERIES_PER_DOMAIN = 50
EVAL_SEED = 909


@dataclass(frozen=True)
class DomainResult:
    domain: str
    queries: int
    candidates: int
    mrr: float
    recall_at_1: float
    recall_at_5: float
    ndcg_at_10: float

    def render(self) -> str:
        return (
            f"{self.domain:<14} queries={self.queries:<4} candidates={self.candidates:<5} "
            f"MRR={self.mrr:.4f}  R@1={self.recall_at_1:.4f}  R@5={self.recall_at_5:.4f}  "
            f"NDCG@10={self.ndcg_at_10:.4f}"
        )


def build_pairs(categories: frozenset[str], limit: int) -> list[tuple[str, str]]:
    pairs = []
    for text in load_documents(categories, limit * 3, EVAL_SEED):
        split = split_query_and_document(text)
        if split:
            pairs.append(split)
        if len(pairs) == limit:
            break
    return pairs


async def embed(client: httpx.AsyncClient, texts: list[str], model: str) -> np.ndarray:
    vectors: list[list[float]] = []
    # The serving container is CPU bound, so a single large request is slower than several
    # moderate ones and risks the request timeout.
    for start in range(0, len(texts), 32):
        response = await client.post(
            SERVER_URL,
            json={"texts": texts[start : start + 32]},
            headers={"x-api-key": API_KEY, "x-model": model},
            timeout=120.0,
        )
        response.raise_for_status()
        vectors.extend(response.json()["embeddings"])

    array = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


async def evaluate(client: httpx.AsyncClient, model: str) -> list[DomainResult]:
    domains = {
        "pc_hardware": build_pairs(BASELINE_CATEGORIES, QUERIES_PER_DOMAIN),
        "mac_hardware": build_pairs(DRIFT_CATEGORIES, QUERIES_PER_DOMAIN),
    }

    # Every document from both domains is a candidate, so a query has to beat the other
    # domain's posts as well as its own neighbours.
    documents = [document for pairs in domains.values() for _, document in pairs]
    doc_vectors = await embed(client, documents, model)

    # Every candidate is labelled with the domain it came from, so NDCG can grade a
    # same-domain miss above a cross-domain one.
    sources = [domain for domain, pairs in domains.items() for _ in pairs]

    results = []
    offset = 0
    for domain, pairs in domains.items():
        queries = [query for query, _ in pairs]
        query_vectors = await embed(client, queries, model)
        # score_ranking expects query i to match document i, so the slice of candidates is
        # rotated to put this domain's documents first.
        order = list(range(offset, offset + len(pairs))) + [
            index for index in range(len(documents)) if not offset <= index < offset + len(pairs)
        ]
        similarities = query_vectors @ doc_vectors[order].T
        metrics = score_ranking(similarities, [sources[index] for index in order])
        results.append(
            DomainResult(
                domain=domain,
                queries=len(pairs),
                candidates=len(documents),
                mrr=metrics["mrr"],
                recall_at_1=metrics["recall_at_1"],
                recall_at_5=metrics["recall_at_5"],
                ndcg_at_10=metrics["ndcg_at_10"],
            )
        )
        offset += len(pairs)

    return results


async def run(model: str, output: Path | None) -> int:
    async with httpx.AsyncClient() as client:
        results = await evaluate(client, model)

    print(f"model={model}")
    for result in results:
        print("  " + result.render())

    overall = statistics.fmean([result.mrr for result in results])
    overall_ndcg = statistics.fmean([result.ndcg_at_10 for result in results])
    print(f"  {'overall':<14} MRR={overall:.4f}  NDCG@10={overall_ndcg:.4f}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "overall_mrr": round(overall, 6),
            "domains": [asdict(result) for result in results],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="auto", help="auto, baseline, or a version")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args.model, args.output)))


if __name__ == "__main__":
    main()
