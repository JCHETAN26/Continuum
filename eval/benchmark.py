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

from corpus import BASELINE_CATEGORIES, DRIFT_CATEGORIES, load_documents  # noqa: E402

SERVER_URL = "http://localhost:8002/v1/embed"
API_KEY = "continuum-secret-key"

QUERIES_PER_DOMAIN = 50
QUERY_WORDS = 15
MIN_DOCUMENT_WORDS = 40
EVAL_SEED = 909


@dataclass(frozen=True)
class DomainResult:
    domain: str
    queries: int
    candidates: int
    mrr: float
    recall_at_1: float
    recall_at_5: float

    def render(self) -> str:
        return (
            f"{self.domain:<14} queries={self.queries:<4} candidates={self.candidates:<5} "
            f"MRR={self.mrr:.4f}  R@1={self.recall_at_1:.4f}  R@5={self.recall_at_5:.4f}"
        )


def split_query_and_document(text: str) -> tuple[str, str] | None:
    """Opening words become the query; the remainder becomes the document.

    Splitting keeps the query out of its own document, so a hit means the model matched
    meaning rather than finding a literal copy of the query string.
    """
    words = text.split()
    if len(words) < QUERY_WORDS + MIN_DOCUMENT_WORDS:
        return None
    return " ".join(words[:QUERY_WORDS]), " ".join(words[QUERY_WORDS:])


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


def score(query_vectors: np.ndarray, doc_vectors: np.ndarray, gold: list[int]) -> tuple:
    """Rank every candidate for each query and locate the one correct document."""
    similarities = query_vectors @ doc_vectors.T
    ranks = []
    for index, correct in enumerate(gold):
        order = np.argsort(similarities[index])[::-1]
        ranks.append(int(np.where(order == correct)[0][0]) + 1)

    reciprocal = [1.0 / rank for rank in ranks]
    return (
        statistics.fmean(reciprocal),
        statistics.fmean([1.0 if rank == 1 else 0.0 for rank in ranks]),
        statistics.fmean([1.0 if rank <= 5 else 0.0 for rank in ranks]),
    )


async def evaluate(client: httpx.AsyncClient, model: str) -> list[DomainResult]:
    domains = {
        "pc_hardware": build_pairs(BASELINE_CATEGORIES, QUERIES_PER_DOMAIN),
        "mac_hardware": build_pairs(DRIFT_CATEGORIES, QUERIES_PER_DOMAIN),
    }

    # Every document from both domains is a candidate, so a query has to beat the other
    # domain's posts as well as its own neighbours.
    documents = [document for pairs in domains.values() for _, document in pairs]
    doc_vectors = await embed(client, documents, model)

    results = []
    offset = 0
    for domain, pairs in domains.items():
        queries = [query for query, _ in pairs]
        query_vectors = await embed(client, queries, model)
        gold = list(range(offset, offset + len(pairs)))
        mrr, recall_1, recall_5 = score(query_vectors, doc_vectors, gold)
        results.append(
            DomainResult(
                domain=domain,
                queries=len(pairs),
                candidates=len(documents),
                mrr=round(mrr, 6),
                recall_at_1=round(recall_1, 6),
                recall_at_5=round(recall_5, 6),
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
    print(f"  {'overall':<14} MRR={overall:.4f}")

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
