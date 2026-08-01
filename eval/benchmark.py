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
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from continuum_shared.pairs import split_query_and_document  # noqa: E402
from continuum_shared.retrieval_metrics import score_ranking  # noqa: E402
from corpus import (  # noqa: E402
    BASELINE_CATEGORIES,
    DRIFT_CATEGORIES,
    load_corpus,
    load_documents,
)

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


def evaluate_full_corpus(pool_size: int, query_count: int, seed: int) -> dict[str, float]:
    """Rank each query against every post in the corpus, not a few hundred neighbours.

    The served benchmark ranks 100 queries against 100 candidates, all drawn from two
    hardware newsgroups. A retrieval score is only as meaningful as the pool it ranks
    against, and at that size the number describes the pool as much as the model.

    This ranks against all twenty groups. Embedding runs in this process rather than
    through the serving API: the API is capped at 2.00 CPU and a round trip per batch,
    which turns twelve thousand documents into half an hour. The model is the same ONNX
    artifact the server loads, so the comparison is with the base model, not with whatever
    version happens to be promoted.
    """
    from continuum_shared.embeddings import embed_texts

    documents = load_corpus(pool_size, seed)
    splits = [
        (split, group)
        for text, group in documents
        if (split := split_query_and_document(text)) is not None
    ]
    if len(splits) <= query_count:
        raise RuntimeError(f"corpus yielded {len(splits)} pairs, need more than {query_count}")

    random.Random(seed).shuffle(splits)

    # score_ranking treats column i as the answer to query i, so the documents belonging to
    # the sampled queries have to sit at the front of the candidate list.
    queries = [pair[0] for pair, _ in splits[:query_count]]
    candidates = [pair[1] for pair, _ in splits]
    sources = [group for _, group in splits]

    print(f"embedding {len(candidates):,} candidates and {len(queries):,} queries")
    query_vectors = normalize_rows(np.asarray(embed_texts(queries), dtype=np.float32))
    candidate_vectors = normalize_rows(np.asarray(embed_texts(candidates), dtype=np.float32))

    metrics = score_ranking(query_vectors @ candidate_vectors.T, sources)
    metrics["queries"] = float(len(queries))
    metrics["candidates"] = float(len(candidates))
    return metrics


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="auto", help="auto, baseline, or a version")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="rank against every newsgroup in process instead of the served 100",
    )
    parser.add_argument("--pool-size", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.full_corpus:
        raise SystemExit(asyncio.run(run(args.model, args.output)))

    metrics = evaluate_full_corpus(args.pool_size, args.queries, EVAL_SEED)
    print(
        f"  full corpus    queries={int(metrics['queries'])} "
        f"candidates={int(metrics['candidates'])}  "
        f"MRR={metrics['mrr']:.4f}  R@1={metrics['recall_at_1']:.4f}  "
        f"R@5={metrics['recall_at_5']:.4f}  NDCG@10={metrics['ndcg_at_10']:.4f}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
