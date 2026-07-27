import asyncio

import httpx
import numpy as np

SERVER_URL = "http://localhost:8002/v1/embed"
API_KEY = "continuum-secret-key"

EVAL_QUERIES = [
    "medicine for high blood pressure",
    "lung issue symptoms",
    "heart scan procedure",
]

EVAL_DOCUMENTS = [
    "Administering 50mg of Losartan for hypertension management.",
    "Patient presented with severe acute respiratory distress syndrome.",
    "Performing an echocardiogram to assess cardiac function.",
    "The Kubernetes cluster needs a rolling restart after the node pool update.",
    "Investigating a memory leak in the Node.js backend worker.",
]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    return float(np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array)))


async def get_embeddings(
    client: httpx.AsyncClient, texts: list[str], model: str
) -> list[list[float]]:
    headers = {"x-api-key": API_KEY, "x-model": model}
    response = await client.post(SERVER_URL, json={"texts": texts}, headers=headers, timeout=20.0)
    response.raise_for_status()
    body = response.json()
    print(f"  served_by={body['model_version_used']} dimension={body['dimension']}")
    return body["embeddings"]


async def evaluate(client: httpx.AsyncClient, model: str) -> float:
    print(f"\nEvaluating model: {model}")
    query_embeddings = await get_embeddings(client, EVAL_QUERIES, model)
    doc_embeddings = await get_embeddings(client, EVAL_DOCUMENTS, model)

    mrr, ranks, best_indices, scores_by_query = score_retrieval(query_embeddings, doc_embeddings)

    for query_index, rank in enumerate(ranks):
        scores = scores_by_query[query_index]
        best_index = best_indices[query_index]
        print(f"  query='{EVAL_QUERIES[query_index]}'")
        print(f"    best='{EVAL_DOCUMENTS[best_index]}' score={scores[best_index]:.3f}")
        print(f"    target_rank={rank}")

    print(f"  MRR={mrr:.3f}")
    return mrr


def score_retrieval(
    query_embeddings: list[list[float]], doc_embeddings: list[list[float]]
) -> tuple[float, list[int], list[int], list[list[float]]]:
    mrr = 0.0
    ranks = []
    best_indices = []
    scores_by_query = []
    for query_index, query_embedding in enumerate(query_embeddings):
        scores = [
            cosine_similarity(query_embedding, document_embedding)
            for document_embedding in doc_embeddings
        ]
        sorted_indices = np.argsort(scores)[::-1]
        rank = int(np.where(sorted_indices == query_index)[0][0] + 1)
        mrr += 1.0 / rank
        ranks.append(rank)
        best_indices.append(int(sorted_indices[0]))
        scores_by_query.append(scores)

    mrr /= len(query_embeddings)
    return mrr, ranks, best_indices, scores_by_query


async def main():
    print("Running Continuum Benchmark")
    async with httpx.AsyncClient() as client:
        baseline_mrr = await evaluate(client, "baseline")
        active_mrr = await evaluate(client, "auto")

    delta = active_mrr - baseline_mrr
    relative = delta / max(baseline_mrr, 1e-9)
    print("\nSummary")
    print(f"  baseline_mrr={baseline_mrr:.3f}")
    print(f"  active_mrr={active_mrr:.3f}")
    print(f"  delta={delta:+.3f} relative={relative:+.1%}")


if __name__ == "__main__":
    asyncio.run(main())
