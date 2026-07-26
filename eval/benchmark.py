import asyncio
import httpx
import time
import numpy as np

SERVER_URL = "http://localhost:8002/v1/embed"
API_KEY = "continuum-secret-key"

# Simulated evaluation set
EVAL_QUERIES = [
    "medicine for high blood pressure",
    "lung issue symptoms",
    "heart scan procedure"
]

EVAL_DOCUMENTS = [
    "Administering 50mg of Losartan for hypertension management.",
    "Patient presented with severe acute respiratory distress syndrome.",
    "Performing an echocardiogram to assess cardiac function.",
    "The Kubernetes cluster needs a rolling restart after the node pool update.",
    "Investigating a memory leak in the Node.js backend worker."
]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def get_embeddings(client, texts, model="auto"):
    headers = {
        "x-api-key": API_KEY,
        "x-model": model
    }
    response = await client.post(SERVER_URL, json={"texts": texts}, headers=headers)
    response.raise_for_status()
    return response.json()["embeddings"]

async def evaluate(model):
    print(f"\nEvaluating Model: {model}")
    async with httpx.AsyncClient() as client:
        # Get embeddings for queries and documents
        query_embeddings = await get_embeddings(client, EVAL_QUERIES, model)
        doc_embeddings = await get_embeddings(client, EVAL_DOCUMENTS, model)
        
        mrr = 0
        for i, q_emb in enumerate(query_embeddings):
            scores = [cosine_similarity(q_emb, d_emb) for d_emb in doc_embeddings]
            # The correct document for query i is document i
            # Rank is 1-indexed position of the correct document when sorted descending
            sorted_indices = np.argsort(scores)[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1
            mrr += 1.0 / rank
            
            print(f"Query: '{EVAL_QUERIES[i]}'")
            print(f"  Best Match: '{EVAL_DOCUMENTS[sorted_indices[0]]}' (Score: {scores[sorted_indices[0]]:.3f})")
            print(f"  Target Rank: {rank}")
            
        mrr /= len(EVAL_QUERIES)
        print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")

async def main():
    print("Running Continuum Benchmark")
    try:
        # Evaluate baseline model
        await evaluate("baseline")
        # Evaluate fine-tuned model (assuming it was hot-swapped and is active)
        await evaluate("auto")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    asyncio.run(main())
