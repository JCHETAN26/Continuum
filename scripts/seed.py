import argparse
import asyncio
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

DEFAULT_INGEST_URL = "http://localhost:8000/v1/ingest/batch"
DEFAULT_BATCH_SIZE = 10
DEFAULT_BASELINE_DOCUMENTS = 1_000
DEFAULT_DRIFT_DOCUMENTS = 500
DEFAULT_INTER_BATCH_DELAY_SECONDS = 0.2
DEFAULT_WINDOW_SETTLE_SECONDS = 15.0
DEFAULT_SEED = 20260727

SOFTWARE_TEXTS = [
    "Refactoring the authentication microservice to use JWTs.",
    "The Kubernetes cluster needs a rolling restart after the node pool update.",
    "Implementing a Redis cache layer for the REST API endpoints.",
    "Investigating a memory leak in the Node.js backend worker.",
    "The CI/CD pipeline failed during the integration tests phase.",
    "Updating the React components to use the new hooks API.",
    "Optimizing PostgreSQL query performance by adding an index.",
    "Deploying the new machine learning model to production via ONNX.",
    "Handling CORS preflight requests in the API Gateway.",
    "Writing end-to-end tests using Playwright and TypeScript.",
]

HEALTHCARE_TEXTS = [
    "Patient presented with severe acute respiratory distress syndrome.",
    "Administering 50mg of Losartan for hypertension management.",
    "The MRI results show a slight abnormality in the prefrontal cortex.",
    "Scheduling a follow-up appointment for the cardiology consultation.",
    "Blood test indicates elevated levels of low-density lipoprotein.",
    "Prescribing broad-spectrum antibiotics for the bacterial infection.",
    "The patient has a family history of Type 2 Diabetes.",
    "Performing an echocardiogram to assess cardiac function.",
    "The biopsy results came back negative for malignancy.",
    "Monitoring vital signs every 4 hours post-operation.",
]


@dataclass(frozen=True)
class SeedConfig:
    ingest_url: str = DEFAULT_INGEST_URL
    batch_size: int = DEFAULT_BATCH_SIZE
    baseline_documents: int = DEFAULT_BASELINE_DOCUMENTS
    drift_documents: int = DEFAULT_DRIFT_DOCUMENTS
    inter_batch_delay_seconds: float = DEFAULT_INTER_BATCH_DELAY_SECONDS
    window_settle_seconds: float = DEFAULT_WINDOW_SETTLE_SECONDS
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SeedResult:
    baseline_documents: int
    drift_documents: int


async def send_batch(
    client: httpx.AsyncClient,
    ingest_url: str,
    texts: Sequence[str],
    source: str,
) -> int:
    payloads = [build_payload(text, source) for text in texts]
    response = await client.post(ingest_url, json=payloads, timeout=10.0)
    response.raise_for_status()
    return len(payloads)


def build_payload(text: str, source: str) -> dict[str, object]:
    return {
        "document_id": str(uuid.uuid4()),
        "text": text,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {},
    }


async def run_seed(config: SeedConfig) -> SeedResult:
    validate_config(config)
    rng = random.Random(config.seed)

    baseline_sent = 0
    drift_sent = 0

    print("🚀 Starting Continuum Seed Script")
    print(f"Using deterministic seed {config.seed}")

    async with httpx.AsyncClient() as client:
        print("\n--- PHASE 1: Baseline Distribution ---")
        baseline_sent = await send_corpus(
            client,
            config,
            SOFTWARE_TEXTS,
            "github_issues",
            config.baseline_documents,
            rng,
        )

        print("\n⏳ Baseline established. Waiting for drift window to compute...")
        await asyncio.sleep(config.window_settle_seconds)

        print("\n--- PHASE 2: Drift Distribution (Healthcare Data) ---")
        drift_sent = await send_corpus(
            client,
            config,
            HEALTHCARE_TEXTS,
            "medical_records",
            config.drift_documents,
            rng,
        )

    print(
        "\n🎉 Seed script complete. "
        f"Ingested {baseline_sent} baseline docs and {drift_sent} drift docs."
    )
    return SeedResult(baseline_documents=baseline_sent, drift_documents=drift_sent)


async def send_corpus(
    client: httpx.AsyncClient,
    config: SeedConfig,
    corpus: Sequence[str],
    source: str,
    total_documents: int,
    rng: random.Random,
) -> int:
    sent = 0
    for batch_size in batch_sizes(total_documents, config.batch_size):
        batch = rng.choices(corpus, k=batch_size)
        sent += await send_batch(client, config.ingest_url, batch, source)
        print(f"✅ Ingested {sent}/{total_documents} documents from {source}")
        await asyncio.sleep(config.inter_batch_delay_seconds)
    return sent


def batch_sizes(total_documents: int, batch_size: int) -> list[int]:
    full_batches, remainder = divmod(total_documents, batch_size)
    sizes = [batch_size] * full_batches
    if remainder:
        sizes.append(remainder)
    return sizes


def validate_config(config: SeedConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch size must be greater than zero")
    if config.baseline_documents <= 0:
        raise ValueError("baseline document count must be greater than zero")
    if config.drift_documents <= 0:
        raise ValueError("drift document count must be greater than zero")
    if config.inter_batch_delay_seconds < 0:
        raise ValueError("inter-batch delay must be non-negative")
    if config.window_settle_seconds < 0:
        raise ValueError("window settle delay must be non-negative")


def parse_args() -> SeedConfig:
    parser = argparse.ArgumentParser(description="Seed the Continuum local drift demo.")
    parser.add_argument("--ingest-url", default=DEFAULT_INGEST_URL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--baseline-docs", type=int, default=DEFAULT_BASELINE_DOCUMENTS)
    parser.add_argument("--drift-docs", type=int, default=DEFAULT_DRIFT_DOCUMENTS)
    parser.add_argument("--delay", type=float, default=DEFAULT_INTER_BATCH_DELAY_SECONDS)
    parser.add_argument("--settle", type=float, default=DEFAULT_WINDOW_SETTLE_SECONDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    return SeedConfig(
        ingest_url=args.ingest_url,
        batch_size=args.batch_size,
        baseline_documents=args.baseline_docs,
        drift_documents=args.drift_docs,
        inter_batch_delay_seconds=args.delay,
        window_settle_seconds=args.settle,
        seed=args.seed,
    )


def main() -> None:
    asyncio.run(run_seed(parse_args()))


if __name__ == "__main__":
    main()
