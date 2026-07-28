import argparse
import asyncio
import random
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

# This file is both executed as a script and loaded by tests through
# spec_from_file_location, which does not put its directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import medical_documents, software_documents  # noqa: E402

DEFAULT_INGEST_URL = "http://localhost:8000/v1/ingest/batch"
DEFAULT_BATCH_SIZE = 10
DEFAULT_BASELINE_DOCUMENTS = 1_000
DEFAULT_DRIFT_DOCUMENTS = 500
DEFAULT_INTER_BATCH_DELAY_SECONDS = 0.2
DEFAULT_WINDOW_SETTLE_SECONDS = 15.0
DEFAULT_SEED = 20260727


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
            software_documents(config.baseline_documents, config.seed),
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
            medical_documents(config.drift_documents, config.seed),
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
    # Walk the corpus rather than sampling with replacement. Drawing repeatedly from a
    # small pool made every window a permutation of the same few vectors, so the drift
    # score reflected how those documents were written rather than a real distribution.
    pool = list(corpus)
    if len(pool) < total_documents:
        raise ValueError(f"corpus for {source} holds {len(pool)} documents, need {total_documents}")
    rng.shuffle(pool)
    cursor = 0
    for batch_size in batch_sizes(total_documents, config.batch_size):
        batch = pool[cursor : cursor + batch_size]
        cursor += batch_size
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
