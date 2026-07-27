import asyncio
import random
import uuid
from datetime import UTC, datetime

import httpx

INGEST_URL = "http://localhost:8000/v1/ingest/batch"
BATCH_SIZE = 10
BASELINE_DOCUMENTS = 1_000
DRIFT_DOCUMENTS = 500
INTER_BATCH_DELAY_SECONDS = 0.2
WINDOW_SETTLE_SECONDS = 15

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


async def send_batch(client: httpx.AsyncClient, texts: list[str], source: str):
    payloads = []
    for text in texts:
        payloads.append(
            {
                "document_id": str(uuid.uuid4()),
                "text": text,
                "source": source,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": {},
            }
        )

    try:
        response = await client.post(INGEST_URL, json=payloads, timeout=10.0)
        response.raise_for_status()
        print(f"✅ Ingested {len(payloads)} documents from {source}")
    except Exception as e:
        print(f"❌ Failed to ingest batch: {e}")


async def main():
    print("🚀 Starting Continuum Seed Script")

    async with httpx.AsyncClient() as client:
        # Phase 1: Establish baseline (Software Engineering)
        print("\n--- PHASE 1: Baseline Distribution ---")
        for _ in range(BASELINE_DOCUMENTS // BATCH_SIZE):
            batch = random.choices(SOFTWARE_TEXTS, k=BATCH_SIZE)
            await send_batch(client, batch, "github_issues")
            await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)

        print("\n⏳ Baseline established. Waiting for drift window to compute...")
        await asyncio.sleep(WINDOW_SETTLE_SECONDS)

        # Phase 2: Induce drift (Healthcare)
        print("\n--- PHASE 2: Drift Distribution (Healthcare Data) ---")
        for _ in range(DRIFT_DOCUMENTS // BATCH_SIZE):
            batch = random.choices(HEALTHCARE_TEXTS, k=BATCH_SIZE)
            await send_batch(client, batch, "medical_records")
            await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)

        print("\n🎉 Seed script complete. Check the Continuum Dashboard for drift alerts!")


if __name__ == "__main__":
    asyncio.run(main())
