import argparse
import asyncio
import time
from dataclasses import dataclass

import httpx

DRIFT_API = "http://localhost:8001"
TRAINER_API = "http://localhost:8003"
SERVER_API = "http://localhost:8002"
API_KEY = "continuum-secret-key"


@dataclass(frozen=True)
class DemoCheckResult:
    name: str
    passed: bool
    detail: str


async def wait_for(
    client: httpx.AsyncClient,
    name: str,
    timeout_seconds: float,
    interval_seconds: float,
    check,
) -> DemoCheckResult:
    deadline = time.monotonic() + timeout_seconds
    last_detail = "not checked yet"

    while time.monotonic() < deadline:
        result = await check(client)
        if result.passed:
            return result
        last_detail = result.detail
        await asyncio.sleep(interval_seconds)

    return DemoCheckResult(name, False, f"timed out after {timeout_seconds:.0f}s: {last_detail}")


async def check_document_flow(client: httpx.AsyncClient) -> DemoCheckResult:
    response = await client.get(f"{DRIFT_API}/v1/drift/summary")
    response.raise_for_status()
    summary = response.json()
    documents = int(summary["documentCount"])
    embeddings = int(summary["embeddingCount"])
    passed = documents >= 1500 and embeddings >= 1500
    return DemoCheckResult(
        "document flow",
        passed,
        f"{documents} documents, {embeddings} embeddings",
    )


async def check_drift_spike(client: httpx.AsyncClient) -> DemoCheckResult:
    response = await client.get(f"{DRIFT_API}/v1/drift/summary")
    response.raise_for_status()
    summary = response.json()
    score = float(summary["latestDriftScore"])
    threshold = float(summary["threshold"])
    breached = bool(summary["breached"])
    return DemoCheckResult(
        "drift spike",
        breached and score > threshold,
        f"score={score:.3f}, threshold={threshold:.3f}, breached={breached}",
    )


async def check_training_job(client: httpx.AsyncClient) -> DemoCheckResult:
    response = await client.get(f"{TRAINER_API}/v1/training/jobs")
    response.raise_for_status()
    jobs = response.json()
    completed = [
        job for job in jobs if job["status"] in {"SUCCEEDED", "FAILED"} and job["sampleCount"]
    ]
    if completed:
        latest = completed[0]
        return DemoCheckResult(
            "training job",
            latest["status"] == "SUCCEEDED",
            f"job={latest['id']}, status={latest['status']}, samples={latest['sampleCount']}",
        )

    return DemoCheckResult("training job", False, f"{len(jobs)} jobs, none complete yet")


async def check_model_registry(client: httpx.AsyncClient) -> DemoCheckResult:
    response = await client.get(f"{TRAINER_API}/v1/models")
    response.raise_for_status()
    models = response.json()
    active = next((model for model in models if model["status"] == "ACTIVE"), None)
    passed = next((model for model in models if model["status"] in {"ACTIVE", "PASSED"}), None)
    model = active or passed

    if not model:
        return DemoCheckResult("model registry", False, f"{len(models)} models, none active/passed")

    improvement = model.get("improvementPct")
    metric = "n/a" if improvement is None else f"{float(improvement):+.1%}"
    return DemoCheckResult(
        "model registry",
        True,
        f"version={model['version']}, status={model['status']}, improvement={metric}",
    )


async def check_retrieval_quality_improvement(client: httpx.AsyncClient) -> DemoCheckResult:
    baseline_version = await get_served_model_version(client, "baseline")
    active_version = await get_served_model_version(client, "auto")
    if active_version == baseline_version or active_version == "baseline":
        return DemoCheckResult(
            "retrieval quality improvement",
            False,
            f"baseline={baseline_version}, active={active_version}",
        )

    response = await client.get(f"{TRAINER_API}/v1/models/{active_version}")
    response.raise_for_status()
    active_model = response.json()
    improvement = active_model.get("improvementPct")
    passed = improvement is not None and float(improvement) > 0
    return DemoCheckResult(
        "retrieval quality improvement",
        passed,
        f"served_by={active_version}, improvement={float(improvement or 0):+.1%}",
    )


async def get_served_model_version(client: httpx.AsyncClient, model: str) -> str:
    response = await client.post(
        f"{SERVER_API}/v1/embed",
        json={"texts": ["medicine for high blood pressure"]},
        headers={"x-api-key": API_KEY, "x-model": model},
    )
    response.raise_for_status()
    return str(response.json()["model_version_used"])


async def run(timeout_seconds: float, interval_seconds: float) -> int:
    checks = [
        check_document_flow,
        check_drift_spike,
        check_training_job,
        check_model_registry,
        check_retrieval_quality_improvement,
    ]

    async with httpx.AsyncClient(timeout=20.0) as client:
        for check in checks:
            result = await wait_for(
                client,
                check.__name__,
                timeout_seconds,
                interval_seconds,
                check,
            )
            prefix = "PASS" if result.passed else "FAIL"
            print(f"{prefix} {result.name}: {result.detail}")
            if not result.passed:
                return 1

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the Continuum local demo narrative.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(asyncio.run(run(args.timeout, args.interval)))


if __name__ == "__main__":
    main()
