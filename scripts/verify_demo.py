import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DRIFT_API = "http://localhost:8001"
TRAINER_API = "http://localhost:8003"
# Mirrors settings.activation_min_improvement. Read from the environment so the check
# tracks whatever bar the stack under test is actually configured with.
ACTIVATION_MIN_IMPROVEMENT = float(os.environ.get("ACTIVATION_MIN_IMPROVEMENT", "0.10"))

# Taken from the seed script rather than restated. These drifted apart once already: the
# baseline dropped to 700 documents when the corpus changed, and this check kept asserting
# the old 1500 and failed a run in which every document had in fact been embedded.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed import DEFAULT_BASELINE_DOCUMENTS, DEFAULT_DRIFT_DOCUMENTS  # noqa: E402

EXPECTED_DOCUMENTS = DEFAULT_BASELINE_DOCUMENTS + DEFAULT_DRIFT_DOCUMENTS
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
    passed = documents >= EXPECTED_DOCUMENTS and embeddings >= EXPECTED_DOCUMENTS
    return DemoCheckResult(
        "document flow",
        passed,
        f"{documents}/{EXPECTED_DOCUMENTS} documents, {embeddings}/{EXPECTED_DOCUMENTS} embeddings",
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
        # SUCCEEDED means the pipeline ran to completion. Whether the candidate earned
        # activation is a separate outcome carried by the model's PASSED/REJECTED status,
        # and a rejection is a correct decision rather than a failed job.
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
    rejected = next((model for model in models if model["status"] == "REJECTED"), None)
    model = active or passed or rejected

    if not model:
        return DemoCheckResult(
            "model registry", False, f"{len(models)} models, none reached a decision"
        )

    improvement = model.get("improvementPct")
    metric = "n/a" if improvement is None else f"{float(improvement):+.1%}"

    # Assert the decision matches the evidence, in both directions. A base model that is
    # already strong on the drifted domain has no headroom left, so REJECTED is the
    # correct result and the registry must not promote anyway.
    decided = model["status"] in {"ACTIVE", "PASSED", "REJECTED"}
    consistent = True
    if improvement is not None:
        promoted = model["status"] in {"ACTIVE", "PASSED"}
        cleared_bar = float(improvement) > ACTIVATION_MIN_IMPROVEMENT
        consistent = promoted == cleared_bar

    return DemoCheckResult(
        "model registry",
        decided and consistent,
        f"version={model['version']}, status={model['status']}, improvement={metric}, "
        f"gate={ACTIVATION_MIN_IMPROVEMENT:+.0%}, decision_consistent={consistent}",
    )


async def check_retrieval_quality_improvement(client: httpx.AsyncClient) -> DemoCheckResult:
    """Serving must agree with the promotion decision, whichever way it went.

    Demanding an activated model would make the demo fail whenever the honest answer is
    that nothing beat the base model. What has to hold is consistency: if a candidate
    cleared the gate it must be the one serving traffic and it must be an improvement; if
    none did, traffic must still be served by the baseline.
    """
    baseline_version = await get_served_model_version(client, "baseline")
    active_version = await get_served_model_version(client, "auto")

    response = await client.get(f"{TRAINER_API}/v1/models")
    response.raise_for_status()
    promoted = [model for model in response.json() if model["status"] in {"ACTIVE", "PASSED"}]
    promoted = [model for model in promoted if model["version"] != "baseline"]

    if not promoted:
        serving_baseline = active_version in {baseline_version, "baseline"}
        return DemoCheckResult(
            "retrieval quality consistency",
            serving_baseline,
            f"no candidate cleared the {ACTIVATION_MIN_IMPROVEMENT:+.0%} gate, "
            f"serving stays on {active_version}",
        )

    if active_version in {baseline_version, "baseline"}:
        return DemoCheckResult(
            "retrieval quality consistency",
            False,
            f"{promoted[0]['version']} was promoted but traffic is served by {active_version}",
        )

    detail = await client.get(f"{TRAINER_API}/v1/models/{active_version}")
    detail.raise_for_status()
    improvement = detail.json().get("improvementPct")
    passed = improvement is not None and float(improvement) > ACTIVATION_MIN_IMPROVEMENT
    return DemoCheckResult(
        "retrieval quality consistency",
        passed,
        f"served_by={active_version}, improvement={float(improvement or 0):+.1%}, "
        f"gate={ACTIVATION_MIN_IMPROVEMENT:+.0%}",
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
