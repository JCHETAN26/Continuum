import importlib.util
from pathlib import Path

import httpx
import pytest

verify_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_demo.py"
spec = importlib.util.spec_from_file_location("continuum_verify_demo", verify_path)
assert spec is not None
verify_demo = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(verify_demo)


def client_for_routes(routes: dict[str, object]) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path not in routes:
            return httpx.Response(404)
        return httpx.Response(200, json=routes[path])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_document_flow_requires_expected_counts():
    async with client_for_routes(
        {"/v1/drift/summary": {"documentCount": 1200, "embeddingCount": 1199}}
    ) as client:
        result = await verify_demo.check_document_flow(client)

    assert result.passed is False
    assert "1199/1200 embeddings" in result.detail


@pytest.mark.asyncio
async def test_drift_spike_requires_breach_above_threshold():
    async with client_for_routes(
        {
            "/v1/drift/summary": {
                "latestDriftScore": 0.7,
                "threshold": 0.35,
                "breached": True,
            }
        }
    ) as client:
        result = await verify_demo.check_drift_spike(client)

    assert result.passed is True
    assert "score=0.700" in result.detail


@pytest.mark.asyncio
async def test_training_job_fails_on_failed_completed_job():
    async with client_for_routes(
        {"/v1/training/jobs": [{"id": "job-1", "status": "FAILED", "sampleCount": 500}]}
    ) as client:
        result = await verify_demo.check_training_job(client)

    assert result.passed is False
    assert "FAILED" in result.detail


@pytest.mark.asyncio
async def test_terminal_failure_is_reported_without_waiting_out_the_budget():
    """A retried-out job records no sampleCount, and waiting cannot change its outcome.

    Requiring sampleCount meant such a job never counted as complete, so the check polled
    until its 1800s budget expired before reporting a failure visible in 22 seconds.
    """
    async with client_for_routes(
        {
            "/v1/training/jobs": [
                {
                    "id": "job-9",
                    "status": "FAILED",
                    "sampleCount": None,
                    "error": "PEFT training requires at least two training texts.",
                }
            ]
        }
    ) as client:
        result = await verify_demo.check_training_job(client)

    assert result.passed is False
    assert "job-9" in result.detail
    assert "at least two training texts" in result.detail


@pytest.mark.asyncio
async def test_model_registry_accepts_passed_or_active_model():
    async with client_for_routes(
        {
            "/v1/models": [
                {
                    "version": "demo-version",
                    "status": "PASSED",
                    "improvementPct": 0.24,
                }
            ]
        }
    ) as client:
        result = await verify_demo.check_model_registry(client)

    assert result.passed is True
    assert "demo-version" in result.detail


def quality_handler(*, served: str, models: list[dict], improvement: float | None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embed":
            requested = request.headers["x-model"]
            served_by = "baseline" if requested == "baseline" else served
            return httpx.Response(200, json={"model_version_used": served_by, "dimension": 384})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=models)
        if request.url.path == f"/v1/models/{served}":
            return httpx.Response(200, json={"version": served, "improvementPct": improvement})
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_promoted_model_must_serve_traffic_and_beat_the_gate():
    handler = quality_handler(
        served="demo-version",
        models=[{"version": "demo-version", "status": "ACTIVE"}],
        improvement=0.42,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await verify_demo.check_retrieval_quality_improvement(client)

    assert result.passed is True
    assert "demo-version" in result.detail


@pytest.mark.asyncio
async def test_promoted_model_failing_the_gate_is_not_accepted():
    handler = quality_handler(
        served="demo-version",
        models=[{"version": "demo-version", "status": "ACTIVE"}],
        improvement=0.008,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await verify_demo.check_retrieval_quality_improvement(client)

    assert result.passed is False


@pytest.mark.asyncio
async def test_serving_the_baseline_is_correct_when_nothing_was_promoted():
    """The honest outcome when the base model already has no headroom left."""
    handler = quality_handler(served="baseline", models=[], improvement=None)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await verify_demo.check_retrieval_quality_improvement(client)

    assert result.passed is True
    assert "gate" in result.detail


@pytest.mark.asyncio
async def test_promotion_that_never_reached_serving_is_a_failure():
    handler = quality_handler(
        served="baseline",
        models=[{"version": "demo-version", "status": "ACTIVE"}],
        improvement=0.42,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await verify_demo.check_retrieval_quality_improvement(client)

    assert result.passed is False
    assert "served by" in result.detail


@pytest.mark.asyncio
async def test_wait_for_times_out_with_last_detail():
    async def never_passes(client: httpx.AsyncClient) -> object:
        return verify_demo.DemoCheckResult("demo", False, "still waiting")

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await verify_demo.wait_for(client, "demo", 0.01, 0, never_passes)

    assert result.passed is False
    assert "still waiting" in result.detail


def test_expected_document_count_tracks_the_seed_script():
    """These drifted apart once: the baseline dropped to 700 when the corpus changed and
    this check kept asserting 1500, failing a run where every document had been embedded."""
    import importlib.util
    import sys
    from pathlib import Path

    seed_path = Path(__file__).resolve().parents[1] / "scripts" / "seed.py"
    spec = importlib.util.spec_from_file_location("continuum_seed_for_counts", seed_path)
    assert spec and spec.loader
    seed_module = importlib.util.module_from_spec(spec)
    sys.modules["continuum_seed_for_counts"] = seed_module
    spec.loader.exec_module(seed_module)

    assert verify_demo.EXPECTED_DOCUMENTS == (
        seed_module.DEFAULT_BASELINE_DOCUMENTS + seed_module.DEFAULT_DRIFT_DOCUMENTS
    )


@pytest.mark.asyncio
async def test_registry_check_reports_the_judged_candidate_not_the_active_model():
    """The measured comparison is the point of the run and must appear in the output.

    When nothing clears the gate the baseline stays ACTIVE carrying no metrics, so the
    check printed improvement=n/a and the numbers were only recoverable from container
    logs that a passing run discarded.
    """
    models = [
        {"version": "baseline", "status": "ACTIVE"},
        {
            "version": "2026.07.29-abc",
            "status": "REJECTED",
            "improvementPct": 0.031,
            "metrics": {"mrr": 0.8871},
            "baselineMetrics": {"mrr": 0.8604},
        },
    ]
    async with client_for_routes({"/v1/models": models}) as client:
        result = await verify_demo.check_model_registry(client)

    assert "baseline_mrr=0.8604" in result.detail
    assert "candidate_mrr=0.8871" in result.detail
    assert "improvement=+3.10%" in result.detail
    assert "status=REJECTED" in result.detail
    # Rejected below the gate is a consistent decision, so the check still passes.
    assert result.passed is True
