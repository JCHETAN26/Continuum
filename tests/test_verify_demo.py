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
        {"/v1/drift/summary": {"documentCount": 1500, "embeddingCount": 1499}}
    ) as client:
        result = await verify_demo.check_document_flow(client)

    assert result.passed is False
    assert "1499 embeddings" in result.detail


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
    assert "status=FAILED" in result.detail


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


@pytest.mark.asyncio
async def test_retrieval_quality_improvement_requires_active_model_with_gain():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embed":
            requested = request.headers["x-model"]
            served_by = "baseline" if requested == "baseline" else "demo-version"
            return httpx.Response(200, json={"model_version_used": served_by, "dimension": 384})
        if request.url.path == "/v1/models/demo-version":
            return httpx.Response(200, json={"version": "demo-version", "improvementPct": 0.008})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await verify_demo.check_retrieval_quality_improvement(client)

    assert result.passed is True
    assert "demo-version" in result.detail


@pytest.mark.asyncio
async def test_wait_for_times_out_with_last_detail():
    async def never_passes(client: httpx.AsyncClient) -> object:
        return verify_demo.DemoCheckResult("demo", False, "still waiting")

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await verify_demo.wait_for(client, "demo", 0.01, 0, never_passes)

    assert result.passed is False
    assert "still waiting" in result.detail
