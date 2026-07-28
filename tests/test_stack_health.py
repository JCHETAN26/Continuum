import importlib.util
from pathlib import Path

import httpx
import pytest

health_path = Path(__file__).resolve().parents[1] / "scripts" / "check_stack_health.py"
spec = importlib.util.spec_from_file_location("continuum_stack_health", health_path)
assert spec is not None
stack_health = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stack_health)


@pytest.mark.asyncio
async def test_check_targets_reports_all_healthy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    targets = [
        stack_health.HealthTarget("ingest", "http://local/ingest"),
        stack_health.HealthTarget("drift", "http://local/drift"),
    ]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await stack_health.check_targets(client, targets)

    assert all(result.healthy for result in results)
    assert [result.name for result in results] == ["ingest", "drift"]


@pytest.mark.asyncio
async def test_check_targets_reports_unhealthy_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        status = 503 if str(request.url).endswith("/server") else 200
        return httpx.Response(status)

    targets = [
        stack_health.HealthTarget("ingest", "http://local/ingest"),
        stack_health.HealthTarget("server", "http://local/server"),
    ]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await stack_health.check_targets(client, targets)

    assert results[0].healthy is True
    assert results[1].healthy is False
    assert "503" in results[1].detail


@pytest.mark.asyncio
async def test_wait_for_stack_returns_success_when_every_target_is_healthy(monkeypatch):
    async def mock_check_targets(client, targets):
        return [stack_health.HealthResult(target.name, True, target.url) for target in targets]

    monkeypatch.setattr(stack_health, "check_targets", mock_check_targets)

    code = await stack_health.wait_for_stack(
        timeout_seconds=1,
        interval_seconds=0,
        targets=[stack_health.HealthTarget("dashboard", "http://local/dashboard")],
    )

    assert code == 0
