import importlib.util
import json
from pathlib import Path

import httpx
import pytest

seed_path = Path(__file__).resolve().parents[1] / "scripts" / "seed.py"
spec = importlib.util.spec_from_file_location("continuum_seed", seed_path)
assert spec is not None
seed = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(seed)


def test_batch_sizes_include_remainder():
    assert seed.batch_sizes(25, 10) == [10, 10, 5]
    assert seed.batch_sizes(20, 10) == [10, 10]


@pytest.mark.asyncio
async def test_send_corpus_posts_expected_batches(monkeypatch):
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"status": "accepted"})

    async def no_sleep(seconds: float) -> None:
        assert seconds == 0

    monkeypatch.setattr(seed.asyncio, "sleep", no_sleep)

    config = seed.SeedConfig(
        ingest_url="http://ingest.local/v1/ingest/batch",
        batch_size=2,
        baseline_documents=3,
        drift_documents=3,
        inter_batch_delay_seconds=0,
        window_settle_seconds=0,
        seed=7,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sent = await seed.send_corpus(
            client,
            config,
            ["alpha", "beta", "gamma"],
            "pc_hardware",
            3,
            seed.random.Random(7),
        )

    assert sent == 3
    assert len(requests) == 2

    # Every document is sent once. Sampling with replacement made a window a permutation
    # of the same handful of vectors, which is not a distribution drift can measure.
    delivered = [
        payload["text"] for request in requests for payload in json.loads(request.content.decode())
    ]
    assert sorted(delivered) == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_send_corpus_refuses_to_pad_a_short_corpus():
    """Silently repeating documents is what made the old seed data meaningless."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"status": "accepted"})

    config = seed.SeedConfig(inter_batch_delay_seconds=0, window_settle_seconds=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="holds 2 documents, need 5"):
            await seed.send_corpus(
                client, config, ["alpha", "beta"], "pc_hardware", 5, seed.random.Random(7)
            )


@pytest.mark.asyncio
async def test_send_batch_raises_on_ingest_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "broken"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await seed.send_batch(
                client,
                "http://ingest.local/v1/ingest/batch",
                ["bad document"],
                "pc_hardware",
            )


def test_validate_config_rejects_invalid_counts():
    with pytest.raises(ValueError, match="baseline document count"):
        seed.validate_config(seed.SeedConfig(baseline_documents=0))
