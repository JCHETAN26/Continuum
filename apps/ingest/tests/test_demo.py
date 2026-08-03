"""The demo controller drives the same scenario the seed script does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from continuum_ingest.api.demo import DemoController, build_payload


@pytest.mark.asyncio
async def test_publishes_both_distributions_in_order(monkeypatch):
    """Baseline first, then the shift. Reversing them would drift against nothing."""
    monkeypatch.setattr(
        "continuum_ingest.api.demo.load_corpus_documents",
        lambda: (["baseline post"] * 3, ["drifted post"] * 2),
    )
    monkeypatch.setattr("continuum_ingest.api.demo.SETTLE_SECONDS", 0.0)
    monkeypatch.setattr("continuum_ingest.api.demo.INTER_BATCH_DELAY_SECONDS", 0.0)

    published: list[dict[str, Any]] = []

    async def publish(payload: dict[str, Any]) -> None:
        published.append(payload)

    controller = DemoController()
    assert controller.start(publish) is True
    await controller._task

    assert controller.state.phase == "complete"
    assert [item["source"] for item in published] == [
        "pc_hardware",
        "pc_hardware",
        "pc_hardware",
        "mac_hardware",
        "mac_hardware",
    ]
    assert controller.state.ingested == 5


@pytest.mark.asyncio
async def test_refuses_a_second_concurrent_run(monkeypatch):
    """Two runs would interleave distributions and the drift score would describe neither."""
    monkeypatch.setattr(
        "continuum_ingest.api.demo.load_corpus_documents",
        lambda: (["a"], ["b"]),
    )
    monkeypatch.setattr("continuum_ingest.api.demo.SETTLE_SECONDS", 0.2)
    monkeypatch.setattr("continuum_ingest.api.demo.INTER_BATCH_DELAY_SECONDS", 0.0)

    async def publish(payload: dict[str, Any]) -> None:
        return None

    controller = DemoController()
    assert controller.start(publish) is True
    assert controller.start(publish) is False

    await controller._task
    assert controller.start(publish) is True
    await controller._task


@pytest.mark.asyncio
async def test_failure_is_reported_rather_than_swallowed(monkeypatch):
    def explode() -> tuple[list[str], list[str]]:
        raise RuntimeError("corpus unavailable")

    monkeypatch.setattr("continuum_ingest.api.demo.load_corpus_documents", explode)

    async def publish(payload: dict[str, Any]) -> None:
        return None

    controller = DemoController()
    controller.start(publish)
    await asyncio.sleep(0)
    await controller._task

    assert controller.state.phase == "failed"
    assert "corpus unavailable" in (controller.state.error or "")


def test_payload_carries_the_domain_as_source():
    payload = build_payload("some text", "mac_hardware")
    assert payload["source"] == "mac_hardware"
    assert payload["metadata"]["demo"] is True
