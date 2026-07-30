"""Activating an adapted encoder is an index migration.

Every stored vector came from the previous model, and cosine distance between vectors
produced by two different encoders is meaningless. Re-encoding happens in the embedding
worker rather than the training pipeline, so it is resumable and spreads across replicas;
see apps/embedding/tests for the claim query itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_evaluation_set_is_drawn_per_source():
    """A single-domain evaluation set cannot measure retrieval at all.

    Relevance is same-source, so a set containing one source has no negatives and every
    metric returns 0.0. Ordering purely by embedding time allowed exactly that: drifted
    documents arrive last and embed last, so the most recent rows can be one source
    entirely, and one run in three reported baseline and candidate MRR both 0.0000.
    """
    from continuum_trainer.pipeline import load_training_examples

    captured: list = []

    async def query_raw(sql: str, *args):
        captured.append((sql, args))
        return [
            {"source": "pc_hardware", "text": "isa irq", "vec_str": "[0.1,0.2]"},
            {"source": "mac_hardware", "text": "quadra vram", "vec_str": "[0.3,0.4]"},
            {"source": "pc_hardware", "text": "bios beep", "vec_str": "[0.5,0.6]"},
            {"source": "mac_hardware", "text": "scsi chain", "vec_str": "[0.7,0.8]"},
        ]

    examples = await load_training_examples(SimpleNamespace(query_raw=query_raw), per_source=250)

    statement = " ".join(captured[0][0].split())
    assert "PARTITION BY d.source" in statement
    assert captured[0][1] == (250,)
    assert {example["source"] for example in examples} == {"pc_hardware", "mac_hardware"}


def test_demo_backend_reports_no_loss_trajectory():
    """The demo adapter solves in closed form, so it has no optimisation steps.

    It used to emit a descending curve by walking a hardcoded margin schedule
    ([0.8, 0.6, 0.4, 0.25, 0.15]) against a model that never changed between "steps". The
    curve fell because the constants fell, and the dashboard rendered it as training
    telemetry for a backend that trains nothing.
    """
    import inspect

    from continuum_trainer import pipeline

    source = inspect.getsource(pipeline)
    assert "estimate_loss_history" not in source
    # The margin schedule that manufactured the curve must not come back either.
    assert "0.25, 0.15" not in source
