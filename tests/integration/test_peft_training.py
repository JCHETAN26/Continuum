import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_ML_INTEGRATION") != "1",
        reason="Set RUN_ML_INTEGRATION=1 to run PEFT/ONNX integration tests.",
    ),
]


@pytest.mark.asyncio
async def test_peft_training_registers_onnx_artifact(tmp_path: Path):
    """Integration contract for Phase 1.

    The default CI path skips this because it requires the ML dependency set and model
    weights. Dedicated ML CI should set RUN_ML_INTEGRATION=1 and provide a local HF cache.
    """

    from continuum_trainer.peft_engine import (
        PeftArtifactSet,
        PeftTrainingConfig,
        TrainingText,
        mark_model_pending_eval,
        train_peft_model,
    )

    # Distinguishable documents. Near-duplicate text drives the in-batch objective to
    # chance level (log(batch_size)) because there is genuinely nothing separating the
    # examples, which would make any trend assertion below meaningless.
    topics = [
        "cardiology stent angioplasty coronary artery",
        "oncology chemotherapy tumour biopsy staging",
        "radiology mri contrast lesion imaging",
        "pediatrics vaccination immunisation infant growth",
        "neurology seizure epilepsy eeg cortex",
        "orthopaedics fracture tibia cast rehabilitation",
        "dermatology melanoma lesion biopsy pigmentation",
        "nephrology dialysis creatinine renal failure",
        "psychiatry depression ssri cognitive therapy",
        "endocrinology insulin thyroid glucose metabolic",
    ]
    texts = [
        TrainingText(
            text=f"{topics[index % len(topics)]} case report number {index}",
            source="medical_records",
            domain_tag="medical_records",
        )
        for index in range(50)
    ]
    telemetry, _, onnx_dir = train_peft_model(
        texts,
        PeftTrainingConfig(
            base_model="prajjwal1/bert-tiny",
            epochs=4,
            batch_size=8,
            max_length=64,
            output_dir=str(tmp_path),
        ),
    )

    onnx_files = list(onnx_dir.glob("*.onnx"))
    assert telemetry.sample_count == 50
    assert onnx_files

    losses = [entry["loss"] for entry in telemetry.loss_history]
    assert losses, "training produced no loss history"

    # Regression guard for the degenerate objective. Scoring a batch against itself put
    # 1.0 on every diagonal entry, so the loss collapsed to ~1e-6 and nothing was
    # learned. A real two-view objective cannot sit that close to zero.
    assert min(losses) > 0.01, f"objective looks degenerate: {losses}"

    # Compare thirds rather than first-vs-last: per-step loss is noisy at batch size 8.
    third = max(1, len(losses) // 3)
    opening = sum(losses[:third]) / third
    closing = sum(losses[-third:]) / third
    assert closing < opening, f"loss did not trend down: {opening:.4f} -> {closing:.4f}"

    db = SimpleNamespace()
    db.execute_raw = AsyncMock(return_value=1)
    await mark_model_pending_eval(
        db,
        model_id="11111111-1111-1111-1111-111111111111",
        artifacts=PeftArtifactSet(
            adapter_config_uri="s3://continuum-models/demo/adapter.json",
            onnx_uri="s3://continuum-models/demo/model.onnx",
            onnx_sha256="a" * 64,
            onnx_bytes=onnx_files[0].stat().st_size,
            domain_tag="medical_records",
        ),
        eval_mrr=None,
    )
    assert db.execute_raw.await_count == 1
