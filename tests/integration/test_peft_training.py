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

    texts = [
        TrainingText(
            text=f"healthcare note {index} patient cardiology hypertension",
            source="medical_records",
            domain_tag="medical_records",
        )
        for index in range(50)
    ]
    telemetry, _, onnx_dir = train_peft_model(
        texts,
        PeftTrainingConfig(
            base_model="prajjwal1/bert-tiny",
            epochs=1,
            batch_size=8,
            max_length=64,
            output_dir=str(tmp_path),
        ),
    )

    onnx_files = list(onnx_dir.glob("*.onnx"))
    assert telemetry.sample_count == 50
    assert onnx_files

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
