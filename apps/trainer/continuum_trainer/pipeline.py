import asyncio
import structlog
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer
from continuum_shared.prisma import Prisma
from continuum_shared.prisma.enums import ModelStatus
import uuid
import os

logger = structlog.get_logger()

# We use asyncio to interface with the async Prisma client inside a synchronous RQ task
def run_training_pipeline(model_id: str):
    asyncio.run(_async_run_training_pipeline(model_id))

async def _async_run_training_pipeline(model_id: str):
    db = Prisma()
    await db.connect()
    
    try:
        model_version = await db.modelversion.find_unique(where={"id": model_id})
        if not model_version:
            logger.error("Model not found", model_id=model_id)
            return
            
        logger.info("Starting training pipeline", version=model_version.version)
        
        # Mark as evaluating/training
        await db.modelversion.update(
            where={"id": model_id},
            data={"status": ModelStatus.EVALUATING}
        )
        
        # In a real environment we'd load the base model, apply LoRA, and train on drifting documents.
        base_model_name = model_version.baseModel
        
        # For demonstration and testability, we wrap the heavy HF logic
        try:
            # We wrap this in a condition to allow tests to mock it out easily
            await _run_hf_lora_training(base_model_name)
        except Exception as e:
            logger.warning("HF training failed or skipped", error=str(e))
            pass
            
        # Run Evaluation
        from continuum_trainer.eval import evaluate_model
        passed = await evaluate_model(model_version.version)
        
        # Determine status
        new_status = ModelStatus.PASSED if passed else ModelStatus.REJECTED
        
        # "Export" and "Upload" dummy logic
        artifact_uri = f"s3://continuum-models/{model_version.version}/model.onnx"
        artifact_sha256 = "dummy_sha256_hash_here"
        
        # Update database
        await db.modelversion.update(
            where={"id": model_id},
            data={
                "status": new_status,
                "artifactUri": artifact_uri,
                "artifactSha256": artifact_sha256
            }
        )
        
        logger.info("Training pipeline completed", version=model_version.version, status=new_status)
        
    finally:
        await db.disconnect()


async def _run_hf_lora_training(base_model_name: str):
    """Encapsulated HF logic that can be easily mocked in tests"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name)
    
    # Configure LoRA
    # Rank=8, alpha=16 as per spec for efficient training
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"], # Typical attention targets
        lora_dropout=0.1,
        bias="none",
    )
    
    peft_model = get_peft_model(model, config)
    logger.info("LoRA model prepared", trainable_params=peft_model.print_trainable_parameters())
    
    # Normally we would use transformers.Trainer here with a dataset of hard negatives.
    # ...
