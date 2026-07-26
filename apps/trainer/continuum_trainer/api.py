from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
import uuid

from continuum_shared.prisma import Prisma
from continuum_shared.prisma.enums import ModelStatus
from continuum_trainer.tasks import queue_training_job

db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

app = FastAPI(title="Continuum Trainer API", lifespan=lifespan)

class ModelVersionResponse(BaseModel):
    id: str
    version: str
    baseModel: str
    status: str
    artifactUri: Optional[str]
    artifactSha256: Optional[str]

@app.post("/v1/models", response_model=ModelVersionResponse)
async def create_model():
    """Trigger a new asynchronous training job"""
    # Create version name like '2026.07.26-1'
    # For simplicity, we just use the date and a short uuid
    date_str = datetime.now().strftime("%Y.%m.%d")
    short_uuid = str(uuid.uuid4())[:8]
    version_name = f"{date_str}-{short_uuid}"
    
    # We must mock training logic as instructed for fast execution if we want tests,
    # but the task will be real queueing.
    
    model = await db.modelversion.create(
        data={
            "version": version_name,
            "baseModel": "sentence-transformers/all-MiniLM-L6-v2",
            "status": ModelStatus.DRAFT,
        }
    )
    
    # Queue the training job
    queue_training_job(model.id)
    
    return ModelVersionResponse(
        id=model.id,
        version=model.version,
        baseModel=model.baseModel,
        status=model.status,
        artifactUri=model.artifactUri,
        artifactSha256=model.artifactSha256
    )

@app.get("/v1/models/{version}", response_model=ModelVersionResponse)
async def get_model(version: str):
    model = await db.modelversion.find_unique(where={"version": version})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    return ModelVersionResponse(
        id=model.id,
        version=model.version,
        baseModel=model.baseModel,
        status=model.status,
        artifactUri=model.artifactUri,
        artifactSha256=model.artifactSha256
    )

@app.post("/v1/models/{version}/activate", response_model=ModelVersionResponse)
async def activate_model(version: str):
    model = await db.modelversion.find_unique(where={"version": version})
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    if model.status != ModelStatus.PASSED and model.status != ModelStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Model must be in PASSED state to activate")
        
    # We must ensure only 1 model is active.
    # Prisma transaction to deactivate old and activate new
    async with db.tx() as tx:
        # Find currently active
        active_models = await tx.modelversion.find_many(where={"status": ModelStatus.ACTIVE})
        for active in active_models:
            # Revert old active to PASSED or something? Let's just say PASSED
            await tx.modelversion.update(
                where={"id": active.id},
                data={"status": ModelStatus.PASSED}
            )
            
        # Set new to active
        updated = await tx.modelversion.update(
            where={"id": model.id},
            data={"status": ModelStatus.ACTIVE}
        )
        
    assert updated is not None
    
    return ModelVersionResponse(
        id=updated.id,
        version=updated.version,
        baseModel=updated.baseModel,
        status=updated.status,
        artifactUri=updated.artifactUri,
        artifactSha256=updated.artifactSha256
    )
