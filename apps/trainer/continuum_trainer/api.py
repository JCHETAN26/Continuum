import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.enums import ModelStatus, TrainingJobStatus, TrainingTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from continuum_trainer.pipeline import run_training_pipeline

db = Prisma()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="Continuum Trainer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelVersionResponse(BaseModel):
    id: str
    version: str
    baseModel: str
    status: str
    artifactUri: str | None
    artifactSha256: str | None
    metrics: Any | None = None
    baselineMetrics: Any | None = None
    improvementPct: float | None = None


class TrainingJobResponse(BaseModel):
    id: str
    status: str
    trigger: str
    modelVersionId: str | None
    driftWindowId: str | None
    sampleCount: int | None
    lossHistory: Any | None
    queuedAt: datetime
    startedAt: datetime | None
    finishedAt: datetime | None


@app.post("/v1/models", response_model=ModelVersionResponse)
async def create_model(background_tasks: BackgroundTasks):
    """Trigger a new asynchronous training job"""
    # Create version name like '2026.07.26-1'
    # For simplicity, we just use the date and a short uuid
    date_str = datetime.now().strftime("%Y.%m.%d")
    short_uuid = str(uuid.uuid4())[:8]
    version_name = f"{date_str}-{short_uuid}"

    model = await db.modelversion.create(
        data={
            "version": version_name,
            "baseModel": "sentence-transformers/all-MiniLM-L6-v2",
            "status": ModelStatus.DRAFT,
        }
    )

    job = await db.trainingjob.create(
        data={
            "status": TrainingJobStatus.QUEUED,
            "trigger": TrainingTrigger.MANUAL,
            "modelVersionId": model.id,
            "baseModel": model.baseModel,
            "hyperparameters": Json({"lora_rank": 8, "lora_alpha": 16, "demo": True}),
        }
    )

    background_tasks.add_task(run_training_pipeline, model.id, job.id)

    return ModelVersionResponse(
        id=model.id,
        version=model.version,
        baseModel=model.baseModel,
        status=model.status,
        artifactUri=model.artifactUri,
        artifactSha256=model.artifactSha256,
        metrics=model.metrics,
        baselineMetrics=model.baselineMetrics,
        improvementPct=model.improvementPct,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/models", response_model=list[ModelVersionResponse])
async def list_models():
    models = await db.modelversion.find_many(take=20, order={"createdAt": "desc"})
    return [
        ModelVersionResponse(
            id=model.id,
            version=model.version,
            baseModel=model.baseModel,
            status=model.status,
            artifactUri=model.artifactUri,
            artifactSha256=model.artifactSha256,
            metrics=model.metrics,
            baselineMetrics=model.baselineMetrics,
            improvementPct=model.improvementPct,
        )
        for model in models
    ]


@app.get("/v1/training/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs():
    jobs = await db.trainingjob.find_many(take=20, order={"queuedAt": "desc"})
    return [
        TrainingJobResponse(
            id=job.id,
            status=job.status,
            trigger=job.trigger,
            modelVersionId=job.modelVersionId,
            driftWindowId=job.driftWindowId,
            sampleCount=job.sampleCount,
            lossHistory=job.lossHistory,
            queuedAt=job.queuedAt,
            startedAt=job.startedAt,
            finishedAt=job.finishedAt,
        )
        for job in jobs
    ]


@app.get("/v1/training/events")
async def stream_training_events():
    async def events():
        while True:
            payload = await get_training_event_payload()
            yield f"event: training\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(events(), media_type="text/event-stream")


async def get_training_event_payload():
    models = await list_models()
    jobs = await list_training_jobs()
    return {
        "models": [model.model_dump(mode="json") for model in models],
        "jobs": [job.model_dump(mode="json") for job in jobs],
    }


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
        artifactSha256=model.artifactSha256,
        metrics=model.metrics,
        baselineMetrics=model.baselineMetrics,
        improvementPct=model.improvementPct,
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
                where={"id": active.id}, data={"status": ModelStatus.PASSED}
            )

        # Set new to active
        updated = await tx.modelversion.update(
            where={"id": model.id},
            data={"status": ModelStatus.ACTIVE, "activatedAt": datetime.now(UTC)},
        )

    assert updated is not None

    return ModelVersionResponse(
        id=updated.id,
        version=updated.version,
        baseModel=updated.baseModel,
        status=updated.status,
        artifactUri=updated.artifactUri,
        artifactSha256=updated.artifactSha256,
        metrics=updated.metrics,
        baselineMetrics=updated.baselineMetrics,
        improvementPct=updated.improvementPct,
    )
