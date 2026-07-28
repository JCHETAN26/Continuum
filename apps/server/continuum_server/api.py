import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from continuum_shared.observability import get_tracer, instrument_fastapi
from continuum_shared.security import constant_time_equals, verify_api_key_hash
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from continuum_server.engine import background_poller, engine

logger = structlog.get_logger()
tracer = get_tracer("continuum-server")

# Prometheus metrics
REQUEST_COUNT = Counter("embed_requests_total", "Total embed requests")
LATENCY = Histogram("embed_latency_seconds", "Latency of embed requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.connect()
    poller_task = asyncio.create_task(background_poller())
    yield
    poller_task.cancel()
    await engine.disconnect()


app = FastAPI(title="Continuum Serving REST API", lifespan=lifespan)
instrument_fastapi(app, "continuum-server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_version_used: str
    dimension: int


class RollbackEventResponse(BaseModel):
    id: str
    failedVersion: str
    restoredVersion: str
    errorRate: float
    requestCount: int
    createdAt: datetime


def verify_api_key(x_api_key: str = Header(None)):
    bcrypt_hash = os.getenv("API_KEY_BCRYPT_HASH")
    expected = os.getenv("API_KEY", "continuum-secret-key")
    if bcrypt_hash:
        valid = verify_api_key_hash(x_api_key or "", bcrypt_hash)
    else:
        valid = constant_time_equals(x_api_key or "", expected)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_version": engine.current_version}


@app.post("/v1/embed", response_model=EmbedResponse)
@LATENCY.time()
async def embed(
    req: EmbedRequest,
    x_api_key: str = Depends(verify_api_key),
    x_model: str = Header("auto"),
):
    REQUEST_COUNT.inc()

    if len(req.texts) > 32:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 32")

    start = time.perf_counter()
    status_code = 200
    version = x_model if x_model != "auto" else engine.current_version or "unknown"
    try:
        with tracer.start_as_current_span(
            "server.embed",
            attributes={"request.batch_size": len(req.texts), "model.requested": x_model},
        ):
            embeddings, version, dim = await engine.embed_batch(req.texts, model_version=x_model)
    except RuntimeError as e:
        status_code = 503
        await engine.record_request_metric(
            version,
            status_code=status_code,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        raise HTTPException(status_code=status_code, detail=str(e))

    await engine.record_request_metric(
        version,
        status_code=status_code,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    await engine.rollback_if_needed()

    return EmbedResponse(embeddings=embeddings, model_version_used=version, dimension=dim)


@app.get("/v1/rollbacks", response_model=list[RollbackEventResponse])
async def list_rollbacks() -> list[RollbackEventResponse]:
    rows = await load_rollback_rows()
    return [rollback_row_to_response(row) for row in rows]


@app.get("/v1/rollbacks/events")
async def stream_rollback_events():
    async def events():
        while True:
            payload = [event.model_dump(mode="json") for event in await list_rollbacks()]
            yield f"event: rollbacks\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(events(), media_type="text/event-stream")


async def load_rollback_rows() -> list[dict[str, Any]]:
    try:
        return await engine.db.query_raw(
            """
            SELECT
                id::text,
                failed_version,
                restored_version,
                error_rate,
                request_count,
                created_at
            FROM model_rollbacks
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
    except Exception as e:
        logger.debug("Unable to load rollback events", error=str(e))
        return []


def rollback_row_to_response(row: dict[str, Any]) -> RollbackEventResponse:
    return RollbackEventResponse(
        id=row["id"],
        failedVersion=row["failed_version"],
        restoredVersion=row["restored_version"],
        errorRate=row["error_rate"],
        requestCount=row["request_count"],
        createdAt=row["created_at"],
    )
