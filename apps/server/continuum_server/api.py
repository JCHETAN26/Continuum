from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import structlog
from prometheus_client import make_asgi_app, Counter, Histogram
from contextlib import asynccontextmanager

from continuum_server.engine import engine, background_poller
import asyncio
import os

logger = structlog.get_logger()

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

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_version_used: str
    dimension: int

def verify_api_key(x_api_key: str = Header(None)):
    expected = os.getenv("API_KEY", "continuum-secret-key")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_version": engine.current_version}

@app.post("/v1/embed", response_model=EmbedResponse)
@LATENCY.time()
async def embed(req: EmbedRequest, x_api_key: str = Depends(verify_api_key)):
    REQUEST_COUNT.inc()
    
    if len(req.texts) > 32:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 32")
        
    try:
        embeddings, version, dim = await engine.embed_batch(req.texts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
        
    return EmbedResponse(
        embeddings=embeddings,
        model_version_used=version,
        dimension=dim
    )
