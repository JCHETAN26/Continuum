import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from continuum_shared.config import settings
from continuum_shared.observability import instrument_fastapi
from continuum_shared.prisma import Prisma
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

db = Prisma()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="Continuum Drift API", lifespan=lifespan)
instrument_fastapi(app, "continuum-drift")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DriftWindowResponse(BaseModel):
    id: str
    windowSize: str
    windowStart: datetime
    windowEnd: datetime
    documentCount: int
    driftScore: float
    wassersteinDistance: float | None
    threshold: float
    breached: bool
    baselineId: str | None
    createdAt: datetime


class DriftSummaryResponse(BaseModel):
    documentCount: int
    embeddingCount: int
    latestDriftScore: float
    breached: bool
    threshold: float


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/drift/status", response_model=list[DriftWindowResponse])
async def get_drift_status():
    """Returns the most recent drift windows across all window sizes."""

    # We want the latest ONE_HOUR, FIVE_MIN, etc.
    # We can fetch the latest 10 windows for simplicity
    windows = await db.driftwindow.find_many(take=10, order={"windowStart": "desc"})

    return [
        DriftWindowResponse(
            id=w.id,
            windowSize=w.windowSize,
            windowStart=w.windowStart,
            windowEnd=w.windowEnd,
            documentCount=w.documentCount,
            driftScore=w.driftScore,
            wassersteinDistance=w.wassersteinDistance,
            threshold=w.threshold,
            breached=w.breached,
            baselineId=w.baselineId,
            createdAt=w.createdAt,
        )
        for w in windows
    ]


@app.get("/v1/drift/summary", response_model=DriftSummaryResponse)
async def get_drift_summary():
    document_rows = await db.query_raw("SELECT COUNT(*)::int AS count FROM documents")
    embedding_rows = await db.query_raw("SELECT COUNT(*)::int AS count FROM embeddings")
    latest = await db.driftwindow.find_first(
        where={"documentCount": {"gt": 0}},
        order={"windowStart": "desc"},
    )

    return DriftSummaryResponse(
        documentCount=document_rows[0]["count"] if document_rows else 0,
        embeddingCount=embedding_rows[0]["count"] if embedding_rows else 0,
        latestDriftScore=latest.driftScore if latest else 0.0,
        breached=latest.breached if latest else False,
        threshold=latest.threshold if latest else settings.drift_threshold,
    )


@app.get("/v1/drift/events")
async def stream_drift_events():
    async def events():
        while True:
            payload = await get_drift_event_payload()
            yield f"event: drift\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(events(), media_type="text/event-stream")


async def get_drift_event_payload():
    windows = await get_drift_status()
    summary = await get_drift_summary()
    projection = await get_embedding_projection()
    return {
        "windows": [window.model_dump(mode="json") for window in windows],
        "summary": summary.model_dump(mode="json"),
        "projection": projection,
    }


@app.get("/v1/embeddings/projection")
async def get_embedding_projection():
    rows = await db.query_raw(
        """
        SELECT d.source, d.text, e.vector::text AS vec_str
        FROM embeddings e
        JOIN documents d ON d.id = e.document_id
        ORDER BY e.created_at DESC
        LIMIT 250
        """
    )

    vectors = []
    metadata = []
    for row in rows:
        vectors.append([float(value) for value in row["vec_str"].strip("[]").split(",")])
        metadata.append({"source": row["source"], "label": row["text"][:80]})

    coordinates, method = compute_projection(vectors)
    points = []
    for coordinate, item in zip(coordinates, metadata):
        points.append(
            {
                "x": coordinate[0],
                "y": coordinate[1],
                "source": item["source"],
                "label": item["label"],
            }
        )
    return {"method": method, "points": points}


def compute_projection(vectors: list[list[float]]) -> tuple[list[list[float]], str]:
    if not vectors:
        return [], "none"

    matrix = np.array(vectors, dtype=np.float32)
    if len(vectors) == 1:
        return [[0.0, 0.0]], "single-point"

    if len(vectors) < 5:
        projection = PCA(n_components=2).fit_transform(matrix)
        return normalize_projection(projection), "pca"

    perplexity = max(2, min(30, (len(vectors) - 1) // 3))
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(matrix)
    return normalize_projection(projection), "tsne"


def normalize_projection(projection: np.ndarray) -> list[list[float]]:
    centered = projection - projection.mean(axis=0)
    scale = float(np.abs(centered).max())
    if scale == 0:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(float).tolist()


@app.get("/v1/drift/windows/{window_id}", response_model=DriftWindowResponse)
async def get_drift_window(window_id: str):
    w = await db.driftwindow.find_unique(where={"id": window_id})
    if not w:
        raise HTTPException(status_code=404, detail="Drift window not found")

    return DriftWindowResponse(
        id=w.id,
        windowSize=w.windowSize,
        windowStart=w.windowStart,
        windowEnd=w.windowEnd,
        documentCount=w.documentCount,
        driftScore=w.driftScore,
        wassersteinDistance=w.wassersteinDistance,
        threshold=w.threshold,
        breached=w.breached,
        baselineId=w.baselineId,
        createdAt=w.createdAt,
    )
