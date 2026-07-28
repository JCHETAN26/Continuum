from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from continuum_shared.config import settings
from continuum_shared.observability import instrument_fastapi
from continuum_shared.prisma import Prisma
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

db = Prisma()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="Continuum Linguistic Drift API", lifespan=lifespan)
instrument_fastapi(app, "continuum-linguistic-drift")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LinguisticWindowResponse(BaseModel):
    id: str
    windowStart: datetime
    windowEnd: datetime
    documentCount: int
    entityKlDivergence: float
    topicWasserstein: float
    vocabChi2Pvalue: float
    compositeScore: float
    threshold: float
    breached: bool
    newEntities: list[dict[str, Any]]
    emergingTopics: list[dict[str, Any]]
    emergingTerms: list[dict[str, Any]]
    createdAt: datetime


class LinguisticSummaryResponse(BaseModel):
    latestCompositeScore: float
    breached: bool
    threshold: float
    windowCount: int


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/linguistic/status", response_model=list[LinguisticWindowResponse])
async def get_linguistic_status():
    rows = await db.query_raw(
        """
        SELECT
            id::text,
            window_start,
            window_end,
            document_count,
            entity_kl_divergence,
            topic_wasserstein,
            vocab_chi2_pvalue,
            composite_score,
            threshold,
            breached,
            new_entities,
            emerging_topics,
            emerging_terms,
            created_at
        FROM linguistic_windows
        ORDER BY window_start DESC
        LIMIT 10
        """
    )
    return [row_to_response(row) for row in rows]


@app.get("/v1/linguistic/summary", response_model=LinguisticSummaryResponse)
async def get_linguistic_summary():
    count_rows = await db.query_raw("SELECT COUNT(*)::int AS count FROM linguistic_windows")
    rows = await db.query_raw(
        """
        SELECT composite_score, breached, threshold
        FROM linguistic_windows
        ORDER BY window_start DESC
        LIMIT 1
        """
    )
    latest = rows[0] if rows else None
    return LinguisticSummaryResponse(
        latestCompositeScore=latest["composite_score"] if latest else 0.0,
        breached=latest["breached"] if latest else False,
        threshold=latest["threshold"] if latest else settings.linguistic_drift_threshold,
        windowCount=count_rows[0]["count"] if count_rows else 0,
    )


@app.get("/v1/linguistic/events")
async def stream_linguistic_events():
    async def events():
        while True:
            payload = await get_linguistic_event_payload()
            yield f"event: linguistic\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(events(), media_type="text/event-stream")


async def get_linguistic_event_payload():
    windows = await get_linguistic_status()
    summary = await get_linguistic_summary()
    return {
        "windows": [window.model_dump(mode="json") for window in windows],
        "summary": summary.model_dump(mode="json"),
    }


@app.get("/v1/linguistic/windows/{window_id}", response_model=LinguisticWindowResponse)
async def get_linguistic_window(window_id: str):
    rows = await db.query_raw(
        """
        SELECT
            id::text,
            window_start,
            window_end,
            document_count,
            entity_kl_divergence,
            topic_wasserstein,
            vocab_chi2_pvalue,
            composite_score,
            threshold,
            breached,
            new_entities,
            emerging_topics,
            emerging_terms,
            created_at
        FROM linguistic_windows
        WHERE id = $1::uuid
        LIMIT 1
        """,
        window_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Linguistic window not found")
    return row_to_response(rows[0])


def row_to_response(row: dict[str, Any]) -> LinguisticWindowResponse:
    return LinguisticWindowResponse(
        id=row["id"],
        windowStart=row["window_start"],
        windowEnd=row["window_end"],
        documentCount=row["document_count"],
        entityKlDivergence=row["entity_kl_divergence"],
        topicWasserstein=row["topic_wasserstein"],
        vocabChi2Pvalue=row["vocab_chi2_pvalue"],
        compositeScore=row["composite_score"],
        threshold=row["threshold"],
        breached=row["breached"],
        newEntities=row["new_entities"],
        emergingTopics=row["emerging_topics"],
        emergingTerms=row["emerging_terms"],
        createdAt=row["created_at"],
    )
