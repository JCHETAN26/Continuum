from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from continuum_shared.prisma import Prisma
from continuum_shared.config import settings

db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

app = FastAPI(title="Continuum Drift API", lifespan=lifespan)

class DriftWindowResponse(BaseModel):
    id: str
    windowSize: str
    windowStart: datetime
    windowEnd: datetime
    documentCount: int
    driftScore: float
    wassersteinDistance: Optional[float]
    threshold: float
    breached: bool
    baselineId: Optional[str]
    createdAt: datetime

@app.get("/v1/drift/status", response_model=List[DriftWindowResponse])
async def get_drift_status():
    """Returns the most recent drift windows across all window sizes."""
    
    # We want the latest ONE_HOUR, FIVE_MIN, etc.
    # We can fetch the latest 10 windows for simplicity
    windows = await db.driftwindow.find_many(
        take=10,
        order={"windowStart": "desc"}
    )
    
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
        ) for w in windows
    ]

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
