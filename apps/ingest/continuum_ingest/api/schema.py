from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    document_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    timestamp: datetime
    metadata: dict[str, Any] | None = None
