from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from continuum_linguistic_drift import api


@pytest.mark.asyncio
async def test_linguistic_event_payload_contains_windows_and_summary(monkeypatch):
    now = datetime.now(UTC)
    monkeypatch.setattr(
        api,
        "get_linguistic_status",
        AsyncMock(
            return_value=[
                api.LinguisticWindowResponse(
                    id="window-id",
                    windowStart=now,
                    windowEnd=now,
                    documentCount=5,
                    entityKlDivergence=0.7,
                    topicWasserstein=0.5,
                    vocabChi2Pvalue=0.01,
                    compositeScore=0.82,
                    threshold=0.65,
                    breached=True,
                    newEntities=[],
                    emergingTopics=[],
                    emergingTerms=[],
                    createdAt=now,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        api,
        "get_linguistic_summary",
        AsyncMock(
            return_value=api.LinguisticSummaryResponse(
                latestCompositeScore=0.82,
                breached=True,
                threshold=0.65,
                windowCount=1,
            )
        ),
    )

    payload = await api.get_linguistic_event_payload()

    assert payload["windows"][0]["id"] == "window-id"
    assert payload["summary"]["breached"] is True
