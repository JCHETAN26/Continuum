"""Drive the seeded demo from an HTTP call instead of a terminal.

`scripts/seed.py` already does this from a shell, which is fine for CI and useless for
showing someone the system. This runs the same scenario as a background task so the
dashboard can start it with a button and watch the pipeline react.

It publishes the same real 20 Newsgroups posts to the same ingest topic. There is no
fabricated data path here: the documents, the drift, the training and the metrics that
follow are the ones the pipeline produces on its own.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[4]

# Matches scripts/seed.py, so the button and the command line produce the same run.
BASELINE_DOCUMENTS = 700
DRIFT_DOCUMENTS = 500
BATCH_SIZE = 10
INTER_BATCH_DELAY_SECONDS = 0.2
# Long enough for a drift window to close over baseline-only documents, so the shift that
# follows lands against an established centroid rather than an empty one.
SETTLE_SECONDS = 15.0
CORPUS_SEED = 20260727


class Publisher(Protocol):
    async def __call__(self, payload: dict[str, Any]) -> None: ...


@dataclass
class DemoState:
    """What the dashboard polls while a run is in flight."""

    phase: str = "idle"
    ingested: int = 0
    total: int = BASELINE_DOCUMENTS + DRIFT_DOCUMENTS
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "ingested": self.ingested,
            "total": self.total,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


def load_corpus_documents() -> tuple[list[str], list[str]]:
    """The real posts, loaded the same way the seed script loads them."""
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from corpus import load_baseline_documents, load_drift_documents

    return (
        load_baseline_documents(BASELINE_DOCUMENTS, CORPUS_SEED),
        load_drift_documents(DRIFT_DOCUMENTS, CORPUS_SEED),
    )


def build_payload(text: str, source: str) -> dict[str, Any]:
    return {
        "document_id": str(uuid.uuid4()),
        "text": text,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {"demo": True},
    }


@dataclass
class DemoController:
    """Owns the single in-flight demo run.

    One at a time: a second run would interleave two distributions into the same drift
    window and the score would describe the overlap rather than the shift.
    """

    state: DemoState = field(default_factory=DemoState)
    _task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, publish: Publisher) -> bool:
        if self.is_running:
            return False
        self.state = DemoState(phase="starting", started_at=datetime.now(UTC))
        self._task = asyncio.create_task(self._run(publish))
        return True

    async def _run(self, publish: Publisher) -> None:
        try:
            # Loading pulls the dataset from the hub on first use and can take a while, so
            # it happens off the event loop and the phase says what is going on.
            self.state.phase = "loading corpus"
            baseline, drift = await asyncio.to_thread(load_corpus_documents)
            self.state.total = len(baseline) + len(drift)

            self.state.phase = "baseline"
            await self._publish_all(publish, baseline, "pc_hardware")

            self.state.phase = "settling"
            await asyncio.sleep(SETTLE_SECONDS)

            self.state.phase = "drift"
            await self._publish_all(publish, drift, "mac_hardware")

            self.state.phase = "complete"
            logger.info("Demo seed complete", ingested=self.state.ingested)
        except asyncio.CancelledError:
            self.state.phase = "cancelled"
            raise
        except Exception as error:
            self.state.phase = "failed"
            self.state.error = str(error)
            logger.exception("Demo seed failed")
        finally:
            self.state.finished_at = datetime.now(UTC)

    async def _publish_all(self, publish: Publisher, texts: list[str], source: str) -> None:
        for start in range(0, len(texts), BATCH_SIZE):
            for text in texts[start : start + BATCH_SIZE]:
                await publish(build_payload(text, source))
                self.state.ingested += 1
            # Paced rather than dumped, so drift windows fill over time and the dashboard
            # has something to animate instead of one instantaneous jump.
            await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)
