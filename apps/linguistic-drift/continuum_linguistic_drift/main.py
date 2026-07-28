import asyncio

import structlog
import uvicorn

from continuum_linguistic_drift.api import app
from continuum_linguistic_drift.worker import run_linguistic_drift_worker

logger = structlog.get_logger()


async def serve_api():
    config = uvicorn.Config(app, host="0.0.0.0", port=8004, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    logger.info("Starting continuum-linguistic-drift service")
    await asyncio.gather(serve_api(), run_linguistic_drift_worker())


if __name__ == "__main__":
    asyncio.run(main())
