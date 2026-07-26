import asyncio
import structlog
import uvicorn

from continuum_server.api import app
from continuum_server.grpc_server import serve_grpc

logger = structlog.get_logger()

async def serve_rest():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    logger.info("Starting REST server", addr="0.0.0.0:8000")
    await server.serve()

async def main():
    logger.info("Starting Continuum Server (REST + gRPC)")
    
    await asyncio.gather(
        serve_rest(),
        serve_grpc()
    )

if __name__ == "__main__":
    asyncio.run(main())
