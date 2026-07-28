from __future__ import annotations

import logging
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def configure_observability(service_name: str) -> None:
    """Configure JSON logs and stdout traces for local development."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=_log_level())
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_log_level()),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)


def instrument_fastapi(app: FastAPI, service_name: str) -> None:
    configure_observability(service_name)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

    @app.middleware("http")
    async def correlation_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = request.headers.get("x-correlation-id") or uuid.uuid4().hex
        token = correlation_id_var.set(correlation_id)
        clear_contextvars()
        bind_contextvars(correlation_id=correlation_id, service=service_name)
        try:
            response = await call_next(request)
            response.headers["x-correlation-id"] = correlation_id
            return response
        finally:
            clear_contextvars()
            correlation_id_var.reset(token)


def get_tracer(service_name: str) -> trace.Tracer:
    configure_observability(service_name)
    return trace.get_tracer(service_name)


def current_correlation_id() -> str | None:
    return correlation_id_var.get()


def span_attributes(**attributes: Any) -> dict[str, str | int | float | bool]:
    allowed = (str, int, float, bool)
    return {key: value for key, value in attributes.items() if isinstance(value, allowed)}


def _log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)
