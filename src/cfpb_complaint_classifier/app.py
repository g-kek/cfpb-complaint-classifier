import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from opentelemetry import propagate
from opentelemetry.trace import SpanKind, StatusCode
from starlette.responses import Response

from cfpb_complaint_classifier import telemetry
from cfpb_complaint_classifier.api import health, process, version
from cfpb_complaint_classifier.core.config import get_settings
from cfpb_complaint_classifier.core.logging import setup_logging, task_context
from cfpb_complaint_classifier.core.version import get_application_version
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.messaging import create_broker, declare_topology

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    telemetry.configure(settings, "api")
    app.state.settings = settings
    app.state.db_engine = create_engine(settings)
    try:
        async with create_broker(settings) as broker:
            await declare_topology(broker, settings)
            app.state.broker = broker
            logger.info("Task API started")
            try:
                yield
            finally:
                logger.info("shutdown_started")
    finally:
        await app.state.db_engine.dispose()
        logger.info("shutdown_completed")


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)
    app = FastAPI(
        title="CFPB Complaint Classifier",
        version=get_application_version(),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        started = time.perf_counter()
        code = 500
        with telemetry.span(
            "HTTP request",
            context=propagate.extract(request.headers),
            kind=SpanKind.SERVER,
        ) as active:
            try:
                response = await call_next(request)
                code = response.status_code
                return response
            finally:
                route = getattr(request.scope.get("route"), "path", "unmatched")
                method = (
                    request.method
                    if request.method
                    in {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}
                    else "OTHER"
                )
                active.update_name(f"{method} {route}")
                active.set_attributes(
                    {
                        "http.route": route,
                        "http.request.method": method,
                        "http.response.status_code": code,
                    }
                )
                if code >= 500:
                    active.set_status(StatusCode.ERROR)
                labels = {"route": route, "status": str(code)}
                telemetry.requests.add(1, labels)
                elapsed = time.perf_counter() - started
                telemetry.api_duration.record(elapsed, labels)
                with task_context(getattr(request.state, "task_id", None)):
                    level = (
                        logging.DEBUG
                        if route in {"/healthz", "/api/v1/health"}
                        else logging.INFO
                    )
                    logger.log(
                        logging.WARNING if code >= 500 else level,
                        "http_response",
                        extra={"fields": {**labels, "duration_seconds": elapsed}},
                    )

    app.include_router(health.router)
    app.include_router(version.router)
    app.include_router(process.router)
    return app


app = create_app()
