import logging
import time

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from cfpb_complaint_classifier.schemas import (
    ComponentHealth,
    HealthReport,
    HealthStatus,
    ReportStatus,
)

logger = logging.getLogger(__name__)


async def check_postgres(engine: AsyncEngine) -> ComponentHealth:
    started_at = time.perf_counter()
    try:
        async with engine.connect() as connection:
            postgres_version = await connection.scalar(text("SHOW server_version"))
    except (OSError, SQLAlchemyError, TimeoutError) as exc:
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning("PostgreSQL health check failed: %s", type(exc).__name__)
        return ComponentHealth(
            status=HealthStatus.unavailable,
            response_time_ms=response_time_ms,
        )

    response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.debug("PostgreSQL health check passed in %.2fms", response_time_ms)
    return ComponentHealth(
        status=HealthStatus.healthy,
        version=str(postgres_version),
        response_time_ms=response_time_ms,
    )


async def build_health_report(engine: AsyncEngine) -> HealthReport:
    postgres = await check_postgres(engine)
    report_status = (
        ReportStatus.healthy
        if postgres.status is HealthStatus.healthy
        else ReportStatus.degraded
    )
    return HealthReport(status=report_status, components={"postgres": postgres})
