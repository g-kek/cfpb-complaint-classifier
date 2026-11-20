import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from contextvars import copy_context
from datetime import UTC, datetime
from hashlib import sha256

from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitMessage
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode
from sqlalchemy.exc import SQLAlchemyError

from cfpb_complaint_classifier import telemetry
from cfpb_complaint_classifier.api.health import liveness
from cfpb_complaint_classifier.core.config import Settings, get_settings
from cfpb_complaint_classifier.core.logging import setup_logging, task_context
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.db.tasks import (
    claim_attempt,
    get_or_create_traceparent,
    get_result,
    save_result,
)
from cfpb_complaint_classifier.messaging import (
    create_broker,
    declare_topology,
    publish_dead_letter,
    task_exchange,
    task_queue,
)
from cfpb_complaint_classifier.schemas import TaskMessage, TaskResult
from cfpb_complaint_classifier.services.inference import load_registered_model

logger = logging.getLogger(__name__)


async def raw_body(message: RabbitMessage) -> bytes:

    return message.body


def create_worker_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    duration = telemetry.inference_histogram(settings.inference_delay_seconds)
    broker = create_broker(settings, consumer=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        telemetry.configure(settings, "worker")
        engine = create_engine(settings)
        pool = ThreadPoolExecutor(max_workers=settings.worker_concurrency)
        app.state.db_engine = engine
        app.state.pool = pool
        app.state.slots = asyncio.Semaphore(settings.worker_concurrency)
        try:
            app.state.model = await asyncio.get_running_loop().run_in_executor(
                pool, load_registered_model, settings
            )
            async with broker:
                await declare_topology(broker, settings)
                await broker.start()
                logger.info(
                    "Worker ready: concurrency=%d, prefetch=%d, model=%s@%s",
                    settings.worker_concurrency,
                    settings.effective_prefetch_count,
                    app.state.model.name,
                    app.state.model.version,
                )
                try:
                    yield
                finally:
                    logger.info("shutdown_started")
        finally:
            await asyncio.to_thread(pool.shutdown, wait=True)
            await engine.dispose()
            logger.info("shutdown_completed")

    app = FastAPI(title="CFPB inference worker", lifespan=lifespan)
    app.state.broker = broker
    app.add_api_route("/healthz", liveness, methods=["GET"])

    @broker.subscriber(
        task_queue(settings),
        task_exchange(settings),
        ack_policy=AckPolicy.MANUAL,
        decoder=raw_body,
        no_reply=True,
    )
    async def handle_task(message: RabbitMessage) -> None:

        key = message.raw_message.message_id or sha256(message.body).hexdigest()
        parent = propagate.extract(message.headers, context=Context())
        task = None
        with suppress(ValueError):
            task = TaskMessage.model_validate_json(message.body)
        identifier = (
            str(task.task_id) if task else (message.raw_message.correlation_id or key)
        )
        with task_context(identifier, message.headers.get("batch_id")):
            if (
                telemetry.enabled()
                and not trace.get_current_span(parent).get_span_context().is_valid
            ):

                def receive_header():
                    with telemetry.span(
                        "receive", context=parent, attributes={"task.id": identifier}
                    ):
                        headers = {}
                        propagate.inject(headers)
                        return headers["traceparent"]

                try:
                    header = await get_or_create_traceparent(
                        app.state.db_engine, key, receive_header
                    )
                    parent = propagate.extract(
                        {"traceparent": header}, context=Context()
                    )
                except (SQLAlchemyError, OSError):
                    logger.warning("trace_context_storage_unavailable")
            with telemetry.span(
                "process",
                context=parent,
                kind=SpanKind.CONSUMER,
                attributes={"messaging.message.id": key, "task.id": identifier},
            ) as active:
                if message.raw_message.redelivered:
                    telemetry.redeliveries.add(1)
                async with app.state.slots:
                    telemetry.in_flight.add(1)
                    try:
                        await process_delivery(message, key, active, task)
                    except Exception as exc:
                        telemetry.error(active, exc)
                        telemetry.processed.add(1, {"result": "error"})
                        logger.exception("infrastructure_error")
                        await asyncio.sleep(settings.task_retry_delay_seconds)
                        await message.nack(requeue=True)
                    finally:
                        telemetry.in_flight.add(-1)

    async def dead_letter(message, attempt, reason, active):
        with telemetry.span(
            "publish.dlq",
            kind=SpanKind.PRODUCER,
            attributes={"delivery.attempt": attempt},
        ):
            await publish_dead_letter(
                broker, settings, message, attempts=attempt, error=reason
            )
            telemetry.processed.add(1, {"result": "dlq"})
            logger.error(
                "task_dlq", extra={"fields": {"attempt": attempt, "reason": reason}}
            )
        active.add_event("dlq_confirmed")
        active.set_status(StatusCode.ERROR, reason)
        await message.ack()

    def predict(task):
        model = app.state.model
        with (
            telemetry.span(
                "inference",
                attributes={
                    "model.alias": settings.mlflow_model_alias,
                    "model.version": model.version,
                },
            ),
            telemetry.timing(duration),
        ):
            time.sleep(settings.inference_delay_seconds)
            return model.predict(task.text)

    async def process_delivery(message, key, active, task):
        sent = message.headers.get("published_ns")
        if isinstance(sent, int):
            active.set_attribute(
                "messaging.publish_to_process_seconds",
                max(0, time.time_ns() - sent) / 1e9,
            )
        if task and await get_result(app.state.db_engine, task.task_id):
            logger.info("duplicate_committed")
            await message.ack()
            return
        attempt = await claim_attempt(
            app.state.db_engine, key, settings.task_max_attempts
        )
        active.set_attribute("delivery.attempt", attempt or settings.task_max_attempts)
        if attempt is None:
            await dead_letter(
                message, settings.task_max_attempts, "AttemptBudgetExhausted", active
            )
            return
        logger.info("task_started", extra={"fields": {"attempt": attempt}})
        try:
            if task is None:
                task = TaskMessage.model_validate_json(message.body)
            started_at = datetime.now(UTC)
            category = await asyncio.get_running_loop().run_in_executor(
                app.state.pool, copy_context().run, predict, task
            )
            result = TaskResult(
                task_id=task.task_id,
                category=category,
                model_name=app.state.model.name,
                model_version=app.state.model.version,
                created_at=task.created_at,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
        except Exception as exc:
            telemetry.error(active, exc)
            logger.exception("task_failed", extra={"fields": {"attempt": attempt}})
            if attempt < settings.task_max_attempts:
                telemetry.processed.add(1, {"result": "error"})
                await asyncio.sleep(settings.task_retry_delay_seconds)
                await message.nack(requeue=True)
            else:
                await dead_letter(message, attempt, type(exc).__name__, active)
            return
        while True:
            try:
                with telemetry.span("db.save_result"):
                    await save_result(app.state.db_engine, result)
                break
            except Exception as exc:
                telemetry.error(active, exc)
                logger.exception(
                    "storage_retry", extra={"fields": {"attempt": attempt}}
                )
                await asyncio.sleep(settings.task_retry_delay_seconds)
        committed = datetime.now(UTC)
        telemetry.task_duration.record(
            max(0, (committed - task.created_at).total_seconds())
        )
        logger.info(
            "task_commit", extra={"fields": {"committed_at": committed.isoformat()}}
        )
        await message.ack()
        telemetry.processed.add(1, {"result": "ok"})
        logger.info("task_completed")

    return app
