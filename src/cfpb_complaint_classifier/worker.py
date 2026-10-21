import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from hashlib import sha256

from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitMessage

from cfpb_complaint_classifier.api.health import liveness
from cfpb_complaint_classifier.core.config import Settings, get_settings
from cfpb_complaint_classifier.core.logging import setup_logging
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.db.tasks import claim_attempt, get_result, save_result
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
    broker = create_broker(settings, consumer=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
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
                yield
        finally:
            await asyncio.to_thread(pool.shutdown, wait=True)
            await engine.dispose()

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
        async with app.state.slots:
            try:
                task = None
                with suppress(ValueError):
                    task = TaskMessage.model_validate_json(message.body)
                if task and await get_result(app.state.db_engine, task.task_id):
                    await message.ack()
                    return
                attempt = await claim_attempt(
                    app.state.db_engine, key, settings.task_max_attempts
                )
                if attempt is None:
                    await publish_dead_letter(
                        broker,
                        settings,
                        message,
                        attempts=settings.task_max_attempts,
                        error="AttemptBudgetExhausted",
                    )
                    await message.ack()
                    return
                try:
                    task = TaskMessage.model_validate_json(message.body)
                    started_at = datetime.now(UTC)
                    await asyncio.sleep(settings.inference_delay_seconds)
                    model = app.state.model
                    category = await asyncio.get_running_loop().run_in_executor(
                        app.state.pool, model.predict, task.text
                    )
                    result = TaskResult(
                        task_id=task.task_id,
                        category=category,
                        model_name=model.name,
                        model_version=model.version,
                        created_at=task.created_at,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    )
                except Exception as exc:
                    error = type(exc).__name__
                    logger.warning("Task %s attempt %d: %s", key, attempt, error)
                    if attempt < settings.task_max_attempts:
                        await asyncio.sleep(settings.task_retry_delay_seconds)
                        await message.nack(requeue=True)
                        return
                    await publish_dead_letter(
                        broker,
                        settings,
                        message,
                        attempts=attempt,
                        error=error,
                    )
                    await message.ack()
                    return

                while True:
                    try:
                        await save_result(app.state.db_engine, result)
                    except Exception as exc:
                        logger.warning(
                            "Task %s storage error: %s", key, type(exc).__name__
                        )
                        await asyncio.sleep(settings.task_retry_delay_seconds)
                    else:
                        break
                logger.info(
                    "task_commit %s",
                    json.dumps(
                        {
                            "task_id": str(task.task_id),
                            "committed_at": datetime.now(UTC).isoformat(),
                        }
                    ),
                )
                await message.ack()
                logger.info("Task %s completed", task.task_id)
            except Exception as exc:
                logger.warning(
                    "Task %s infrastructure error: %s", key, type(exc).__name__
                )
                await asyncio.sleep(settings.task_retry_delay_seconds)
                await message.nack(requeue=True)

    return app
