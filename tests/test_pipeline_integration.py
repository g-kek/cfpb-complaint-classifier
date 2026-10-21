import asyncio
import os
import time
from unittest.mock import Mock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from cfpb_complaint_classifier import app as app_module
from cfpb_complaint_classifier import worker as worker_module
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.db.tasks import (
    create_schema,
    get_result,
    save_result,
    task_results,
)
from cfpb_complaint_classifier.messaging import (
    QueueOverloadedError,
    create_broker,
    declare_topology,
    publish_task,
    task_exchange,
    task_queue,
)
from cfpb_complaint_classifier.schemas import TaskMessage
from cfpb_complaint_classifier.services.inference import LoadedModel
from cfpb_complaint_classifier.source import publish_file, read_tasks

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Set RUN_INTEGRATION_TESTS=1 and start Docker",
    ),
]


@pytest.fixture(scope="module")
def services():
    with (
        RabbitMqContainer("rabbitmq:4.1.8-management-alpine") as rabbit,
        PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres,
    ):
        yield Settings(
            _env_file=None,
            rabbitmq_url=(
                f"amqp://guest:guest@{rabbit.get_container_host_ip()}:"
                f"{rabbit.get_exposed_port(5672)}/"
            ),
            database_url=postgres.get_connection_url(),
            task_queue_name=f"test.{uuid4().hex}",
            queue_max_length=2,
            worker_concurrency=1,
            inference_delay_seconds=0.1,
        )


@pytest.fixture(autouse=True)
async def initialize_storage(services):
    engine = create_engine(services)
    try:
        await create_schema(engine)
    finally:
        await engine.dispose()


async def test_overflow_source_resume_api_rejection_and_unique_results(
    services, tmp_path, monkeypatch
):
    settings = services
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "create_broker", create_broker)
    monkeypatch.setattr(app_module, "declare_topology", declare_topology)
    model = LoadedModel(
        Mock(predict=Mock(return_value=["Debt collection"])), "test", "7"
    )
    monkeypatch.setattr(worker_module, "load_registered_model", lambda settings: model)
    engine = create_engine(settings)
    await create_schema(engine)
    app = app_module.create_app()
    first = TaskMessage(task_id=uuid4(), text="first debt complaint")
    second = TaskMessage(task_id=uuid4(), text="second debt complaint")
    path = tmp_path / "batch.csv"
    path.write_text("text\nthird debt complaint\nfourth debt complaint\n")
    source_job = None
    try:
        async with (
            create_broker(settings) as broker,
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            await declare_topology(broker, settings)
            await publish_task(broker, settings, first)
            await publish_task(broker, settings, second)
            with pytest.raises(QueueOverloadedError):
                await publish_task(
                    broker, settings, TaskMessage(task_id=uuid4(), text="overflow")
                )
            queue = await broker.declare_queue(task_queue(settings))
            await queue.declare()
            assert queue.declaration_result.message_count == settings.queue_max_length
            started = time.monotonic()
            rejected_id = uuid4()
            response = await asyncio.wait_for(
                client.post(
                    "/api/v1/tasks", json={"text": "debt", "task_id": str(rejected_id)}
                ),
                timeout=2,
            )
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"
            assert time.monotonic() - started < 2
            assert await get_result(engine, rejected_id) is None
            source_job = asyncio.create_task(
                publish_file(path, "integration", settings)
            )
            await asyncio.sleep(0.15)
            assert not source_job.done()

            worker = worker_module.create_worker_app(settings)
            async with worker.router.lifespan_context(worker):
                assert await asyncio.wait_for(source_job, timeout=10) == 2
                task_ids = [first.task_id, second.task_id] + [
                    task.task_id for task in read_tasks(path, "integration")
                ]
                for _ in range(200):
                    if all([await get_result(engine, task_id) for task_id in task_ids]):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Accepted tasks did not reach PostgreSQL")
                response = await client.get(f"/api/v1/tasks/{first.task_id}")
                assert response.status_code == 200
                assert response.json()["model_version"] == "7"
                result = await get_result(engine, first.task_id)
                await asyncio.gather(
                    save_result(engine, result), save_result(engine, result)
                )
                async with engine.connect() as connection:
                    assert await connection.scalar(
                        select(func.count()).select_from(task_results)
                    ) == len(task_ids)
    finally:
        if source_job is not None and not source_job.done():
            source_job.cancel()
            await asyncio.gather(source_job, return_exceptions=True)
        await engine.dispose()


async def wait_until(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := await check():
            return result
        await asyncio.sleep(0.025)
    pytest.fail("Timed out waiting for the pipeline condition")


async def test_retry_budget_survives_restart_and_dlq_outage(services, monkeypatch):
    from faststream.rabbit import RabbitExchange, RabbitQueue

    from cfpb_complaint_classifier.db.tasks import claim_attempt

    settings = services.model_copy(
        update={
            "task_queue_name": f"retry.{uuid4().hex}",
            "task_retry_delay_seconds": 0.025,
            "inference_delay_seconds": 0,
        }
    )
    engine = create_engine(settings)
    await create_schema(engine)
    poison_id = str(uuid4())

    assert await claim_attempt(engine, poison_id, 3) == 1
    assert await claim_attempt(engine, poison_id, 3) == 2
    predictor = Mock(side_effect=ValueError("poison"))
    model = LoadedModel(Mock(predict=predictor), "test", "1")
    monkeypatch.setattr(worker_module, "load_registered_model", lambda _: model)
    worker = worker_module.create_worker_app(settings)
    try:
        async with (
            create_broker(settings) as broker,
            worker.router.lifespan_context(worker),
        ):
            dlq = await broker.declare_queue(
                RabbitQueue(f"{settings.task_queue_name}.dlq", durable=True)
            )
            dlx = RabbitExchange(f"{settings.task_queue_name}.dlx", durable=True)
            await dlq.unbind(dlx.name, routing_key=dlq.name)
            task = TaskMessage(task_id=uuid4(), text="poison")
            await broker.publish(
                task.model_dump(mode="json"),
                exchange=task_exchange(settings),
                routing_key=settings.task_queue_name,
                message_id=poison_id,
                persist=True,
            )

            async def attempted():
                return predictor.call_count == 1

            await wait_until(attempted)
            await asyncio.sleep(0.2)
            assert predictor.call_count == 1
            assert await get_result(engine, task.task_id) is None

            await dlq.bind(dlx.name, routing_key=dlq.name)
            dead = await wait_until(lambda: dlq.get(fail=False))
            assert dead.message_id == poison_id
            assert dead.headers["x-task-attempts"] == 3
            assert dead.body == task.model_dump_json().encode() or __import__(
                "json"
            ).loads(dead.body) == task.model_dump(mode="json")
            await dead.ack()
            assert predictor.call_count == 1
    finally:
        await engine.dispose()


async def test_invalid_json_reaches_dlq_on_real_broker(services, monkeypatch):
    from faststream.rabbit import RabbitQueue

    settings = services.model_copy(
        update={
            "task_queue_name": f"invalid.{uuid4().hex}",
            "task_retry_delay_seconds": 0.01,
        }
    )
    model = Mock(name="test", version="1")
    monkeypatch.setattr(worker_module, "load_registered_model", lambda _: model)
    worker = worker_module.create_worker_app(settings)
    async with (
        create_broker(settings) as broker,
        worker.router.lifespan_context(worker),
    ):
        await broker.publish(
            b"{invalid json",
            content_type="application/json",
            message_id="invalid-json",
            exchange=task_exchange(settings),
            routing_key=settings.task_queue_name,
            persist=True,
        )
        queue = await broker.declare_queue(
            RabbitQueue(f"{settings.task_queue_name}.dlq", durable=True)
        )
        dead = await wait_until(lambda: queue.get(fail=False))
        assert dead.body == b"{invalid json"
        assert dead.headers["x-task-attempts"] == 3
        await dead.ack()
        model.predict.assert_not_called()


async def test_requeue_while_queue_is_full_preserves_other_tasks(services, monkeypatch):
    from faststream.rabbit import RabbitQueue

    from cfpb_complaint_classifier.db.tasks import claim_attempt, task_attempts

    settings = services.model_copy(
        update={
            "task_queue_name": f"full-retry.{uuid4().hex}",
            "task_retry_delay_seconds": 0.5,
            "inference_delay_seconds": 0,
        }
    )
    engine = create_engine(settings)
    await create_schema(engine)

    def predict(texts):
        if texts[0] == "poison":
            raise ValueError("poison")
        return ["Debt"]

    model = LoadedModel(Mock(predict=Mock(side_effect=predict)), "test", "1")
    monkeypatch.setattr(worker_module, "load_registered_model", lambda _: model)
    poison = TaskMessage(task_id=uuid4(), text="poison")
    others = [TaskMessage(task_id=uuid4(), text=f"valid {index}") for index in range(2)]
    worker = worker_module.create_worker_app(settings)
    try:
        async with (
            create_broker(settings) as broker,
            worker.router.lifespan_context(worker),
        ):
            await publish_task(broker, settings, poison)

            async def first_attempt_started():
                async with engine.connect() as conn:
                    return await conn.scalar(
                        select(task_attempts.c.attempts).where(
                            task_attempts.c.message_key == str(poison.task_id)
                        )
                    )

            await wait_until(first_attempt_started)
            for task in others:
                await publish_task(broker, settings, task)
            main_queue = await broker.declare_queue(task_queue(settings))
            await main_queue.declare()
            assert main_queue.declaration_result.message_count == 2
            dead_queue = await broker.declare_queue(
                RabbitQueue(f"{settings.task_queue_name}.dlq", durable=True)
            )
            dead = await wait_until(lambda: dead_queue.get(fail=False))
            assert dead.message_id == str(poison.task_id)
            assert dead.headers["x-task-attempts"] == 3
            await dead.ack()

            async def completed():
                return all([await get_result(engine, task.task_id) for task in others])

            await wait_until(completed)
            assert await claim_attempt(engine, str(poison.task_id), 3) is None
    finally:
        await engine.dispose()


async def test_postgres_attempt_claims_and_results_are_atomic(services):
    from datetime import UTC, datetime

    from cfpb_complaint_classifier.db.tasks import claim_attempt
    from cfpb_complaint_classifier.schemas import TaskResult

    engine = create_engine(services)
    await create_schema(engine)
    try:
        key = str(uuid4())
        claims = await asyncio.gather(
            *(claim_attempt(engine, key, 3) for _ in range(12))
        )
        assert sorted(c for c in claims if c is not None) == [1, 2, 3]
        now = datetime.now(UTC)
        result = TaskResult(
            task_id=uuid4(),
            category="Debt",
            model_name="test",
            model_version="1",
            created_at=now,
            started_at=now,
            completed_at=now,
        )
        await asyncio.gather(*(save_result(engine, result) for _ in range(12)))
        async with engine.connect() as conn:
            assert (
                await conn.scalar(
                    select(func.count())
                    .select_from(task_results)
                    .where(task_results.c.task_id == result.task_id)
                )
                == 1
            )
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "signal_name,after_commit",
    [("SIGTERM", False), ("SIGKILL", False), ("SIGKILL", True)],
)
async def test_process_shutdown_and_redelivery(
    services, tmp_path, signal_name, after_commit
):
    import signal
    import socket
    import sys

    import httpx

    settings = services.model_copy(
        update={
            "task_queue_name": f"signal.{uuid4().hex}",
            "inference_delay_seconds": 0.7,
            "worker_concurrency": 2,
            "shutdown_timeout_seconds": 5,
        }
    )
    engine = create_engine(settings)
    await create_schema(engine)
    marker = tmp_path / "committed"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        **{
            key.upper(): str(value)
            for key, value in settings.model_dump().items()
            if value is not None
        },
        "TEST_PORT": str(port),
        "PYTHONPATH": "src",
        "LOG_LEVEL": "WARNING",
    }
    if after_commit:
        env["TEST_PAUSE_AFTER_COMMIT"] = str(marker)
    log = (tmp_path / "worker.log").open("w")
    process = None
    async with httpx.AsyncClient() as client, create_broker(settings) as broker:
        try:
            await declare_topology(broker, settings)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "tests/worker_process.py",
                env=env,
                stdout=log,
                stderr=log,
            )

            async def healthy():
                try:
                    return (
                        await client.get(
                            f"http://127.0.0.1:{port}/healthz", timeout=0.2
                        )
                    ).status_code == 200
                except httpx.HTTPError:
                    return False

            await wait_until(healthy, timeout=20)
            task = TaskMessage(task_id=uuid4(), text="debt calls")
            await publish_task(broker, settings, task)
            companion = None
            if signal_name == "SIGTERM":
                companion = TaskMessage(task_id=uuid4(), text="second accepted task")
                await publish_task(broker, settings, companion)

            async def started(task_id=task.task_id):
                from cfpb_complaint_classifier.db.tasks import task_attempts

                async with engine.connect() as conn:
                    return await conn.scalar(
                        select(task_attempts.c.attempts).where(
                            task_attempts.c.message_key == str(task_id)
                        )
                    )

            await wait_until(started)
            if companion is not None:
                await wait_until(lambda: started(companion.task_id))
            if after_commit:

                async def committed():
                    return marker.exists()

                await wait_until(committed)

            assert await healthy()
            process.send_signal(getattr(signal, signal_name))
            await asyncio.wait_for(process.wait(), timeout=12)
            if signal_name == "SIGTERM":
                assert process.returncode in (0, -signal.SIGTERM)
                assert await get_result(engine, task.task_id) is not None
                assert await get_result(engine, companion.task_id) is not None
            else:
                env.pop("TEST_PAUSE_AFTER_COMMIT", None)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "tests/worker_process.py",
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                await wait_until(healthy, timeout=20)

                async def completed():
                    return await get_result(engine, task.task_id)

                await wait_until(completed)
                await asyncio.sleep(0.2)
                assert await started() == (1 if after_commit else 2)
            async with engine.connect() as conn:
                assert (
                    await conn.scalar(
                        select(func.count())
                        .select_from(task_results)
                        .where(task_results.c.task_id == task.task_id)
                    )
                    == 1
                )
        finally:
            if process and process.returncode is None:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=12)
            log.close()
            await engine.dispose()


async def test_two_workers_handle_concurrent_duplicate(services, monkeypatch):
    settings = services.model_copy(
        update={
            "task_queue_name": f"duplicates.{uuid4().hex}",
            "inference_delay_seconds": 0.15,
        }
    )
    engine = create_engine(settings)
    await create_schema(engine)
    model = LoadedModel(Mock(predict=Mock(return_value=["Debt"])), "test", "1")
    monkeypatch.setattr(worker_module, "load_registered_model", lambda _: model)
    first = worker_module.create_worker_app(settings)
    second = worker_module.create_worker_app(settings)
    task = TaskMessage(task_id=uuid4(), text="debt calls")
    try:
        async with (
            create_broker(settings) as broker,
            first.router.lifespan_context(first),
            second.router.lifespan_context(second),
        ):
            await publish_task(broker, settings, task)
            await publish_task(broker, settings, task)

            async def both_finished():
                return model.pipeline.predict.call_count == 2 and await get_result(
                    engine, task.task_id
                )

            await wait_until(both_finished)
        async with engine.connect() as conn:
            assert (
                await conn.scalar(
                    select(func.count())
                    .select_from(task_results)
                    .where(task_results.c.task_id == task.task_id)
                )
                == 1
            )
    finally:
        await engine.dispose()
