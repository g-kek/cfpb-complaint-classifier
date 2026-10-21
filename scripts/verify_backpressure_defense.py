import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from faststream.rabbit import RabbitQueue
from run_backpressure_experiments import compose, docker_client
from sqlalchemy import select

from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.db.tasks import get_result, task_attempts
from cfpb_complaint_classifier.messaging import (
    create_broker,
    declare_topology,
    publish_task,
    task_exchange,
)
from cfpb_complaint_classifier.schemas import TaskMessage


async def wait_until(check, seconds=90):
    async with asyncio.timeout(seconds):
        while not (result := await check()):
            await asyncio.sleep(0.025)
        return result


def host_rabbitmq_url():
    return (
        Settings().rabbitmq_url
        if Path(".env").exists() or os.getenv("RABBITMQ_URL")
        else "amqp://cfpb:cfpb-local-password@127.0.0.1:5672/"
    )


async def main():
    queue_name = f"defense.{uuid4().hex[:12]}"
    env = {
        **os.environ,
        "TASK_QUEUE_NAME": queue_name,
        "QUEUE_MAX_LENGTH": "3",
        "BACKPRESSURE_ENABLED": "true",
        "PREFETCH_COUNT": "1",
        "WORKER_CONCURRENCY": "2",
        "INFERENCE_DELAY_SECONDS": "1",
    }
    settings = Settings(
        rabbitmq_url=host_rabbitmq_url(),
        task_queue_name=queue_name,
        queue_max_length=3,
        prefetch_count=1,
        worker_concurrency=2,
        inference_delay_seconds=1,
    )
    engine = create_engine(settings)
    client = docker_client()
    report = {"queue": queue_name, "queue_limit": 3, "prefetch": 1, "concurrency": 2}

    async def command(*args):
        await asyncio.to_thread(compose, *args, env=env)

    async def started(task):
        async with engine.connect() as conn:
            return await conn.scalar(
                select(task_attempts.c.attempts).where(
                    task_attempts.c.message_key == str(task.task_id)
                )
            )

    try:
        await command(
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--scale",
            "worker=1",
            "app",
            "worker",
        )
        await command("stop", "worker")
        async with create_broker(settings) as broker:
            await declare_topology(broker, settings)
            tasks = [
                TaskMessage(
                    task_id=uuid4(),
                    text="My credit card payment was applied incorrectly.",
                )
                for _ in range(3)
            ]
            for task in tasks:
                await publish_task(broker, settings, task)
            import httpx

            async with httpx.AsyncClient() as http:

                async def healthy():
                    try:
                        return (
                            await http.get("http://127.0.0.1:8080/healthz", timeout=0.5)
                        ).status_code == 200
                    except httpx.HTTPError:
                        return False

                await wait_until(healthy)
                before = time.monotonic()
                refused_id = uuid4()
                response = await http.post(
                    "http://127.0.0.1:8080/api/v1/tasks",
                    json={"task_id": str(refused_id), "text": "debt"},
                )
                assert (
                    response.status_code == 503
                    and response.headers["Retry-After"] == "1"
                )
                assert await get_result(engine, refused_id) is None
                report["overflow"] = {
                    "status": response.status_code,
                    "retry_after": 1,
                    "response_seconds": time.monotonic() - before,
                }
            await command("up", "-d", "--no-deps", "worker")
            await wait_until(lambda: started(tasks[0]))
            await command("stop", "-t", "40", "worker")
            assert await get_result(engine, tasks[0].task_id)
            report["sigterm_completed"] = str(tasks[0].task_id)
            kill_task = TaskMessage(
                task_id=uuid4(), text="My mortgage payment was not applied."
            )
            await publish_task(broker, settings, kill_task)
            await command("up", "-d", "--no-deps", "worker")
            await wait_until(lambda: started(kill_task))
            await command("kill", "-s", "SIGKILL", "worker")
            await command("up", "-d", "--no-deps", "worker")
            for task in [*tasks, kill_task]:
                await wait_until(lambda task=task: get_result(engine, task.task_id))
            report["sigkill_redelivery_attempts"] = await started(kill_task)
            assert report["sigkill_redelivery_attempts"] == 2
            poison_id = str(uuid4())
            await broker.publish(
                b"{invalid json",
                content_type="application/json",
                message_id=poison_id,
                exchange=task_exchange(settings),
                routing_key=queue_name,
                persist=True,
            )
            dead_queue = await broker.declare_queue(
                RabbitQueue(f"{queue_name}.dlq", durable=True)
            )
            dead = await wait_until(lambda: dead_queue.get(fail=False))
            assert dead.message_id == poison_id and dead.headers["x-task-attempts"] == 3
            report["dlq"] = {
                "message_id": dead.message_id,
                "attempts": 3,
                "body_preserved": dead.body == b"{invalid json",
            }
            await dead.ack()
            print(json.dumps(report, indent=2), flush=True)
    finally:
        await engine.dispose()
        client.close()
        await asyncio.to_thread(
            compose,
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--scale",
            "worker=1",
            "app",
            "worker",
            env=os.environ.copy(),
        )


if __name__ == "__main__":
    asyncio.run(main())
