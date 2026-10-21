from hashlib import sha256

from aio_pika.exceptions import AMQPException, DeliveryError
from faststream.rabbit import (
    Channel,
    RabbitBroker,
    RabbitExchange,
    RabbitMessage,
    RabbitQueue,
)
from pamqp.commands import Basic

from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.schemas import TaskMessage


class QueueOverloadedError(Exception):
    pass


class BrokerUnavailableError(Exception):
    pass


def task_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(settings.task_queue_name, durable=True)


def task_queue(settings: Settings) -> RabbitQueue:
    arguments = {
        "x-dead-letter-exchange": f"{settings.task_queue_name}.dlx",
        "x-dead-letter-routing-key": f"{settings.task_queue_name}.dlq",
    }
    if settings.backpressure_enabled:
        arguments.update(
            {"x-max-length": settings.queue_max_length, "x-overflow": "reject-publish"}
        )
    return RabbitQueue(
        settings.task_queue_name,
        durable=True,
        routing_key=settings.task_queue_name,
        arguments=arguments,
    )


def create_broker(settings: Settings, *, consumer: bool = False) -> RabbitBroker:
    return RabbitBroker(
        settings.rabbitmq_url,
        timeout=settings.publish_timeout_seconds,
        graceful_timeout=settings.shutdown_timeout_seconds,
        default_channel=Channel(
            prefetch_count=settings.effective_prefetch_count if consumer else None,
            publisher_confirms=True,
            on_return_raises=True,
        ),
    )


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    dead_exchange = await broker.declare_exchange(
        RabbitExchange(f"{settings.task_queue_name}.dlx", durable=True)
    )
    dead_queue = await broker.declare_queue(
        RabbitQueue(f"{settings.task_queue_name}.dlq", durable=True)
    )
    await dead_queue.bind(dead_exchange, routing_key=dead_queue.name)
    exchange = await broker.declare_exchange(task_exchange(settings))
    queue = await broker.declare_queue(task_queue(settings))
    await queue.bind(exchange, routing_key=settings.task_queue_name)


async def publish_task(
    broker: RabbitBroker, settings: Settings, task: TaskMessage
) -> None:
    try:
        confirmation = await broker.publish(
            task.model_dump(mode="json"),
            exchange=task_exchange(settings),
            routing_key=settings.task_queue_name,
            message_id=str(task.task_id),
            persist=True,
            mandatory=True,
            timeout=settings.publish_timeout_seconds,
        )
    except DeliveryError as exc:
        if isinstance(exc.frame, Basic.Nack):
            raise QueueOverloadedError from None
        raise BrokerUnavailableError from None
    except (AMQPException, OSError, TimeoutError):
        raise BrokerUnavailableError from None
    if isinstance(confirmation, (Basic.Nack, Basic.Reject)):
        raise QueueOverloadedError
    if not isinstance(confirmation, Basic.Ack):
        raise BrokerUnavailableError


async def publish_dead_letter(
    broker: RabbitBroker,
    settings: Settings,
    message: RabbitMessage,
    *,
    attempts: int,
    error: str,
) -> None:

    confirmation = await broker.publish(
        message.body,
        exchange=f"{settings.task_queue_name}.dlx",
        routing_key=f"{settings.task_queue_name}.dlq",
        persist=True,
        mandatory=True,
        message_id=message.raw_message.message_id or sha256(message.body).hexdigest(),
        content_type=message.content_type,
        headers={
            **message.headers,
            "x-task-attempts": attempts,
            "x-task-error": error,
        },
        timeout=settings.publish_timeout_seconds,
    )
    if not isinstance(confirmation, Basic.Ack):
        raise BrokerUnavailableError
