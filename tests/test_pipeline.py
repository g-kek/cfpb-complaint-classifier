from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from pamqp.commands import Basic
from pydantic import ValidationError

from cfpb_complaint_classifier import source
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.messaging import (
    BrokerUnavailableError,
    QueueOverloadedError,
    declare_topology,
    publish_task,
    task_queue,
)
from cfpb_complaint_classifier.schemas import TaskMessage


async def test_publish_requires_confirmation_and_persistent_routing():
    broker = Mock(publish=AsyncMock(return_value=Basic.Ack()))
    settings = Settings()
    task = TaskMessage(task_id=uuid4(), text="debt")
    await publish_task(broker, settings, task)
    kwargs = broker.publish.call_args.kwargs
    assert kwargs["persist"] is True
    assert kwargs["mandatory"] is True
    assert kwargs["message_id"] == str(task.task_id)
    assert kwargs["routing_key"] == settings.task_queue_name
    assert kwargs["timeout"] == settings.publish_timeout_seconds


@pytest.mark.parametrize("frame", [Basic.Nack(), Basic.Reject()])
async def test_negative_confirmation_signals_overload(frame):
    broker = Mock(publish=AsyncMock(return_value=frame))
    with pytest.raises(QueueOverloadedError):
        await publish_task(
            broker, Settings(), TaskMessage(task_id=uuid4(), text="debt")
        )


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (DeliveryError(None, Basic.Nack()), QueueOverloadedError),
        (DeliveryError(None, Basic.Return()), BrokerUnavailableError),
        (TimeoutError(), BrokerUnavailableError),
        (ConnectionError(), BrokerUnavailableError),
    ],
)
async def test_publication_errors_are_classified(failure, expected):
    broker = Mock(publish=AsyncMock(side_effect=failure))
    with pytest.raises(expected):
        await publish_task(
            broker, Settings(), TaskMessage(task_id=uuid4(), text="debt")
        )


async def test_missing_confirmation_is_not_success():
    broker = Mock(publish=AsyncMock(return_value=None))
    with pytest.raises(BrokerUnavailableError):
        await publish_task(
            broker, Settings(), TaskMessage(task_id=uuid4(), text="debt")
        )


async def test_topology_binds_main_queue_and_dlq():
    queue = Mock(name="queue", bind=AsyncMock())
    queue.name = "test.dlq"
    broker = Mock(
        declare_exchange=AsyncMock(), declare_queue=AsyncMock(return_value=queue)
    )
    await declare_topology(broker, Settings())
    assert broker.declare_queue.await_count == 2
    assert queue.bind.await_count == 2


def test_limits_and_unbounded_comparison_mode():
    settings = Settings(worker_concurrency=3, inference_delay_seconds=9)
    assert settings.effective_prefetch_count == 3
    assert settings.retry_after_seconds == 3
    queue = task_queue(settings)
    assert queue.arguments["x-max-length"] == 20
    assert queue.arguments["x-overflow"] == "reject-publish"
    settings = Settings(backpressure_enabled=False)
    assert settings.effective_prefetch_count == 0
    assert "x-max-length" not in task_queue(settings).arguments


@pytest.mark.parametrize(
    "values",
    [
        {"worker_concurrency": 1, "prefetch_count": 2},
        {"queue_max_length": 0},
        {"source_retry_initial_seconds": 2, "source_retry_max_seconds": 1},
        {"inference_delay_seconds": float("inf")},
    ],
)
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValidationError):
        Settings(**values)


def test_broker_message_requires_stable_task_id():
    with pytest.raises(ValidationError, match="task_id"):
        TaskMessage.model_validate({"text": "debt"})


async def test_source_retries_same_task_before_reading_next(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    path.write_text("text\ndebt calls\nmortgage payments\n")
    tasks_read = []
    read_tasks = source.read_tasks

    def read(path, batch_id):
        for task in read_tasks(path, batch_id):
            tasks_read.append(task.task_id)
            yield task

    attempts = []

    async def publish(broker, settings, task):
        attempts.append(task.task_id)
        if len(attempts) < 4:
            assert len(tasks_read) == 1
            raise QueueOverloadedError

    broker = Mock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock()
    monkeypatch.setattr(source, "create_broker", lambda settings: broker)
    monkeypatch.setattr(source, "declare_topology", AsyncMock())
    monkeypatch.setattr(source, "read_tasks", read)
    monkeypatch.setattr(source, "publish_task", publish)
    sleep = AsyncMock()
    monkeypatch.setattr(source.asyncio, "sleep", sleep)
    settings = Settings(source_retry_initial_seconds=0.1, source_retry_max_seconds=0.2)
    metrics = Mock()
    assert await source.publish_file(path, "batch-1", settings, metrics) == 2
    assert [call.kwargs["accepted"] for call in metrics.record.call_args_list] == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert attempts[:4] == [tasks_read[0]] * 4
    assert attempts[4] == tasks_read[1]
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2, 0.2]
    assert [task.task_id for task in read_tasks(path, "batch-1")] == tasks_read


def test_source_rejects_missing_text_column(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("product\nmortgage\n")
    with pytest.raises(ValueError, match="text column"):
        list(source.read_tasks(path, "batch"))


def test_source_metrics_separates_accepted_and_rejected(tmp_path):
    import csv

    from cfpb_complaint_classifier.metrics import SourceMetrics

    path = tmp_path / "events.csv"
    metrics = SourceMetrics(path)
    metrics.record("same-id", accepted=False)
    metrics.record("same-id", accepted=True)
    metrics.close()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["event"] for row in rows] == ["rejected", "accepted"]
    assert rows[-1]["accepted"] == "1"
    assert rows[-1]["rejected"] == "1"
    assert rows[0]["task_id"] == rows[1]["task_id"]
    with pytest.raises(FileExistsError):
        SourceMetrics(path)
    no_file = SourceMetrics()
    no_file.record("id", accepted=True)
    no_file.close()
    assert no_file.accepted == 1
