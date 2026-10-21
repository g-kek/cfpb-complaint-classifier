import asyncio
from collections import defaultdict
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from faststream.rabbit import TestRabbitBroker
from faststream.rabbit.message import RabbitMessage as Message
from pamqp.commands import Basic

from cfpb_complaint_classifier import worker
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.messaging import (
    BrokerUnavailableError,
    publish_dead_letter,
    task_exchange,
    task_queue,
)
from cfpb_complaint_classifier.schemas import TaskMessage


@pytest.fixture
async def harness(monkeypatch):
    settings = Settings(inference_delay_seconds=0, task_retry_delay_seconds=0.001)
    model = Mock(name="model", version="1", predict=Mock(return_value="Debt"))
    model.name = "test"
    attempts = defaultdict(int)
    events = []

    async def claim(engine, key, limit):
        if attempts[key] >= limit:
            return None
        attempts[key] += 1
        return attempts[key]

    async def save(*args):
        events.append("commit")

    async def ack(self, *args, **kwargs):
        events.append("ack")

    monkeypatch.setattr(worker, "create_engine", lambda _: Mock(dispose=AsyncMock()))
    monkeypatch.setattr(worker, "load_registered_model", lambda _: model)
    monkeypatch.setattr(worker, "declare_topology", AsyncMock())
    monkeypatch.setattr(worker, "claim_attempt", claim)
    monkeypatch.setattr(worker, "get_result", AsyncMock(return_value=None))
    saved = AsyncMock(side_effect=save)
    monkeypatch.setattr(worker, "save_result", saved)
    monkeypatch.setattr(Message, "ack", ack)
    nack = AsyncMock()
    monkeypatch.setattr(Message, "nack", nack)
    dead = AsyncMock(side_effect=lambda *a, **k: events.append("dlq-confirm"))
    monkeypatch.setattr(worker, "publish_dead_letter", dead)
    app = worker.create_worker_app(settings)
    async with (
        TestRabbitBroker(app.state.broker) as broker,
        app.router.lifespan_context(app),
    ):

        async def send(body=None, message_id="stable-id"):
            if body is None:
                body = TaskMessage(task_id=uuid4(), text="debt").model_dump(mode="json")
            await broker.publish(
                body,
                queue=task_queue(settings),
                exchange=task_exchange(settings),
                message_id=message_id,
            )

        yield Mock(
            send=send,
            model=model,
            events=events,
            attempts=attempts,
            dead=dead,
            nack=nack,
            saved=saved,
            app=app,
        )


async def test_commit_precedes_ack(harness):
    await harness.send()
    assert harness.events == ["commit", "ack"]
    assert harness.attempts["stable-id"] == 1


async def test_transient_error_retries_and_then_commits(harness):
    harness.model.predict.side_effect = [RuntimeError(), "Debt"]
    await harness.send()
    harness.nack.assert_awaited_once_with(requeue=True)
    await harness.send()
    assert harness.model.predict.call_count == 2
    assert harness.events == ["commit", "ack"]
    harness.dead.assert_not_awaited()


async def test_poison_message_exhausts_budget_then_confirms_dlq_before_ack(harness):
    harness.model.predict.side_effect = ValueError("poison")
    for _ in range(3):
        await harness.send()
    assert harness.model.predict.call_count == 3
    assert harness.nack.await_count == 2
    assert harness.events == ["dlq-confirm", "ack"]
    assert harness.dead.call_args.kwargs == {"attempts": 3, "error": "ValueError"}
    harness.saved.assert_not_awaited()


@pytest.mark.parametrize("body", [{"text": "missing id"}, b"{bad json", {"text": " "}])
async def test_malformed_payload_uses_bounded_budget_and_dlq(harness, body):
    for _ in range(3):
        await harness.send(body)
    assert harness.attempts["stable-id"] == 3
    assert harness.nack.await_count == 2
    assert harness.events == ["dlq-confirm", "ack"]
    harness.model.predict.assert_not_called()


async def test_dlq_failure_preserves_original_without_resetting_budget(harness):
    harness.model.predict.side_effect = ValueError()
    harness.dead.side_effect = BrokerUnavailableError()
    for _ in range(3):
        await harness.send()
    assert harness.nack.await_count == 3
    assert "ack" not in harness.events
    harness.dead.side_effect = lambda *a, **k: harness.events.append("dlq-confirm")
    await harness.send()
    assert harness.model.predict.call_count == 3
    assert harness.events == ["dlq-confirm", "ack"]


async def test_storage_outage_retries_same_result_without_dlq(harness):
    failures = 0

    async def save(*args):
        nonlocal failures
        failures += 1
        if failures < 3:
            raise OSError("database unavailable")
        harness.events.append("commit")

    harness.saved.side_effect = save
    await harness.send()
    assert harness.saved.await_count == 3
    assert harness.model.predict.call_count == 1
    assert harness.attempts["stable-id"] == 1
    assert harness.events == ["commit", "ack"]
    harness.dead.assert_not_awaited()
    harness.nack.assert_not_awaited()


async def test_committed_redelivery_skips_inference_and_budget(harness, monkeypatch):
    monkeypatch.setattr(worker, "get_result", AsyncMock(return_value=Mock()))
    await harness.send()
    assert harness.events == ["ack"]
    assert not harness.attempts
    harness.model.predict.assert_not_called()


async def test_storage_unavailable_requeues_without_ack(harness, monkeypatch):
    monkeypatch.setattr(worker, "claim_attempt", AsyncMock(side_effect=OSError()))
    await harness.send()
    assert harness.events == []
    harness.nack.assert_awaited_once_with(requeue=True)


async def test_cancellation_leaves_message_unacknowledged(harness):
    harness.saved.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await harness.send()
    assert "ack" not in harness.events
    harness.dead.assert_not_awaited()


@pytest.mark.parametrize("confirmation", [Basic.Ack(), Basic.Nack(), None])
async def test_dlq_publication_requires_confirm_and_preserves_body(confirmation):
    broker = Mock(publish=AsyncMock(return_value=confirmation))
    message = Mock(
        body=b"original",
        headers={"custom": "value"},
        message_id="stable",
        raw_message=Mock(message_id="stable"),
        content_type="application/json",
    )
    if isinstance(confirmation, Basic.Ack):
        await publish_dead_letter(
            broker, Settings(), message, attempts=3, error="Error"
        )
    else:
        with pytest.raises(BrokerUnavailableError):
            await publish_dead_letter(
                broker, Settings(), message, attempts=3, error="Error"
            )
    args = broker.publish.call_args
    assert args.args == (b"original",)
    assert args.kwargs["mandatory"] and args.kwargs["persist"]
    assert args.kwargs["headers"]["x-task-attempts"] == 3
    assert args.kwargs["message_id"] == "stable"
