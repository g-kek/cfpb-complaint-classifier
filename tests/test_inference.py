import asyncio
import threading
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from cfpb_complaint_classifier import worker as worker_module
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.messaging import task_exchange, task_queue
from cfpb_complaint_classifier.schemas import TaskMessage
from cfpb_complaint_classifier.services import inference
from cfpb_complaint_classifier.services.inference import LoadedModel


@pytest.fixture(autouse=True)
def stub_retry_storage(monkeypatch):
    monkeypatch.setattr(worker_module, "get_result", AsyncMock(return_value=None))
    monkeypatch.setattr(worker_module, "claim_attempt", AsyncMock(return_value=1))


async def test_startup_loads_once_and_offloads_inference(monkeypatch):
    event_loop_thread = threading.get_ident()
    worker_threads = []
    texts_seen = []

    def predict(texts):
        worker_threads.append(threading.get_ident())
        texts_seen.append(texts)
        return np.asarray(["Debt collection"])

    pipeline = Mock(predict=Mock(side_effect=predict))

    def load(settings):
        worker_threads.append(threading.get_ident())
        return LoadedModel(pipeline, settings.mlflow_model_name, "12")

    loader = Mock(side_effect=load)
    monkeypatch.setattr(worker_module, "load_registered_model", loader)
    test_app = worker_module.create_worker_app(Settings(inference_delay_seconds=0))
    saved = AsyncMock()
    monkeypatch.setattr(worker_module, "save_result", saved)
    monkeypatch.setattr(worker_module, "declare_topology", AsyncMock())
    settings = Settings(inference_delay_seconds=0)
    async with (
        TestRabbitBroker(test_app.state.broker) as broker,
        test_app.router.lifespan_context(test_app),
    ):
        for _ in range(3):
            task = TaskMessage(task_id=uuid4(), text="  debt\n calls  ")
            await broker.publish(
                task.model_dump(mode="json"),
                queue=task_queue(settings),
                exchange=task_exchange(settings),
            )
    assert saved.await_count == 3
    assert saved.call_args.args[1].model_version == "12"
    loader.assert_called_once()
    assert texts_seen == [["debt calls"]] * 3
    assert all(thread != event_loop_thread for thread in worker_threads)


async def test_startup_failure_closes_engine(monkeypatch):
    engine = Mock(dispose=AsyncMock())
    loader = Mock(side_effect=RuntimeError("Cannot load the registered model"))
    monkeypatch.setattr(worker_module, "create_engine", lambda settings: engine)
    monkeypatch.setattr(worker_module, "load_registered_model", loader)
    test_app = worker_module.create_worker_app(Settings(inference_delay_seconds=0))
    with pytest.raises(RuntimeError, match="Cannot load the registered model"):
        async with test_app.router.lifespan_context(test_app):
            pytest.fail("Application must not start without a model")
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize("failure", ["alias", "download"])
def test_model_loading_errors_are_sanitized(monkeypatch, caplog, failure):
    error_uri = "http://user:private-password@tracking.example"
    client = Mock()
    client.get_model_version_by_alias.return_value = Mock(version="3", status="READY")
    if failure == "alias":
        client.get_model_version_by_alias.side_effect = MlflowException(error_uri)
    monkeypatch.setattr(inference, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(
        inference.mlflow.sklearn, "load_model", Mock(side_effect=OSError(error_uri))
    )
    with pytest.raises(RuntimeError, match="Cannot load the registered model") as error:
        inference.load_registered_model(Settings())
    assert "private-password" not in str(error.value)
    assert "private-password" not in caplog.text


def test_loading_pins_version_even_when_alias_moves(monkeypatch):
    client = Mock()
    alias = Mock(version="3", status="READY")
    client.get_model_version_by_alias.return_value = alias
    monkeypatch.setattr(inference, "MlflowClient", Mock(return_value=client))

    def download(uri):
        alias.version = "4"
        assert uri == "models:/cfpb-complaint-classifier/3"
        return Mock()

    load = Mock(side_effect=download)
    monkeypatch.setattr(inference.mlflow.sklearn, "load_model", load)
    model = inference.load_registered_model(Settings())
    assert model.version == "3"
    client.get_model_version_by_alias.assert_called_once_with(
        "cfpb-complaint-classifier", "champion"
    )
    load.assert_called_once()


def test_pending_registry_version_is_rejected(monkeypatch):
    client = Mock()
    client.get_model_version_by_alias.return_value = Mock(status="PENDING_REGISTRATION")
    monkeypatch.setattr(inference, "MlflowClient", Mock(return_value=client))
    with pytest.raises(RuntimeError, match="not READY"):
        inference.resolve_model_version(Settings())


@pytest.mark.parametrize("predictions", [[], ["a", "b"], [42], [" "]])
def test_invalid_model_output_is_rejected(predictions):
    pipeline = Mock(predict=Mock(return_value=predictions))
    with pytest.raises(ValueError, match="category"):
        LoadedModel(pipeline, "test", "1").predict("debt")


async def test_worker_health_responds_while_predict_is_busy(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def predict(texts):
        started.set()
        assert release.wait(timeout=5)
        return ["Debt collection"]

    model = LoadedModel(Mock(predict=predict), "test", "1")
    monkeypatch.setattr(worker_module, "load_registered_model", lambda settings: model)
    monkeypatch.setattr(worker_module, "declare_topology", AsyncMock())
    monkeypatch.setattr(worker_module, "save_result", AsyncMock())
    settings = Settings(inference_delay_seconds=0, worker_concurrency=1)
    app = worker_module.create_worker_app(settings)
    async with (
        TestRabbitBroker(app.state.broker) as broker,
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        processing = asyncio.create_task(
            broker.publish(
                TaskMessage(task_id=uuid4(), text="debt").model_dump(mode="json"),
                queue=task_queue(settings),
                exchange=task_exchange(settings),
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 2)
            response = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
            assert response.status_code == 200
            assert not processing.done()
        finally:
            release.set()
            await processing


async def test_real_registry_alias_changes_only_after_restart(tmp_path, monkeypatch):

    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        experiment = mlflow.create_experiment(
            "inference-restart", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        registry = MlflowClient(tracking_uri=uri, registry_uri=uri)
        versions = []
        for category in ("Debt collection", "Mortgage"):
            model = Pipeline(
                [
                    (
                        "classifier",
                        DummyClassifier(strategy="constant", constant=category),
                    )
                ]
            )
            model.fit(["debt", "mortgage"], ["Debt collection", "Mortgage"])
            with mlflow.start_run(experiment_id=experiment):
                logged = mlflow.sklearn.log_model(
                    model, name="model", pip_requirements=[]
                )
                versions.append(
                    str(mlflow.register_model(logged.model_uri, "restart-test").version)
                )
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "restart-test")
        monkeypatch.setenv("MLFLOW_MODEL_ALIAS", "champion")
        engine = Mock(dispose=AsyncMock())
        monkeypatch.setattr(worker_module, "create_engine", lambda settings: engine)
        loader = Mock(wraps=inference.load_registered_model)
        monkeypatch.setattr(worker_module, "load_registered_model", loader)
        registry.set_registered_model_alias("restart-test", "champion", versions[0])
        monkeypatch.setattr(worker_module, "declare_topology", AsyncMock())
        saved = AsyncMock()
        monkeypatch.setattr(worker_module, "save_result", saved)
        settings = Settings(inference_delay_seconds=0)
        first_app = worker_module.create_worker_app(settings)
        async with (
            TestRabbitBroker(first_app.state.broker) as broker,
            first_app.router.lifespan_context(first_app),
        ):

            async def publish():
                await broker.publish(
                    TaskMessage(task_id=uuid4(), text="debt calls").model_dump(
                        mode="json"
                    ),
                    queue=task_queue(settings),
                    exchange=task_exchange(settings),
                )
                return saved.call_args.args[1]

            first = await publish()
            assert first.category == "Debt collection"
            assert first.model_version == versions[0]
            registry.set_registered_model_alias("restart-test", "champion", versions[1])
            unchanged = await publish()
            assert unchanged.model_version == first.model_version
            assert unchanged.category == first.category
        second_app = worker_module.create_worker_app(settings)
        async with (
            TestRabbitBroker(second_app.state.broker) as broker,
            second_app.router.lifespan_context(second_app),
        ):
            second = await publish()
            assert second.category == "Mortgage"
            assert second.model_version == versions[1]
        assert loader.call_count == 2
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
