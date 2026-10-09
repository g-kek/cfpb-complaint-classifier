import threading
from unittest.mock import AsyncMock, Mock

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from cfpb_complaint_classifier import app as app_module
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.services import inference
from cfpb_complaint_classifier.services.inference import LoadedModel


async def test_process_returns_category_and_loaded_version(client):
    response = await client.post(
        "/process", json={"text": "A collector keeps calling."}
    )
    assert response.status_code == 200
    assert response.json() == {
        "category": "Debt collection",
        "model_name": "cfpb-complaint-classifier",
        "model_version": "7",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": " \n\t "},
        {"text": None},
        {"text": 42},
        {"text": ["debt"]},
        {"text": "x" * 20001},
        {"text": "debt", "extra": True},
    ],
)
async def test_process_validates_text(client, payload):
    assert (await client.post("/process", json=payload)).status_code == 422


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
    monkeypatch.setattr(app_module, "load_registered_model", loader)
    test_app = app_module.create_app()
    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client,
    ):
        for _ in range(3):
            response = await client.post("/process", json={"text": "  debt\n calls  "})
            assert response.json()["model_version"] == "12"
    loader.assert_called_once()
    assert texts_seen == [["debt calls"]] * 3
    assert all(thread != event_loop_thread for thread in worker_threads)


async def test_startup_failure_closes_engine(monkeypatch):
    engine = Mock(dispose=AsyncMock())
    loader = Mock(side_effect=RuntimeError("Cannot load the registered model"))
    monkeypatch.setattr(app_module, "create_engine", lambda settings: engine)
    monkeypatch.setattr(app_module, "load_registered_model", loader)
    test_app = app_module.create_app()
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


async def test_inference_failure_returns_503(monkeypatch, client, caplog):
    monkeypatch.setattr(
        LoadedModel, "predict", Mock(side_effect=ValueError("private model details"))
    )
    response = await client.post("/process", json={"text": "debt calls"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Model inference is unavailable"}
    assert "private model details" not in caplog.text


async def test_real_registry_alias_changes_only_after_restart(tmp_path, monkeypatch):
    """Use real MLflow versions and artifacts across two application lifespans."""
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
        monkeypatch.setattr(app_module, "create_engine", lambda settings: engine)
        loader = Mock(wraps=inference.load_registered_model)
        monkeypatch.setattr(app_module, "load_registered_model", loader)
        registry.set_registered_model_alias("restart-test", "champion", versions[0])
        first_app = app_module.create_app()
        async with (
            first_app.router.lifespan_context(first_app),
            AsyncClient(
                transport=ASGITransport(app=first_app), base_url="http://test"
            ) as client,
        ):
            first = await client.post("/process", json={"text": "debt calls"})
            assert first.json()["category"] == "Debt collection"
            assert first.json()["model_version"] == versions[0]
            registry.set_registered_model_alias("restart-test", "champion", versions[1])
            unchanged = await client.post("/process", json={"text": "debt calls"})
            assert unchanged.json() == first.json()
        second_app = app_module.create_app()
        async with (
            second_app.router.lifespan_context(second_app),
            AsyncClient(
                transport=ASGITransport(app=second_app), base_url="http://test"
            ) as client,
        ):
            second = await client.post("/process", json={"text": "debt calls"})
            assert second.json()["category"] == "Mortgage"
            assert second.json()["model_version"] == versions[1]
        assert loader.call_count == 2
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
