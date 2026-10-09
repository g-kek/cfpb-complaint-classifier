import pytest
from httpx import ASGITransport, AsyncClient

from cfpb_complaint_classifier import app as app_module
from cfpb_complaint_classifier.services.inference import LoadedModel


class FakePipeline:
    def predict(self, texts):
        return ["Debt collection"] * len(texts)


@pytest.fixture(autouse=True)
def stub_model_loading(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "load_registered_model",
        lambda settings: LoadedModel(FakePipeline(), "cfpb-complaint-classifier", "7"),
    )


class HealthyConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def scalar(self, query):
        return "16.4"


class HealthyEngine:
    def connect(self):
        return HealthyConnection()

    async def dispose(self):
        return None


class UnavailableConnection:
    async def __aenter__(self):
        raise OSError("connection refused")

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class UnavailableEngine:
    def connect(self):
        return UnavailableConnection()

    async def dispose(self):
        return None


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr(app_module, "create_engine", lambda settings: HealthyEngine())
    test_app = app_module.create_app()
    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as http_client:
            yield http_client


@pytest.fixture
async def unavailable_client(monkeypatch):
    monkeypatch.setattr(
        app_module, "create_engine", lambda settings: UnavailableEngine()
    )
    test_app = app_module.create_app()
    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as http_client:
            yield http_client
