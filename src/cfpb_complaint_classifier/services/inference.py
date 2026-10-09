"""Resolve the deployment alias once, then load an immutable model version."""

import logging
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from cfpb_complaint_classifier.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    name: str
    version: str

    def predict(self, text: str) -> str:
        predictions = self.pipeline.predict([text])
        if len(predictions) != 1 or not isinstance(predictions[0], str):
            raise ValueError("Model must return exactly one category string")
        category = str(predictions[0]).strip()
        if not category:
            raise ValueError("Model returned an empty category")
        return category


def resolve_model_version(settings: Settings) -> str:
    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_tracking_uri,
    )
    version = client.get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )
    if version.status != "READY":
        raise RuntimeError("Registered model version is not READY")
    return str(version.version)


def load_registered_model(settings: Settings) -> LoadedModel:
    # The host and containers both download through the tracking server;
    # a presigned MinIO URL could reference a hostname unavailable to the host.
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    try:
        version = resolve_model_version(settings)
        model_uri = f"models:/{settings.mlflow_model_name}/{version}"
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        logger.error("Model startup failed: %s", type(exc).__name__)
        raise RuntimeError(
            "Cannot load the registered model. Check MLflow and register the "
            "configured model alias before starting the application."
        ) from None
    logger.info("Loaded model %s version %s", settings.mlflow_model_name, version)
    return LoadedModel(
        pipeline=pipeline, name=settings.mlflow_model_name, version=version
    )


if __name__ == "__main__":
    # Compose preflight: a healthy server alone does not imply a registered model.
    settings = get_settings()
    version = resolve_model_version(settings)
    print(f"{settings.mlflow_model_name}@{settings.mlflow_model_alias} -> {version}")
