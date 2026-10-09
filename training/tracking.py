import os
from pathlib import Path

import mlflow
import pandas as pd

from training.data import file_digest

EXPERIMENT_NAME = "cfpb-product-classification"
MODEL_NAME = "cfpb-complaint-classifier"


def configure_tracking(tracking_uri: str | None, experiment: str) -> None:
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(experiment)


def log_dataset(frame: pd.DataFrame, manifest: dict, context: str) -> None:
    dataset = mlflow.data.from_pandas(
        frame[["complaint_id", "text", "product"]],
        source=manifest["source"],
        targets="product",
        name=f"cfpb-{'train' if context == 'training' else context}",
    )
    mlflow.log_input(dataset, context=context)


def log_provenance(directory: Path, manifest: dict) -> None:
    mlflow.set_tags(
        {
            "dataset.source": manifest["source"],
            "dataset.raw_sha256": manifest["files"]["raw.csv"],
            "dataset.manifest_sha256": file_digest(directory / "manifest.json"),
        }
    )
    mlflow.log_params(
        {
            key: manifest[key]
            for key in ("seed", "min_length", "min_class_count", "max_samples")
        }
    )
    mlflow.log_artifact(str(directory / "manifest.json"), "data")
    root = Path(__file__).resolve().parent.parent
    for source in sorted((root / "training").glob("*.py")):
        mlflow.log_artifact(str(source), "code/training")
    mlflow.log_artifact(str(root / "uv.lock"), "code")
    mlflow.log_artifact(str(root / "pyproject.toml"), "code")
