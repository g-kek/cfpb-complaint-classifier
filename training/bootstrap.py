"""Prepare the previous practical's model only when starting an empty registry."""

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from training.tracking import EXPERIMENT_NAME, MODEL_NAME, configure_tracking


def main():
    model = os.environ.get("MLFLOW_MODEL_NAME", MODEL_NAME)
    alias = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")
    client = MlflowClient()
    try:
        client.get_registered_model(model)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        # An existing registry is authoritative, including a missing alias.
        version = client.get_model_version_by_alias(model, alias)
        if version.status != "READY":
            raise RuntimeError("Registered model version is not READY")
        return

    from training import prepare
    from training.eda import run_eda
    from training.train import train_models

    directory = Path("data/cfpb")
    if not (directory / "manifest.json").is_file():
        prepare.main()
    configure_tracking(None, EXPERIMENT_NAME)
    with TemporaryDirectory() as work:
        output = Path(work)
        eda_run = run_eda(directory, output / "eda")
        train_models(directory, output / "training", model, alias, eda_run)
    logging.getLogger(__name__).info("Model registered")


if __name__ == "__main__":
    main()
