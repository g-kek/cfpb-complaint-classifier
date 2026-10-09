import argparse
import json
import os
import time
from importlib.metadata import version as package_version
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training.data import file_digest, load_data, write_json
from training.tracking import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    configure_tracking,
    log_dataset,
    log_provenance,
)

plt.switch_backend("Agg")


def model_variants(seed: int) -> dict[str, Pipeline]:
    return {
        "baseline": Pipeline(
            [("classifier", DummyClassifier(strategy="most_frequent"))]
        ),
        "tfidf-logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=30000, min_df=3, sublinear_tf=True),
                ),
                (
                    "classifier",
                    LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
                ),
            ]
        ),
        "tfidf-linear-svc": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=30000,
                        min_df=3,
                        sublinear_tf=True,
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
    }


def evaluate(
    model: Pipeline, frame: pd.DataFrame, labels: list[str], output: Path
) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    texts = frame["text"].tolist()
    targets = frame["product"]
    model.predict(texts[:10])
    started = time.perf_counter()
    predictions = model.predict(texts)
    elapsed = time.perf_counter() - started

    metrics = {
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "prediction_ms_per_sample": elapsed * 1000 / len(frame),
    }

    report = classification_report(
        targets, predictions, labels=labels, output_dict=True, zero_division=0
    )

    write_json(output / "classification_report.json", report)
    write_json(output / "metrics.json", metrics)
    fig, ax = plt.subplots(figsize=(13, 11))

    ConfusionMatrixDisplay.from_predictions(
        targets,
        predictions,
        labels=labels,
        normalize="true",
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=False,
    )

    ax.set_title("Confusion matrix (normalized by true class)")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    error_mask = targets.ne(predictions)
    errors = frame.loc[error_mask].copy()
    errors["prediction"] = np.asarray(predictions)[error_mask.to_numpy()]
    errors["text"] = errors["text"].str.slice(0, 1000)
    errors.head(100).to_csv(output / "errors.csv", index=False)

    return metrics


def log_parameters(model: Pipeline) -> None:
    parameters = model.get_params()
    selected = {
        name: parameters[name]
        for name in (
            "classifier__C",
            "classifier__class_weight",
            "classifier__max_iter",
            "classifier__strategy",
            "tfidf__max_features",
            "tfidf__min_df",
            "tfidf__ngram_range",
            "tfidf__sublinear_tf",
        )
        if name in parameters
    }
    selected["classifier"] = type(model.named_steps["classifier"]).__name__
    mlflow.log_params(selected)


def log_evaluation(
    model: Pipeline,
    frame: pd.DataFrame,
    labels: list[str],
    output: Path,
    split: str,
) -> dict[str, float]:
    metrics = evaluate(model, frame, labels, output)
    mlflow.log_metrics({f"{split}_{key}": value for key, value in metrics.items()})
    mlflow.log_artifacts(str(output), split)
    return metrics


def register_pipeline(
    model: Pipeline,
    example: list[str],
    model_name: str,
    tags: dict[str, str],
    client: MlflowClient,
) -> str:
    inputs = np.asarray(example)
    expected = model.predict(example)
    packages = (
        "mlflow",
        "scikit-learn",
        "numpy",
        "scipy",
        "pandas",
        "cloudpickle",
        "joblib",
    )
    logged = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(inputs, expected),
        input_example=inputs,
        pip_requirements=[
            f"{package}=={package_version(package)}" for package in packages
        ],
    )
    portable = mlflow.pyfunc.load_model(logged.model_uri)
    if not np.array_equal(portable.predict(inputs), expected):
        raise ValueError("Logged model failed the pyfunc check")

    version = mlflow.register_model(logged.model_uri, model_name)
    for key, value in {"export_status": "validated", **tags}.items():
        client.set_model_version_tag(model_name, version.version, key, value)
    return version.version


def verify_registered_model(
    model: Pipeline, model_uri: str, example: list[str]
) -> None:
    expected = model.predict(example)
    registered = mlflow.sklearn.load_model(model_uri)
    if not np.array_equal(registered.predict(example), expected):
        raise ValueError("Registered model predictions differ from the fitted pipeline")
    portable = mlflow.pyfunc.load_model(model_uri)
    if not np.array_equal(portable.predict(np.asarray(example)), expected):
        raise ValueError("Pyfunc predictions differ from the fitted pipeline")


def train_models(
    directory: Path, output: Path, model_name: str, alias: str, eda_run_id: str
) -> dict:
    frames, manifest = load_data(directory)
    labels = sorted(frames["train"]["product"].unique())
    client = MlflowClient()
    eda_run = client.get_run(eda_run_id)
    if eda_run.data.tags.get("dataset.manifest_sha256") != file_digest(
        directory / "manifest.json"
    ):
        raise ValueError("EDA run uses a different dataset manifest")
    results = []
    fitted = {}
    for name, model in model_variants(manifest["seed"]).items():
        with mlflow.start_run(run_name=name) as run:
            log_provenance(directory, manifest)
            mlflow.set_tags(
                {"eda.run_id": eda_run_id, "selection_metric": "validation_macro_f1"}
            )
            log_dataset(frames["train"], manifest, "training")
            log_dataset(frames["validation"], manifest, "validation")
            log_parameters(model)
            started = time.perf_counter()
            model.fit(frames["train"]["text"].tolist(), frames["train"]["product"])
            mlflow.log_metric("training_seconds", time.perf_counter() - started)
            diagnostic_path = output / name / "validation"
            metrics = log_evaluation(
                model, frames["validation"], labels, diagnostic_path, "validation"
            )
            result = {"name": name, "run_id": run.info.run_id, **metrics}
            if name != "baseline":
                example = frames["validation"]["text"].head(2).tolist()
                result["model_version"] = register_pipeline(
                    model,
                    example,
                    model_name,
                    {
                        "variant": name,
                        "dataset.raw_sha256": manifest["files"]["raw.csv"],
                    },
                    client,
                )
            results.append(result)
            fitted[name] = model
    candidates = [result for result in results if "model_version" in result]
    winner = max(candidates, key=lambda result: result["macro_f1"])
    winner_model = fitted[winner["name"]]
    with mlflow.start_run(run_id=winner["run_id"]):
        log_dataset(frames["test"], manifest, "test")
        test_path = output / winner["name"] / "test"
        test_metrics = log_evaluation(
            winner_model, frames["test"], labels, test_path, "test"
        )
        verify_registered_model(
            winner_model,
            f"models:/{model_name}/{winner['model_version']}",
            frames["test"]["text"].head(5).tolist(),
        )
        comparison = pd.DataFrame(results)
        comparison.to_csv(output / "comparison.csv", index=False)
        mlflow.log_artifact(str(output / "comparison.csv"), "comparison")
    client.set_registered_model_alias(model_name, alias, winner["model_version"])
    summary = {
        "model_name": model_name,
        "alias": alias,
        "model_version": winner["model_version"],
        "run_id": winner["run_id"],
        "variant": winner["name"],
        "validation_macro_f1": winner["macro_f1"],
        "test_metrics": test_metrics,
        "eda_run_id": eda_run_id,
        "runs": results,
    }
    write_json(output / "registry.json", summary)
    print(
        comparison[["name", "macro_f1", "accuracy", "model_version"]].to_string(
            index=False
        )
    )
    print(f"{model_name}@{alias} -> version {winner['model_version']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register CFPB classifiers")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cfpb"))
    parser.add_argument("--output", type=Path, default=Path("reports/training"))
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument(
        "--model-name", default=os.getenv("MLFLOW_MODEL_NAME", MODEL_NAME)
    )
    parser.add_argument("--alias", default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"))
    parser.add_argument("--eda-run-id")
    args = parser.parse_args()
    eda_run_id = args.eda_run_id
    if eda_run_id is None:
        eda_run_id = json.loads(Path("reports/eda/tracking.json").read_text())[
            "eda_run_id"
        ]
    configure_tracking(args.tracking_uri, args.experiment)
    train_models(args.data_dir, args.output, args.model_name, args.alias, eda_run_id)


if __name__ == "__main__":
    main()
