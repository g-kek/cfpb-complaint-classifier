import argparse
import json
import shutil
from contextlib import contextmanager
from zipfile import ZIP_DEFLATED, ZipFile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import requests
from mlflow import MlflowClient

from training import prepare as prepare_module
from training.data import file_digest, load_data
from training.eda import run_eda
from training.prepare import (
    clean_data,
    download_snapshot,
    extract_period,
    prepare,
    split_data,
)
from training.tracking import configure_tracking
from training.train import train_models


@pytest.fixture
def raw_complaints():
    vocabulary = {
        "Mortgage": "mortgage home escrow interest loan payment house",
        "Debt collection": "collector debt calls agency harassment owed",
        "Credit card": "credit card purchase charge merchant refund",
    }
    return pd.DataFrame(
        [
            {
                "Complaint ID": f"{label}-{index}",
                "Date received": "2023-01-15",
                "Consumer complaint narrative": f"{text} complaint number {index}.",
                "Product": label,
            }
            for label, text in vocabulary.items()
            for index in range(50)
        ]
    )


@pytest.fixture
def prepared_data(tmp_path, raw_complaints):
    input_path = tmp_path / "input.csv"
    raw_complaints.to_csv(input_path, index=False)
    args = argparse.Namespace(
        input=input_path,
        source="https://example.org/cfpb.csv",
        output=tmp_path / "prepared",
        start_date="2023-01-01",
        end_date="2023-02-01",
        max_samples=150,
        min_length=40,
        min_class_count=10,
        seed=42,
    )
    prepare(args)
    return args.output


def test_clean_data_removes_duplicates_and_conflicting_labels(raw_complaints):
    extra = raw_complaints.iloc[:3].copy()
    extra["Complaint ID"] = ["duplicate", "conflict", "empty"]
    extra.loc[1, "Product"] = "Credit card"
    extra.loc[2, "Consumer complaint narrative"] = " "
    raw = pd.concat([raw_complaints, extra], ignore_index=True)
    frame, counts = clean_data(raw, "2023-01-01", "2023-02-01", 40, 10)
    assert frame["complaint_id"].is_unique
    assert frame["text"].str.casefold().is_unique
    assert counts["duplicate_text_rows"] == 1
    assert counts["conflicting_text_rows"] == 2
    assert counts["missing_or_short_rows"] == 1


def test_clean_data_rejects_missing_columns(raw_complaints):
    with pytest.raises(ValueError, match="Missing CFPB columns"):
        clean_data(
            raw_complaints.drop(columns="Product"), "2023-01-01", "2023-02-01", 40, 10
        )


def test_clean_data_filters_dates_and_rare_classes(raw_complaints):
    raw_complaints.loc[0, "Date received"] = "2023-02-01"
    raw_complaints.loc[1, "Product"] = "Rare"
    frame, counts = clean_data(raw_complaints, "2023-01-01", "2023-02-01", 40, 10)
    assert counts["outside_date_range"] == 1
    assert counts["excluded_classes"] == {"Rare": 1}
    assert "Rare" not in frame["product"].to_list()
    with pytest.raises(ValueError, match="at least two classes"):
        clean_data(raw_complaints, "2023-01-01", "2023-02-01", 40, 1000)


def test_splits_are_reproducible_and_disjoint(raw_complaints):
    frame, _ = clean_data(raw_complaints, "2023-01-01", "2023-02-01", 40, 10)
    first = split_data(frame, max_samples=120, seed=42)
    second = split_data(frame, max_samples=120, seed=42)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
        assert set(first[name]["product"]) == set(frame["product"])
    ids = [set(split["complaint_id"]) for split in first.values()]
    assert len(set.union(*ids)) == sum(map(len, ids)) == 120
    assert [len(split) for split in first.values()] == [84, 18, 18]
    with pytest.raises(ValueError, match="Too few rows"):
        split_data(frame, max_samples=20, seed=42)


def test_manifest_detects_modified_data(prepared_data):
    frames, manifest = load_data(prepared_data)
    assert sum(len(frame) for frame in frames.values()) == 150
    assert manifest["seed"] == 42
    with (prepared_data / "train.csv").open("a") as stream:
        stream.write("modified\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_data(prepared_data)


def test_prepare_does_not_overwrite_snapshot(prepared_data):
    args = argparse.Namespace(
        output=prepared_data,
        start_date="2023-01-01",
        end_date="2023-02-01",
        max_samples=150,
        min_length=40,
        min_class_count=10,
    )
    with pytest.raises(ValueError, match="not empty"):
        prepare(args)


def test_extract_archive_filters_period(tmp_path, raw_complaints):
    raw_complaints.loc[0, "Date received"] = "2023-02-01"
    archive = tmp_path / "snapshot.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr("complaints.csv", raw_complaints.to_csv(index=False))
    destination = tmp_path / "raw.csv"
    counts = extract_period(archive, destination, "2023-01-01", "2023-02-01")
    assert counts == {"archive_rows": 150, "period_rows": 149}
    assert len(pd.read_csv(destination)) == 149


def test_prepare_downloads_and_checks_archive(tmp_path, raw_complaints, monkeypatch):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr("complaints.csv", raw_complaints.to_csv(index=False))
    monkeypatch.setattr(prepare_module, "ARCHIVE_SHA256", file_digest(archive))
    monkeypatch.setattr(
        prepare_module,
        "download_snapshot",
        lambda url, destination: shutil.copyfile(archive, destination),
    )
    args = argparse.Namespace(
        input=None,
        source=None,
        output=tmp_path / "prepared",
        start_date="2023-01-01",
        end_date="2023-02-01",
        max_samples=150,
        min_length=40,
        min_class_count=10,
        seed=42,
    )
    manifest = prepare(args)
    assert manifest["acquisition"] == {"archive_rows": 150, "period_rows": 150}
    assert manifest["files"]["snapshot.zip"] == file_digest(archive)
    args.output = tmp_path / "invalid"
    monkeypatch.setattr(prepare_module, "ARCHIVE_SHA256", "invalid")
    with pytest.raises(ValueError, match="checksum differs"):
        prepare(args)


def test_download_failure_preserves_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "snapshot.zip"
    destination.write_bytes(b"existing snapshot")

    class Response:
        def raise_for_status(self):
            raise requests.HTTPError("503")

    class Session:
        def mount(self, prefix, adapter):
            pass

        @contextmanager
        def get(self, url, stream, timeout):
            yield Response()

    @contextmanager
    def session():
        yield Session()

    monkeypatch.setattr(prepare_module.requests, "Session", session)
    with pytest.raises(requests.HTTPError):
        download_snapshot("https://example.org/snapshot.zip", destination)
    assert destination.read_bytes() == b"existing snapshot"
    assert not destination.with_suffix(".part").exists()


def test_training_rejects_unrelated_eda_run(tmp_path, prepared_data):
    configure_tracking(f"sqlite:///{tmp_path / 'mlflow.db'}", "test-lineage")
    with mlflow.start_run() as run:
        mlflow.set_tag("dataset.manifest_sha256", "different")
        eda_run_id = run.info.run_id
    with pytest.raises(ValueError, match="different dataset manifest"):
        train_models(prepared_data, tmp_path / "models", "test", "champion", eda_run_id)


def test_training_registers_versions_and_loads_champion(tmp_path, prepared_data):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    configure_tracking(uri, "test-cfpb")
    eda_run_id = run_eda(prepared_data, tmp_path / "eda")
    output = tmp_path / "training"
    summary = train_models(
        prepared_data, output, "test-classifier", "champion", eda_run_id
    )
    client = MlflowClient()
    versions = client.search_model_versions("name='test-classifier'")
    assert len(versions) == 2
    assert {version.run_id for version in versions} == {
        result["run_id"] for result in summary["runs"] if "model_version" in result
    }
    assert len(summary["runs"]) == 3
    champion = client.get_model_version_by_alias("test-classifier", "champion")
    assert champion.version == summary["model_version"]
    model = mlflow.sklearn.load_model("models:/test-classifier@champion")
    prediction = model.predict(
        ["collector debt calls agency harassment owed complaint"]
    )
    assert prediction.tolist() == ["Debt collection"]
    portable = mlflow.pyfunc.load_model("models:/test-classifier@champion")
    assert portable.predict(
        np.array(["collector debt calls agency harassment owed complaint"])
    ).tolist() == ["Debt collection"]
    run = client.get_run(summary["run_id"])
    assert run.info.status == "FINISHED"
    assert len(run.inputs.dataset_inputs) == 3
    assert "test_macro_f1" in run.data.metrics
    for result in summary["runs"]:
        tracked = client.get_run(result["run_id"])
        assert "validation_macro_f1" in tracked.data.metrics
        if result["run_id"] != summary["run_id"]:
            assert "test_macro_f1" not in tracked.data.metrics
    assert (
        json.loads((output / "registry.json").read_text())["run_id"]
        == summary["run_id"]
    )
    other = next(version for version in versions if version.version != champion.version)
    client.set_registered_model_alias("test-classifier", "champion", other.version)
    assert (
        client.get_model_version_by_alias("test-classifier", "champion").version
        == other.version
    )
    assert mlflow.sklearn.load_model("models:/test-classifier@champion").predict(
        ["collector debt calls agency harassment owed complaint"]
    ).tolist() == ["Debt collection"]
