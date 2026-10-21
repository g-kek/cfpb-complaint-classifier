import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import requests
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
TEXT = "A debt collector keeps calling about a debt I do not owe."


def predict(url: str) -> dict:
    task_id = str(uuid4())
    response = requests.post(
        f"{url.rstrip('/')}/api/v1/tasks",
        json={"text": TEXT, "task_id": task_id},
        timeout=10,
    )
    response.raise_for_status()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = requests.get(f"{url.rstrip('/')}/api/v1/tasks/{task_id}", timeout=10)
        if result.status_code == 404:
            time.sleep(0.5)
            continue
        result.raise_for_status()
        payload = result.json()
        return {
            key: payload[key] for key in ("category", "model_name", "model_version")
        }
    raise RuntimeError(f"Task {task_id} did not complete within 120 seconds")


def restart_and_wait(url: str, expected_version: str) -> dict:
    subprocess.run(["docker", "compose", "restart", "worker"], cwd=ROOT, check=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            result = predict(url)
            if result["model_version"] == expected_version:
                return result
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Worker did not start with model version {expected_version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MLflow alias switching")
    parser.add_argument("--app-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "cfpb-complaint-classifier"),
    )
    parser.add_argument("--alias", default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"))
    parser.add_argument("--target-version")
    args = parser.parse_args()
    client = MlflowClient(
        tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri
    )
    original = str(
        client.get_model_version_by_alias(args.model_name, args.alias).version
    )
    candidates = client.search_model_versions(f"name='{args.model_name}'")
    target = args.target_version or next(
        (
            str(v.version)
            for v in candidates
            if str(v.version) != original and v.status == "READY"
        ),
        None,
    )
    if target is None or target == original:
        raise RuntimeError(
            "Need another registered version to demonstrate an alias switch"
        )
    before = predict(args.app_url)
    if before["model_version"] != original or before["model_name"] != args.model_name:
        raise RuntimeError(
            "Restart worker first: its loaded model differs from the configured alias"
        )
    try:
        client.set_registered_model_alias(args.model_name, args.alias, target)
        without_restart = predict(args.app_url)
        if without_restart != before:
            raise RuntimeError("A running worker changed its loaded model")
        after = restart_and_wait(args.app_url, target)
    finally:
        client.set_registered_model_alias(args.model_name, args.alias, original)
        restored = restart_and_wait(args.app_url, original)
    report = {
        "before": before,
        "alias_changed_without_restart": without_restart,
        "after_restart": after,
        "restored": restored,
    }
    destination = ROOT / "reports" / "inference" / "alias-switch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
