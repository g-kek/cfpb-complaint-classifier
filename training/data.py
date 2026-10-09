import hashlib
import json
from pathlib import Path

import pandas as pd


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_data(directory: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    frames = {}
    for name, expected_digest in manifest["files"].items():
        path = directory / name
        if file_digest(path) != expected_digest:
            raise ValueError(f"Checksum mismatch: {path}")
        if name in {"train.csv", "validation.csv", "test.csv"}:
            frames[path.stem] = pd.read_csv(path, dtype={"complaint_id": str})
    return frames, manifest
