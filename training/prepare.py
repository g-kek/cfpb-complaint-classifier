import argparse
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.model_selection import train_test_split
from urllib3.util.retry import Retry

from training.data import file_digest, write_json

ARCHIVE_URL = (
    "https://files.consumerfinance.gov/f/documents/"
    "CCDB_Export_4_November_2022_through_August_2023.zip"
)
ARCHIVE_SHA256 = "547e8bdb07961861d4bacabff74fd2dd7358c04d225b492b286c37920cb39921"
COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Consumer complaint narrative": "text",
    "Product": "product",
}


def download_snapshot(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(".part")
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            with session.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_period(archive: Path, destination: Path, start: str, end: str) -> dict:
    total_rows = 0
    selected_rows = 0
    with ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]

        if len(names) != 1:
            raise ValueError("Expected one CSV in the CFPB archive")

        with zipped.open(names[0]) as stream:
            chunks = pd.read_csv(
                stream,
                usecols=list(COLUMNS),
                dtype={"Complaint ID": str},
                chunksize=20000,
                low_memory=False,
            )

            for chunk in chunks:
                total_rows += len(chunk)
                dates = pd.to_datetime(chunk["Date received"], errors="coerce")
                selected = chunk.loc[(dates >= start) & (dates < end)]
                selected.to_csv(
                    destination, mode="a", header=not destination.exists(), index=False
                )
                selected_rows += len(selected)

    return {"archive_rows": total_rows, "period_rows": selected_rows}


def clean_data(
    raw: pd.DataFrame,
    start_date: str,
    end_date: str,
    min_length: int,
    min_class_count: int,
) -> tuple[pd.DataFrame, dict]:
    missing = set(COLUMNS) - set(raw.columns)

    if missing:
        raise ValueError(f"Missing CFPB columns: {sorted(missing)}")

    frame = raw[list(COLUMNS)].rename(columns=COLUMNS).copy()
    counts = {"raw_rows": len(frame)}
    frame["date_received"] = pd.to_datetime(frame["date_received"], errors="coerce")
    frame = frame.loc[
        (frame["date_received"] >= start_date) & (frame["date_received"] < end_date)
    ].copy()
    counts["outside_date_range"] = counts["raw_rows"] - len(frame)
    frame["text"] = (
        frame["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    frame["product"] = frame["product"].fillna("").str.strip()
    usable = (
        (frame["text"].str.len() >= min_length)
        & frame["product"].ne("")
        & frame["complaint_id"].notna()
    )
    counts["missing_or_short_rows"] = int((~usable).sum())
    frame = frame.loc[usable].sort_values("complaint_id").copy()
    before = len(frame)
    frame = frame.drop_duplicates("complaint_id")
    counts["duplicate_ids"] = before - len(frame)
    frame["_text_key"] = frame["text"].str.casefold()
    conflicting = frame.groupby("_text_key")["product"].transform("nunique") > 1
    counts["conflicting_text_rows"] = int(conflicting.sum())
    frame = frame.loc[~conflicting]
    before = len(frame)
    frame = frame.drop_duplicates("_text_key").drop(columns="_text_key")
    counts["duplicate_text_rows"] = before - len(frame)
    class_counts = frame["product"].value_counts()
    excluded = class_counts[class_counts < min_class_count]
    counts["excluded_classes"] = excluded.to_dict()
    frame = frame.loc[
        frame["product"].isin(class_counts[class_counts >= min_class_count].index)
    ]
    frame["complaint_id"] = frame["complaint_id"].astype(str)
    frame["date_received"] = frame["date_received"].dt.strftime("%Y-%m-%d")
    counts["eligible_rows"] = len(frame)
    counts["eligible_class_counts"] = frame["product"].value_counts().to_dict()
    if frame["product"].nunique() < 2:
        raise ValueError("Need at least two classes after filtering")
    return frame.reset_index(drop=True), counts


def split_data(
    frame: pd.DataFrame, max_samples: int, seed: int
) -> dict[str, pd.DataFrame]:
    if len(frame) > max_samples:
        frame, _ = train_test_split(
            frame, train_size=max_samples, stratify=frame["product"], random_state=seed
        )
    if frame["product"].value_counts().min() < 10:
        raise ValueError(
            "Too few rows per class; increase max_samples or min_class_count"
        )
    train, holdout = train_test_split(
        frame, test_size=0.3, stratify=frame["product"], random_state=seed
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout["product"], random_state=seed
    )
    return {"train": train, "validation": validation, "test": test}


def prepare(args: argparse.Namespace) -> dict:
    if date.fromisoformat(args.start_date) >= date.fromisoformat(args.end_date):
        raise ValueError("start_date must precede end_date")
    if args.max_samples < 20 or args.min_length < 1 or args.min_class_count < 10:
        raise ValueError("Invalid sampling or filtering settings")
    directory = args.output
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {directory}; use a new directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "raw.csv"
    url = args.source or (args.input.resolve().as_uri() if args.input else ARCHIVE_URL)
    acquisition = {}
    if args.input and args.input.suffix.lower() != ".zip":
        shutil.copyfile(args.input, snapshot)
    else:
        archive = directory / "snapshot.zip"
        if args.input:
            shutil.copyfile(args.input, archive)
        else:
            download_snapshot(url, archive)
        if url == ARCHIVE_URL and file_digest(archive) != ARCHIVE_SHA256:
            raise ValueError("CFPB archive checksum differs from the pinned snapshot")
        acquisition = extract_period(archive, snapshot, args.start_date, args.end_date)
    raw = pd.read_csv(snapshot, dtype={"Complaint ID": str}, low_memory=False)
    frame, counts = clean_data(
        raw, args.start_date, args.end_date, args.min_length, args.min_class_count
    )
    splits = split_data(frame, args.max_samples, args.seed)
    for name, split in splits.items():
        split.sort_values("complaint_id").to_csv(directory / f"{name}.csv", index=False)
    manifest = {
        "source": url,
        "created_at": datetime.now(UTC).isoformat(),
        "task": "Consumer complaint narrative -> Product",
        "acquisition": acquisition,
        "seed": args.seed,
        "start_date_inclusive": args.start_date,
        "end_date_exclusive": args.end_date,
        "max_samples": args.max_samples,
        "min_length": args.min_length,
        "min_class_count": args.min_class_count,
        "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "counts": counts,
        "splits": {
            name: {
                "rows": len(split),
                "class_counts": split["product"].value_counts().to_dict(),
            }
            for name, split in splits.items()
        },
        "files": {
            path.name: file_digest(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fixed CFPB snapshot")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--source", help="Source URI of a supplied CSV")
    parser.add_argument("--output", type=Path, default=Path("data/cfpb"))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-02-01")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--min-class-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    manifest = prepare(parser.parse_args())
    print(f"Prepared splits: {manifest['splits']}")


if __name__ == "__main__":
    main()
