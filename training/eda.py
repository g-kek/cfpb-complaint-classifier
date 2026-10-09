import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from training.data import load_data, write_json
from training.tracking import (
    EXPERIMENT_NAME,
    configure_tracking,
    log_dataset,
    log_provenance,
)

plt.switch_backend("Agg")


def plot_class_distribution(class_counts: pd.Series, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    class_counts.plot.barh(ax=ax, color="#377eb8")
    ax.set(xlabel="Complaints", ylabel="Product", title="Class distribution")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_text_lengths(lengths: pd.Series, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=80, range=(0, lengths.quantile(0.99)), color="#377eb8")
    ax.set(
        xlabel="Characters",
        ylabel="Complaints",
        title="Text length (up to 99th percentile)",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def frequent_words(train: pd.DataFrame) -> dict[str, list[str]]:
    vectorizer = CountVectorizer(stop_words="english", min_df=5, max_features=20000)
    words = vectorizer.fit_transform(train["text"])
    vocabulary = vectorizer.get_feature_names_out()
    top_words = {}
    for label in sorted(train["product"].unique()):
        mask = train["product"].eq(label).to_numpy()
        frequencies = words[mask].sum(axis=0).A1
        top_indices = frequencies.argsort()[-10:][::-1]
        top_words[label] = vocabulary[top_indices].tolist()
    return top_words


def create_artifacts(
    frames: dict[str, pd.DataFrame], manifest: dict, output: Path
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames.values(), ignore_index=True)
    class_counts = frame["product"].value_counts().sort_values()
    lengths = frame["text"].str.len()

    plot_class_distribution(class_counts, output / "class_distribution.png")
    plot_text_lengths(lengths, output / "text_length.png")
    length_summary = (
        frame.assign(characters=lengths)
        .groupby("product")["characters"]
        .agg(["count", "median", "mean", "min", "max"])
    )
    length_summary.to_csv(output / "length_by_class.csv")
    proportions = pd.DataFrame(
        {
            name: split["product"].value_counts(normalize=True)
            for name, split in frames.items()
        }
    ).fillna(0)
    proportions.to_csv(output / "split_class_proportions.csv")
    write_json(output / "top_words.json", frequent_words(frames["train"]))
    stats = {
        "rows": len(frame),
        "classes": len(class_counts),
        "filtering": manifest["counts"],
        "majority_share": float(class_counts.max() / len(frame)),
        "median_text_length": float(lengths.median()),
        "p99_text_length": float(lengths.quantile(0.99)),
        "max_split_proportion_difference": float(
            proportions.max(axis=1).sub(proportions.min(axis=1)).max()
        ),
    }
    write_json(output / "summary.json", stats)
    return [
        output / name
        for name in (
            "class_distribution.png",
            "text_length.png",
            "length_by_class.csv",
            "split_class_proportions.csv",
            "top_words.json",
            "summary.json",
        )
    ]


def run_eda(directory: Path, output: Path) -> str:
    frames, manifest = load_data(directory)
    artifacts = create_artifacts(frames, manifest, output)
    with mlflow.start_run(run_name="eda") as run:
        log_provenance(directory, manifest)
        for name, frame in frames.items():
            log_dataset(frame, manifest, name)
            mlflow.log_metric(f"{name}_rows", len(frame))
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), "eda")
        mlflow.log_artifacts(str(directory), "data")
        run_id = run.info.run_id
    write_json(output / "tracking.json", {"eda_run_id": run_id})
    print(f"EDA run: {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Log CFPB EDA to MLflow")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cfpb"))
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args()
    configure_tracking(args.tracking_uri, args.experiment)
    run_eda(args.data_dir, args.output)


if __name__ == "__main__":
    main()
