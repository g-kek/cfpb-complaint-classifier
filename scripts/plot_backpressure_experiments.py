import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"E1": "#2457A7", "E2": "#B56B16", "E3": "#6B7532", "E4": "#BA527B"}
LABELS = {
    "E1": "Unbounded · 1 worker",
    "E2": "Bounded · 1 worker",
    "E3": "Unbounded · 2 workers",
    "E4": "Bounded · 2 workers",
}


def scenario_axes(axes, runs):
    for ax, mode in zip(axes.flat, COLORS, strict=True):
        if mode in runs:
            yield ax, (mode, runs[mode])
        else:
            ax.set_title(f"{mode}: no measurements")
            ax.text(
                0.5, 0.5, "Not run", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()


def main(root):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    runs = {}
    summary = []
    for mode in COLORS:
        if not (root / mode / "config.json").exists():
            continue
        config = json.loads((root / mode / "config.json").read_text())
        samples = pd.read_csv(root / mode / "samples.csv")
        tasks = pd.read_csv(root / mode / "tasks.csv")
        tasks = tasks.loc[tasks.population == "batch"].copy()
        assert len(tasks) == config["tasks"] == config["balance"]["accepted_batch"]
        assert tasks.task_id.is_unique
        assert config["balance"]["remaining"] == config["balance"]["dlq"] == 0
        events = pd.read_csv(root / mode / "source.csv")
        accepted = events.loc[events.event == "accepted"]
        assert len(accepted) == config["tasks"]
        assert accepted.task_id.is_unique
        load = samples.loc[samples.elapsed_seconds <= config["observation_seconds"]]
        steady = load.loc[load.elapsed_seconds >= 5]
        span = steady.elapsed_seconds.iloc[-1] - steady.elapsed_seconds.iloc[0]
        source_rate = (steady.accepted.iloc[-1] - steady.accepted.iloc[0]) / span
        completion_rate = (steady.completed.iloc[-1] - steady.completed.iloc[0]) / span
        api = pd.read_csv(root / mode / "api.csv")
        failures = api.loc[api.status == 503]
        start = datetime.fromisoformat(config["started_at"])
        tasks["elapsed_seconds"] = (
            pd.to_datetime(tasks.committed_at, utc=True) - start
        ).dt.total_seconds()

        tasks["cohort"] = (tasks.elapsed_seconds // 5 * 5 + 5).astype(int)
        latency = tasks.groupby("cohort").end_to_end_seconds.agg(
            p50=lambda values: values.quantile(0.5),
            p95=lambda values: values.quantile(0.95),
        )
        summary.append(
            {
                "mode": mode,
                "workers": config["workers"],
                "bounded": config["bounded"],
                "max_ready": int(samples.ready.max()),
                "max_unacked": int(samples.unacked.max()),
                "max_backlog": int((samples.ready + samples.unacked).max()),
                "steady_source_per_second": source_rate,
                "steady_batch_completed_per_second": completion_rate,
                "source_observation_per_second": load.accepted.iloc[-1]
                / load.elapsed_seconds.iloc[-1],
                "task_p50_seconds": tasks.end_to_end_seconds.quantile(0.5),
                "task_p95_seconds": tasks.end_to_end_seconds.quantile(0.95),
                "processing_p50_seconds": tasks.processing_seconds.quantile(0.5),
                "processing_p95_seconds": tasks.processing_seconds.quantile(0.95),
                "source_peak_mib": samples.source_bytes.max() / 2**20,
                "api_peak_mib": samples.api_bytes.max() / 2**20,
                "rabbitmq_peak_mib": samples.rabbitmq_bytes.max() / 2**20,
                "workers_peak_mib": samples.workers_bytes.max() / 2**20,
                "api_503": len(failures),
                "api_202": int((api.status == 202).sum()),
                "api_503_p95_seconds": failures.response_seconds.quantile(0.95)
                if len(failures)
                else None,
                "batch_results": len(tasks),
                "remaining": config["balance"]["remaining"],
            }
        )
        runs[mode] = (config, samples, tasks, latency)
    if not runs:
        raise ValueError("No completed measurements found")
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (mode, (config, samples, _, _)) in scenario_axes(axes, runs):
        view = samples.loc[samples.elapsed_seconds <= config["observation_seconds"]]
        ax.plot(
            view.elapsed_seconds,
            view.ready + view.unacked,
            color=COLORS[mode],
            label="Ready + unacked",
        )
        ax.plot(
            view.elapsed_seconds,
            view.ready,
            color="#555555",
            linestyle="--",
            label="Ready",
        )
        ax.plot(
            view.elapsed_seconds,
            view.unacked,
            color=COLORS[mode],
            linestyle=":",
            label="Unacked",
        )
        if config["bounded"]:
            ax.axhline(
                config["queue_limit"],
                color="#888888",
                linewidth=0.8,
                linestyle="-.",
                label="Ready limit",
            )
        ax.set_title(f"{mode}: {LABELS[mode]}")
        ax.set_ylabel("Messages")
        ax.set_xlabel("Seconds from source start")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.suptitle("Broker backlog during the same observation window")
    fig.tight_layout()
    fig.savefig(root / "depth.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (mode, (config, samples, _, _)) in scenario_axes(axes, runs):
        view = samples.loc[samples.elapsed_seconds <= config["observation_seconds"]]
        for column, style, label in [
            ("source_per_second", "-", "Confirmed batch publications"),
            ("completed_per_second", "--", "Committed batch results"),
        ]:
            ax.plot(
                view.elapsed_seconds,
                view[column].rolling(3, min_periods=1).mean(),
                color=COLORS[mode] if style == "-" else "#444444",
                linestyle=style,
                label=label,
            )
        ax.set_title(f"{mode}: {LABELS[mode]}")
        ax.set_xlabel("Seconds from source start")
        ax.set_ylabel("Tasks / second · 3-sample mean")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.suptitle("Source adaptation and completions · independent y scales")
    fig.tight_layout()
    fig.savefig(root / "rates.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (column, title) in zip(
        axes.flat,
        [
            ("source_bytes", "Source (zero after process exit)"),
            ("api_bytes", "API"),
            ("workers_bytes", "Workers, summed"),
            ("rabbitmq_bytes", "RabbitMQ"),
        ],
        strict=True,
    ):
        for mode, (config, samples, _, _) in runs.items():
            view = samples.loc[samples.elapsed_seconds <= config["observation_seconds"]]
            ax.plot(
                view.elapsed_seconds,
                view[column] / 2**20,
                label=mode,
                color=COLORS[mode],
                linestyle="--" if mode in {"E3", "E4"} else "-",
            )
        ax.set_title(title)
        ax.set_xlabel("Seconds from source start")
        ax.set_ylabel("Working set · MiB")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle("Measured container memory · cgroup usage minus inactive_file")
    fig.tight_layout()
    fig.savefig(root / "memory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (mode, (_, _, _, latency)) in scenario_axes(axes, runs):
        ax.plot(latency.index, latency.p50, color=COLORS[mode], label="p50")
        ax.plot(
            latency.index, latency.p95, color=COLORS[mode], linestyle="--", label="p95"
        )
        ax.set_title(f"{mode}: {LABELS[mode]}")
        ax.set_xlabel("Completion cohort end · seconds from source start")
        ax.set_ylabel("First publication attempt → commit · seconds")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.legend()
    fig.suptitle("Batch task latency by 5-second completion cohort · including drain")
    fig.tight_layout()
    fig.savefig(root / "latency.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot backpressure measurements")
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
