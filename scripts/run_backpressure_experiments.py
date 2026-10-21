import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import httpx
from sqlalchemy import select

import docker
from cfpb_complaint_classifier.core.config import Settings
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.db.tasks import task_results

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
DOCKER = shutil.which("docker") or "docker"
GIT = shutil.which("git") or "git"


def compose(*args, env):
    subprocess.run([DOCKER, "compose", *args], cwd=ROOT, env=env, check=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_events(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def memory(container):
    stats = container.stats(stream=False, one_shot=True)["memory_stats"]

    return max(
        0, stats.get("usage", 0) - stats.get("stats", {}).get("inactive_file", 0)
    )


def docker_client():
    endpoint = (
        os.getenv("DOCKER_HOST")
        or subprocess.check_output(
            [DOCKER, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True,
        ).strip()
    )
    return docker.DockerClient(base_url=endpoint)


async def run_mode(args, client, mode, bounded, workers, run_id):
    folder = args.output / mode
    folder.mkdir(exist_ok=args.resume)
    if (folder / "config.json").exists():
        print(f"Skipping completed {mode}", flush=True)
        return
    if any(folder.iterdir()):
        raise FileExistsError(
            f"Partial measurements in {folder}; use a new output directory"
        )
    batch = f"{run_id}:{mode}"
    queue = f"experiment.{run_id}.{mode.lower()}"
    env = {
        **os.environ,
        "TASK_QUEUE_NAME": queue,
        "BACKPRESSURE_ENABLED": str(bounded).lower(),
        "QUEUE_MAX_LENGTH": str(args.queue_limit),
        "PREFETCH_COUNT": "2",
        "WORKER_CONCURRENCY": "2",
        "INFERENCE_DELAY_SECONDS": str(args.delay),
        "LOG_LEVEL": "INFO",
    }
    await asyncio.to_thread(compose, "up", "-d", "--no-deps", "mlflow", env=env)
    await asyncio.to_thread(compose, "run", "--rm", "--no-deps", "db-init", env=env)
    await asyncio.to_thread(
        compose,
        "up",
        "-d",
        "--no-deps",
        "--force-recreate",
        "--scale",
        "worker=1",
        "app",
        "worker",
        env=env,
    )

    def service(name):
        ids = subprocess.check_output(
            [DOCKER, "compose", "ps", "-q", name],
            cwd=ROOT,
            env=env,
            text=True,
        ).split()
        return [client.containers.get(cid) for cid in ids]

    worker_containers = await asyncio.to_thread(service, "worker")
    api = (await asyncio.to_thread(service, "app"))[0]
    rabbit = (await asyncio.to_thread(service, "rabbitmq"))[0]
    mlflow = (await asyncio.to_thread(service, "mlflow"))[0]

    async def wait_healthy(containers):
        deadline = time.monotonic() + 120
        while True:
            states = []
            for container in containers:
                await asyncio.to_thread(container.reload)
                states.append(container.attrs["State"].get("Health", {}).get("Status"))
                if container.status == "exited":
                    raise RuntimeError(container.logs(tail=30).decode())
            if all(state == "healthy" for state in states):
                return
            if time.monotonic() > deadline:
                raise TimeoutError("Workers did not become healthy; check MLflow")
            await asyncio.sleep(0.5)

    await wait_healthy([mlflow, api, *worker_containers])
    if workers == 2:
        await asyncio.to_thread(
            compose, "up", "-d", "--no-deps", "--scale", "worker=2", "worker", env=env
        )
        worker_containers = await asyncio.to_thread(service, "worker")
        await wait_healthy(worker_containers)
    await asyncio.to_thread(compose, "stop", "mlflow", env=env)
    await asyncio.to_thread(mlflow.reload)
    if mlflow.status != "exited":
        raise RuntimeError("MLflow did not stop before the load phase")
    network = next(iter(api.attrs["NetworkSettings"]["Networks"]))
    task_ids = [uuid5(NAMESPACE_URL, f"{batch}:{i}") for i in range(1, args.tasks + 1)]
    engine = create_engine(Settings())
    user = os.getenv("RABBITMQ_USER", "cfpb")
    password = os.getenv("RABBITMQ_PASSWORD", "cfpb-local-password")
    source = None
    samples = []
    api_requests = []
    started = datetime.now(UTC)
    start_clock = time.monotonic()
    previous_accepted = previous_completed = 0
    previous_elapsed = 0.0
    try:
        source = await asyncio.to_thread(
            client.containers.run,
            "cfpb-complaint-classifier:local",
            [
                "python",
                "-m",
                "cfpb_complaint_classifier.source",
                "/measurements/input.csv",
                "--batch-id",
                batch,
                "--metrics-path",
                f"/measurements/{mode}/source.csv",
            ],
            detach=True,
            network=network,
            user=f"{os.getuid()}:{os.getgid()}",
            environment={
                "RABBITMQ_URL": f"amqp://{user}:{password}@rabbitmq:5672/",
                "TASK_QUEUE_NAME": queue,
                "BACKPRESSURE_ENABLED": str(bounded).lower(),
                "QUEUE_MAX_LENGTH": str(args.queue_limit),
                "LOG_LEVEL": "WARNING",
            },
            volumes={
                str(args.output.resolve()): {"bind": "/measurements", "mode": "rw"}
            },
        )

        started = datetime.now(UTC)
        start_clock = time.monotonic()
        async with httpx.AsyncClient(timeout=3) as http:
            while True:
                tick = time.monotonic()
                elapsed = tick - start_clock
                async with engine.connect() as conn:
                    results = (
                        (
                            await conn.execute(
                                select(task_results).where(
                                    task_results.c.task_id.in_(task_ids)
                                )
                            )
                        )
                        .mappings()
                        .all()
                    )
                events = await asyncio.to_thread(source_events, folder / "source.csv")
                accepted = int(events[-1]["accepted"]) if events else 0
                rejected = int(events[-1]["rejected"]) if events else 0
                response = await http.get(
                    f"http://127.0.0.1:15672/api/queues/%2F/{quote(queue, safe='')}",
                    auth=(user, password),
                )
                response.raise_for_status()
                info = response.json()
                dlq_response = await http.get(
                    "http://127.0.0.1:15672/api/queues/%2F/"
                    f"{quote(queue + '.dlq', safe='')}",
                    auth=(user, password),
                )
                dlq_response.raise_for_status()
                dlq = dlq_response.json().get("messages", 0)
                memory_values = await asyncio.gather(
                    *(
                        asyncio.to_thread(memory, container)
                        for container in [source, api, rabbit, *worker_containers]
                    )
                )
                dt = max(0.001, elapsed - previous_elapsed)
                row = {
                    "elapsed_seconds": round(elapsed, 3),
                    "phase": "load" if elapsed <= args.seconds else "drain",
                    "ready": info.get("messages_ready", 0),
                    "unacked": info.get("messages_unacknowledged", 0),
                    "dlq": dlq,
                    "accepted": accepted,
                    "publication_rejections": rejected,
                    "completed": len(results),
                    "source_per_second": (accepted - previous_accepted) / dt,
                    "completed_per_second": (len(results) - previous_completed) / dt,
                    "source_bytes": memory_values[0],
                    "api_bytes": memory_values[1],
                    "rabbitmq_bytes": memory_values[2],
                    "workers_bytes": sum(memory_values[3:]),
                    "worker_1_bytes": memory_values[3],
                    "worker_2_bytes": memory_values[4] if workers == 2 else 0,
                }
                samples.append(row)
                write_csv(folder / "samples.csv", samples)
                previous_accepted, previous_completed, previous_elapsed = (
                    accepted,
                    len(results),
                    elapsed,
                )

                if elapsed <= args.seconds and elapsed >= 3:
                    probe_id = str(uuid4())
                    probe_started = time.monotonic()
                    probe = await http.post(
                        "http://127.0.0.1:8080/api/v1/tasks",
                        json={
                            "task_id": probe_id,
                            "text": "My credit card payment was applied incorrectly.",
                        },
                    )
                    api_requests.append(
                        {
                            "elapsed_seconds": elapsed,
                            "task_id": probe_id,
                            "status": probe.status_code,
                            "retry_after": probe.headers.get("retry-after", ""),
                            "response_seconds": time.monotonic() - probe_started,
                        }
                    )
                await asyncio.to_thread(source.reload)
                if source.status == "exited" and source.attrs["State"]["ExitCode"] != 0:
                    raise RuntimeError(source.logs(tail=30).decode())
                for container in worker_containers:
                    await asyncio.to_thread(container.reload)
                    if container.status != "running":
                        raise RuntimeError(
                            f"Worker {container.name} exited during load: "
                            f"{container.attrs['State']['ExitCode']}"
                        )
                if elapsed >= args.seconds and len(results) == args.tasks:
                    break
                if elapsed > args.drain_timeout:
                    raise TimeoutError(
                        "Batch did not drain; inspect saved samples and logs"
                    )
                await asyncio.sleep(max(0, args.interval - (time.monotonic() - tick)))

            await asyncio.to_thread(compose, "stop", "worker", env=env)
            probe_ids = [
                UUID(row["task_id"]) for row in api_requests if row["status"] == 202
            ]
            async with engine.connect() as conn:
                all_results = (
                    (
                        await conn.execute(
                            select(task_results).where(
                                task_results.c.task_id.in_([*task_ids, *probe_ids])
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
            commits = {}
            worker_commits = []
            for container in worker_containers:
                logs = (await asyncio.to_thread(container.logs)).decode()
                count = 0
                for line in logs.splitlines():
                    if "task_commit " in line:
                        event = json.loads(line.split("task_commit ", 1)[1])
                        commits.setdefault(event["task_id"], event["committed_at"])
                        count += 1
                worker_commits.append(count)
            if any(count == 0 for count in worker_commits):
                raise RuntimeError(f"An expected worker was idle: {worker_commits}")
            if any(str(r["task_id"]) not in commits for r in all_results):
                raise RuntimeError("Missing post-commit metrics; use LOG_LEVEL=INFO")
            write_csv(
                folder / "tasks.csv",
                [
                    {
                        **dict(result),
                        "committed_at": commits[str(result["task_id"])],
                        "inference_seconds": (
                            result["completed_at"] - result["started_at"]
                        ).total_seconds(),
                        "processing_seconds": (
                            datetime.fromisoformat(commits[str(result["task_id"])])
                            - result["started_at"]
                        ).total_seconds(),
                        "end_to_end_seconds": (
                            datetime.fromisoformat(commits[str(result["task_id"])])
                            - result["created_at"]
                        ).total_seconds(),
                        "population": (
                            "batch" if result["task_id"] in task_ids else "api"
                        ),
                    }
                    for result in all_results
                ],
            )
            write_csv(folder / "api.csv", api_requests)

            for _ in range(30):
                final = (
                    await http.get(
                        "http://127.0.0.1:15672/api/queues/%2F/"
                        f"{quote(queue, safe='')}",
                        auth=(user, password),
                    )
                ).json()
                if final.get("messages", 0) == 0:
                    break
                await asyncio.sleep(0.5)
            accepted_total = accepted + len(probe_ids)
            balance = {
                "accepted_batch": accepted,
                "accepted_api": len(probe_ids),
                "results": len(all_results),
                "dlq": dlq,
                "remaining": final.get("messages", 0),
            }
            assert accepted_total == len(all_results) + dlq + balance["remaining"], (
                balance
            )
            assert balance["remaining"] == 0 and dlq == 0, balance
            config = {
                "mode": mode,
                "bounded": bounded,
                "workers": workers,
                "queue": queue,
                "queue_limit": args.queue_limit if bounded else None,
                "prefetch_per_worker": 2 if bounded else 0,
                "concurrency_per_worker": 2,
                "inference_delay_seconds": args.delay,
                "tasks": args.tasks,
                "observation_seconds": args.seconds,
                "sample_interval_seconds": args.interval,
                "started_at": started.isoformat(),
                "balance": balance,
                "image_id": api.image.id,
                "model_versions": sorted({r["model_version"] for r in all_results}),
                "worker_commits": worker_commits,
                "mlflow_stopped_during_load": True,
            }
            (folder / "config.json").write_text(json.dumps(config, indent=2))
            print(f"{mode}: {balance}", flush=True)
    finally:
        await engine.dispose()
        if source is not None:
            (folder / "source.log").write_bytes(await asyncio.to_thread(source.logs))
            await asyncio.to_thread(source.remove, force=True)
        for index, container in enumerate(worker_containers):
            (folder / f"worker-{index + 1}.log").write_bytes(
                await asyncio.to_thread(container.logs)
            )
        await asyncio.to_thread(compose, "start", "mlflow", env=env)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sentence = (
        "My credit card payment was applied incorrectly "
        "and the bank did not resolve my complaint. "
    )
    input_path = args.output / "input.csv"
    previous_metadata = None
    if args.resume:
        previous_metadata = json.loads((args.output / "environment.json").read_text())
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if digest != previous_metadata["input_sha256"]:
            raise ValueError("Resume input differs from the recorded measurement")
        for config_path in args.output.glob("E*/config.json"):
            config = json.loads(config_path.read_text())
            if (
                config["tasks"],
                config["inference_delay_seconds"],
                config["observation_seconds"],
            ) != (args.tasks, args.delay, args.seconds):
                raise ValueError(
                    "Resume must use the same tasks, delay and observation window"
                )
            if config["bounded"] and config["queue_limit"] != args.queue_limit:
                raise ValueError("Resume must use the same bounded queue limit")
    else:
        with input_path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text"])
            for _ in range(args.tasks):
                writer.writerow([sentence * 40])
    metadata = {
        "host": platform.platform(),
        "input_sha256": hashlib.sha256(
            (args.output / "input.csv").read_bytes()
        ).hexdigest(),
        "git_commit": (
            await asyncio.to_thread(
                subprocess.check_output, [GIT, "rev-parse", "HEAD"], text=True
            )
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted((ROOT / "src").rglob("*.py")),
                SCRIPT,
                ROOT / "uv.lock",
            ]
        },
        "working_tree_changes": await asyncio.to_thread(
            subprocess.check_output, [GIT, "diff", "--stat"], text=True
        ),
        "measurement": (
            "Docker cgroup working set: usage minus inactive_file; "
            "exited source reports zero"
        ),
    }
    if previous_metadata is not None:
        metadata["previous_execution"] = previous_metadata
    metadata["requested_modes"] = args.modes
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    client = docker_client()
    info = await asyncio.to_thread(client.info)
    metadata.update(
        {
            "docker_memory_bytes": info["MemTotal"],
            "docker_cpus": info["NCPU"],
            "docker_server_version": info["ServerVersion"],
        }
    )
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    run_id = uuid4().hex[:12]
    try:
        for mode, bounded, workers in [
            ("E1", False, 1),
            ("E2", True, 1),
            ("E3", False, 2),
            ("E4", True, 2),
        ]:
            if mode not in args.modes:
                continue
            print(f"Starting {mode}", flush=True)
            await run_mode(args, client, mode, bounded, workers, run_id)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure four backpressure modes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed modes in an existing directory",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["E1", "E2", "E3", "E4"],
        default=["E1", "E2", "E3", "E4"],
    )
    parser.add_argument("--tasks", type=int, default=800)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--queue-limit", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--drain-timeout", type=float, default=300)
    asyncio.run(main(parser.parse_args()))
