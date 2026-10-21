import argparse
import asyncio
import csv
import logging
from collections.abc import Iterator
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from cfpb_complaint_classifier.core.config import Settings, get_settings
from cfpb_complaint_classifier.core.logging import setup_logging
from cfpb_complaint_classifier.messaging import (
    BrokerUnavailableError,
    QueueOverloadedError,
    create_broker,
    declare_topology,
    publish_task,
)
from cfpb_complaint_classifier.metrics import SourceMetrics
from cfpb_complaint_classifier.schemas import TaskMessage

logger = logging.getLogger(__name__)


def read_tasks(path: Path, batch_id: str) -> Iterator[TaskMessage]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain a text column")
        for row_number, row in enumerate(reader, start=1):
            yield TaskMessage(
                task_id=uuid5(NAMESPACE_URL, f"{batch_id}:{row_number}"),
                text=row["text"],
            )


async def publish_file(
    path: Path,
    batch_id: str,
    settings: Settings,
    metrics: SourceMetrics | None = None,
) -> int:
    count = 0
    async with create_broker(settings) as broker:
        await declare_topology(broker, settings)
        for task in read_tasks(path, batch_id):
            delay = settings.source_retry_initial_seconds
            while True:
                try:
                    await publish_task(broker, settings, task)
                    if metrics is not None:
                        metrics.record(str(task.task_id), accepted=True)
                    break
                except (QueueOverloadedError, BrokerUnavailableError) as exc:
                    if metrics is not None:
                        metrics.record(str(task.task_id), accepted=False)
                    logger.info(
                        "Task %s waiting: %s, retry in %.3fs",
                        task.task_id,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, settings.source_retry_max_seconds)
            count += 1
        logger.info("Published %d tasks from batch %s", count, batch_id)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish CSV tasks to RabbitMQ")
    parser.add_argument("path", type=Path, help="CSV with a text column")
    parser.add_argument(
        "--batch-id", required=True, help="Stable ID for this unchanged input file"
    )
    parser.add_argument(
        "--metrics-path", type=Path, help="New CSV for publication events"
    )
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    metrics = SourceMetrics(args.metrics_path)
    try:
        asyncio.run(publish_file(args.path, args.batch_id, settings, metrics))
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
