import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


class SourceMetrics:
    def __init__(self, path: Path | None = None):
        self.accepted = 0
        self.rejected = 0
        self.stream: TextIO | None = None
        self.writer = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("x", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=["timestamp", "task_id", "event", "accepted", "rejected"],
            )
            self.writer.writeheader()

    def record(self, task_id: str, *, accepted: bool) -> None:
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        if self.writer is not None:
            self.writer.writerow(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "task_id": task_id,
                    "event": "accepted" if accepted else "rejected",
                    "accepted": self.accepted,
                    "rejected": self.rejected,
                }
            )
            self.stream.flush()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
