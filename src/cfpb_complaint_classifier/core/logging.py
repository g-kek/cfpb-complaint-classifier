import json
import logging
import os
import re
import socket
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry import trace

batch_id = ContextVar("batch_id", default=None)
task_id = ContextVar("task_id", default=None)
_credentials = re.compile(r"(://)[^/\s@]+@")


@contextmanager
def task_context(identifier, batch=None):
    batch_token = batch_id.set(batch)
    token = task_id.set(str(identifier) if identifier else None)
    try:
        yield
    finally:
        task_id.reset(token)
        batch_id.reset(batch_token)


def safe_stack(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    return "\n".join(
        [f"{f.filename}:{f.lineno} in {f.name}" for f in frames] + [type(exc).__name__]
    )


class JsonFormatter(logging.Formatter):
    def format(self, record):
        span = trace.get_current_span().get_span_context()
        message = record.getMessage()
        # Transport exceptions and SQL diagnostics may embed a full message or row.
        if record.name.split(".")[0] in {"aio_pika", "aiormq", "sqlalchemy"}:
            message = f"{record.name} {record.funcName}"
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "service": os.getenv("OTEL_SERVICE_NAME", "pipeline"),
            "level": record.levelname,
            "instance": os.getenv("OTEL_INSTANCE_ID") or socket.gethostname(),
            "env": os.getenv("OTEL_ENVIRONMENT", "development"),
            "logger": record.name,
            "message": _credentials.sub(r"\1***@", message),
            "task_id": task_id.get(),
            "batch_id": batch_id.get(),
            "trace_id": format(span.trace_id, "032x") if span.is_valid else None,
            "span_id": format(span.span_id, "016x") if span.is_valid else None,
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info and record.exc_info[1]:
            entry["stacktrace"] = safe_stack(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for existing in root.handlers[:]:
        if (
            isinstance(existing, logging.StreamHandler)
            and existing.stream in (sys.stdout, sys.stderr)
            and not isinstance(existing.formatter, JsonFormatter)
        ):
            root.removeHandler(existing)
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "mlflow"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    logging.captureWarnings(True)
