import logging
import os
import socket
from contextlib import contextmanager
from time import perf_counter
from urllib.parse import urlsplit

import psutil
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.trace import StatusCode

from cfpb_complaint_classifier.core.logging import safe_stack

tracer = trace.get_tracer("cfpb.pipeline")
meter = metrics.get_meter("cfpb.pipeline")
requests = meter.create_counter("pipeline_api_requests")
api_duration = meter.create_histogram(
    "pipeline_api_duration_seconds",
    explicit_bucket_boundaries_advisory=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
    ),
)
published = meter.create_counter("pipeline_source_published")
nacks = meter.create_counter("pipeline_source_nacks")
wait_seconds = meter.create_counter("pipeline_source_wait_seconds")
processed = meter.create_counter("pipeline_worker_processed")
in_flight = meter.create_up_down_counter("pipeline_worker_in_flight")
redeliveries = meter.create_counter("pipeline_worker_redeliveries")
task_duration = meter.create_histogram(
    "pipeline_task_duration_seconds",
    explicit_bucket_boundaries_advisory=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 60, 120),
)

_providers = None


def enabled():
    return _providers is not None


def collector_present(endpoint_variable):
    # Runs on the SDK export thread. Docker removes the service DNS entry
    # while its profile is inactive and restores it on the next compose up.
    if os.getenv("OTEL_DISCOVER_COLLECTOR", "").lower() != "true":
        return True
    url = os.getenv(endpoint_variable) or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not url:
        return False
    host = urlsplit(url).hostname
    try:
        socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    return True


class ProfileSpanExporter(OTLPSpanExporter):
    def export(self, spans):
        if collector_present("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            return super().export(spans)
        return SpanExportResult.SUCCESS


class ProfileMetricExporter(OTLPMetricExporter):
    def export(self, metrics_data, timeout_millis=10000, **kwargs):
        if collector_present("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
            return super().export(metrics_data, timeout_millis=timeout_millis, **kwargs)
        return MetricExportResult.SUCCESS


def configure(settings, role):
    global _providers
    if _providers or os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    if not any(
        os.getenv(key)
        for key in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        )
    ):
        return
    attributes = {
        "service.instance.id": os.getenv("OTEL_INSTANCE_ID") or socket.gethostname()
    }
    resource = Resource.create()
    if "service.instance.id=" not in os.getenv("OTEL_RESOURCE_ATTRIBUTES", ""):
        resource = resource.merge(Resource(attributes))
    traces = TracerProvider(resource=resource)
    traces.add_span_processor(BatchSpanProcessor(ProfileSpanExporter()))
    reader = PeriodicExportingMetricReader(ProfileMetricExporter())
    meters = MeterProvider(resource=resource, metric_readers=[reader])
    trace.set_tracer_provider(traces)
    metrics.set_meter_provider(meters)
    _providers = (traces, meters)
    process = psutil.Process()
    meter.create_observable_gauge(
        "pipeline_process_resident_bytes",
        callbacks=[lambda _: [Observation(process.memory_info().rss)]],
    )
    if role == "source":
        for counter in (published, nacks, wait_seconds):
            counter.add(0)
    if role == "worker":
        for result in ("ok", "error", "dlq"):
            processed.add(0, {"result": result})
        in_flight.add(0)
        redeliveries.add(0)
    if role == "api":
        for status in ("202", "422", "503"):
            requests.add(0, {"route": "/api/v1/tasks", "status": status})
    if role == "api" and settings.backpressure_enabled:
        meter.create_observable_gauge(
            "pipeline_queue_limit",
            callbacks=[
                lambda _: [
                    Observation(
                        settings.queue_max_length, {"queue": settings.task_queue_name}
                    )
                ]
            ],
        )


def flush():
    if _providers:
        for provider in _providers:
            try:
                provider.force_flush(timeout_millis=2000)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "telemetry_flush_failed",
                    extra={"fields": {"error": type(exc).__name__}},
                )


def error(span, exc):
    span.set_status(StatusCode.ERROR, type(exc).__name__)
    span.record_exception(
        exc,
        attributes={
            "exception.message": type(exc).__name__,
            "exception.stacktrace": safe_stack(exc),
        },
    )


@contextmanager
def span(name, **options):
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False, **options
    ) as active:
        try:
            yield active
        except Exception as exc:
            error(active, exc)
            raise


@contextmanager
def timing(histogram, attributes=None):
    start = perf_counter()
    try:
        yield
    finally:
        histogram.record(perf_counter() - start, attributes)


def inference_histogram(delay):
    base = max(delay, 0.005)
    return meter.create_histogram(
        "pipeline_worker_inference_seconds",
        explicit_bucket_boundaries_advisory=tuple(
            base * ratio for ratio in (0.1, 0.5, 0.9, 1, 1.05, 1.1, 1.25, 1.5, 2, 4)
        ),
    )
