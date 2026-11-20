import json
import logging
import socket

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from cfpb_complaint_classifier import telemetry
from cfpb_complaint_classifier.core.logging import JsonFormatter, task_context


def test_json_log_correlates_without_exposing_dsn_or_validation_input():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    formatter = JsonFormatter()
    with (
        task_context("task-1", "batch-7"),
        tracer.start_as_current_span("process") as span,
    ):
        try:
            raise ValueError("private contents of the CSV row")
        except ValueError as exc:
            record = logging.LogRecord(
                "pipeline",
                logging.ERROR,
                __file__,
                1,
                "connect %s",
                ("amqp://user:private-password@rabbitmq",),
                (ValueError, exc, exc.__traceback__),
            )
            line = formatter.format(record)
    provider.shutdown()
    assert "private-password" not in line
    assert "private contents" not in line
    event = json.loads(line)
    assert event["task_id"] == "task-1"
    assert event["batch_id"] == "batch-7"
    assert event["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert event["span_id"] == format(span.get_span_context().span_id, "016x")
    assert "ValueError" in event["stacktrace"]
    assert json.loads(formatter.format(record))["task_id"] is None


def test_inference_buckets_follow_delay(monkeypatch):
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry, "meter", provider.get_meter("test"))
    instrument = telemetry.inference_histogram(0.2)
    instrument.record(0.215)
    data = reader.get_metrics_data()
    metric = data.resource_metrics[0].scope_metrics[0].metrics[0]
    point = metric.data.data_points[0]
    assert len(point.explicit_bounds) == 10
    assert 0.2 in point.explicit_bounds
    assert point.bucket_counts[point.explicit_bounds.index(0.2 * 1.1)] == 1
    provider.shutdown()


def test_skips_missing_collector_and_rediscovers_it(monkeypatch):
    key = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    monkeypatch.setenv("OTEL_DISCOVER_COLLECTOR", "true")
    monkeypatch.setenv(key, "http://collector:4318/v1/traces")

    def missing(*args, **kwargs):
        raise socket.gaierror

    monkeypatch.setattr(socket, "getaddrinfo", missing)
    assert not telemetry.collector_present(key)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(1,)])
    assert telemetry.collector_present(key)


def test_otlp_flush_exports_trace_and_configured_queue_limit(monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    from cfpb_complaint_classifier.core.config import Settings

    received = {}

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()

    with HTTPServer(("127.0.0.1", 0), Receiver) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}"
        )
        monkeypatch.setenv("OTEL_SERVICE_NAME", "api")
        monkeypatch.setenv("OTEL_DISCOVER_COLLECTOR", "true")
        monkeypatch.setattr(telemetry, "_providers", None)
        monkeypatch.setattr(telemetry.trace, "set_tracer_provider", lambda p: None)
        monkeypatch.setattr(
            telemetry.metrics,
            "set_meter_provider",
            lambda p: monkeypatch.setattr(telemetry, "meter", p.get_meter("test")),
        )
        try:
            telemetry.configure(Settings(queue_max_length=17), "api")
            traces, _ = telemetry._providers
            with traces.get_tracer("test").start_as_current_span("publish"):
                pass
            telemetry.flush()
            spans = ExportTraceServiceRequest.FromString(received["/v1/traces"])
            assert spans.resource_spans[0].scope_spans[0].spans[0].name == "publish"
            data = ExportMetricsServiceRequest.FromString(received["/v1/metrics"])
            values = {
                m.name: m
                for r in data.resource_metrics
                for scope in r.scope_metrics
                for m in scope.metrics
            }
            assert values["pipeline_queue_limit"].gauge.data_points[0].as_int == 17
            assert "pipeline_process_resident_bytes" in values
        finally:
            if telemetry.enabled():
                for provider in telemetry._providers:
                    provider.shutdown()
            server.shutdown()
            thread.join()
