"""Tracing functionality for the AlignX Telemetry SDK."""

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


class TracingInstrumentor:
    """Instrumentor for tracing functionality."""

    def __init__(self):
        """Initialize the tracing instrumentor."""
        self._tracer_provider = None
        self._instrumented = False

    def instrument(
        self,
        resource: Resource,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        batch_span_processor_queue_size: int = 2048,
    ) -> None:
        """Instrument the tracing system.

        Args:
            resource: Existing OpenTelemetry resource to use.
            otlp_endpoint: OTLP exporter endpoint (defaults to OTEL_EXPORTER_OTLP_ENDPOINT_TRACES env var)
            console_export: Whether to export spans to console (for debugging)
            batch_span_processor_queue_size: The maximum queue size for the BatchSpanProcessor
        """
        if self._instrumented:
            return

        # Get otlp_endpoint from env vars if not provided
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_TRACES")

        # Create provider with resource
        self._tracer_provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint is configured
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter, max_queue_size=batch_span_processor_queue_size
                )
            )

        # Add console exporter for debugging if enabled
        if console_export:
            console_exporter = ConsoleSpanExporter()
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    console_exporter,
                    max_queue_size=batch_span_processor_queue_size,
                )
            )

        # Set as global provider
        trace.set_tracer_provider(self._tracer_provider)

        self._instrumented = True

    def get_tracer(self, name: str) -> trace.Tracer:
        """Get a tracer for the specified name.

        Args:
            name: Name of the tracer

        Returns:
            A tracer instance
        """
        if not self._instrumented:
            raise RuntimeError("Tracing not instrumented. Call instrument() first.")

        return trace.get_tracer(name)

    def shutdown(self) -> None:
        """Shut down the telemetry provider, flushing any pending spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
            self._instrumented = False
