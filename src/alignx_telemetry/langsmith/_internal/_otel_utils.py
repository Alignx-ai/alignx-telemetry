"""OpenTelemetry utilities for AlignX LangSmith integration."""

from __future__ import annotations

from uuid import UUID


def get_otel_trace_id_from_uuid(uuid_val: UUID) -> int:
    """Get OpenTelemetry trace ID as integer from UUID.

    Args:
        uuid_val: The UUID to convert.

    Returns:
        Integer representation of the trace ID.
    """
    trace_id_hex = uuid_val.hex
    return int(trace_id_hex, 16)


def get_otel_span_id_from_uuid(uuid_val: UUID) -> int:
    """Get OpenTelemetry span ID as integer from UUID.

    Args:
        uuid_val: The UUID to convert.

    Returns:
        Integer representation of the span ID.
    """
    uuid_bytes = uuid_val.bytes
    span_id_bytes = uuid_bytes[:8]
    span_id_hex = span_id_bytes.hex()
    return int(span_id_hex, 16)


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    try:
        import opentelemetry

        return True
    except ImportError:
        return False


def create_otel_span_from_run_tree(run_tree, tracer=None):
    """Create an OpenTelemetry span from a RunTree.

    Args:
        run_tree: The RunTree to create a span from.
        tracer: Optional OpenTelemetry tracer instance.

    Returns:
        OpenTelemetry span instance or None if OTel is not available.
    """
    if not is_otel_available():
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        if tracer is None:
            tracer = trace.get_tracer(__name__)

        # Convert RunTree attributes to OTel span attributes
        attributes = run_tree.to_opentelemetry_span_attributes()

        # Create the span
        span = tracer.start_span(name=run_tree.name, attributes=attributes)

        # Set trace and span IDs to match RunTree
        if hasattr(span, "get_span_context"):
            span_context = span.get_span_context()
            # Note: This is mainly for correlation; actual IDs are managed by OTel

        return span

    except Exception:
        # Silently fail if OpenTelemetry setup fails
        return None


def finalize_otel_span(span, run_tree):
    """Finalize an OpenTelemetry span with RunTree data.

    Args:
        span: The OpenTelemetry span to finalize.
        run_tree: The RunTree with final data.
    """
    if span is None or not is_otel_available():
        return

    try:
        from opentelemetry.trace import Status, StatusCode

        # Add final attributes
        if run_tree.outputs:
            span.set_attribute("langsmith.run.outputs", run_tree.outputs)

        if run_tree.error:
            span.set_status(Status(StatusCode.ERROR, run_tree.error))
            span.set_attribute("langsmith.run.error", run_tree.error)
        else:
            span.set_status(Status(StatusCode.OK))

        # End the span
        span.end()

    except Exception:
        # Silently fail
        pass
