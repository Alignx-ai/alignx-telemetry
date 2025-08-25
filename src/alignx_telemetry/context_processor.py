"""Context processor for AlignX correlation headers.

This module provides functionality to detect and process AlignX correlation headers
for trace linking between the SaaS backend and customer AI agents.
"""

import logging
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.trace import set_span_in_context, SpanContext, TraceFlags
from opentelemetry.context import Context
from opentelemetry.propagate import get_global_textmap

logger = logging.getLogger(__name__)


class AlignXContextProcessor:
    """Processor for AlignX correlation headers and trace context."""

    # AlignX correlation header names
    CORRELATION_HEADERS = {
        "trace_id": "x-alignx-correlation-trace-id",
        "org_id": "x-alignx-org-id",
        "correlation_enabled": "x-alignx-correlation-enabled",
        # Workflow-specific headers
        "workflow_id": "x-alignx-workflow-id",
        "node_name": "x-alignx-node-name",
        "node_sequence": "x-alignx-node-sequence",
        "node_type": "x-alignx-node-type",
    }

    @classmethod
    def extract_correlation_context(
        cls, headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Extract AlignX correlation information from headers.

        Args:
            headers: HTTP headers dictionary (case-insensitive)

        Returns:
            Dictionary with correlation info if found, None otherwise
        """
        # Convert headers to lowercase for case-insensitive lookup
        headers_lower = {k.lower(): v for k, v in headers.items()}

        # Check if correlation is enabled
        correlation_enabled = headers_lower.get(
            cls.CORRELATION_HEADERS["correlation_enabled"]
        )
        if correlation_enabled != "true":
            logger.debug("AlignX correlation not enabled or header not found")
            return None

        # Extract trace ID
        trace_id_header = headers_lower.get(cls.CORRELATION_HEADERS["trace_id"])
        if not trace_id_header:
            logger.debug("AlignX correlation trace ID header not found")
            return None

        # Validate trace ID format (32 hex characters)
        if not cls._is_valid_trace_id(trace_id_header):
            logger.warning(
                f"Invalid AlignX correlation trace ID format: {trace_id_header}"
            )
            return None

        org_id = headers_lower.get(cls.CORRELATION_HEADERS["org_id"], "")

        correlation_info = {
            "trace_id": trace_id_header,
            "org_id": org_id,
            "correlation_enabled": True,
            # Extract workflow context
            "workflow_id": headers_lower.get(
                cls.CORRELATION_HEADERS["workflow_id"], ""
            ),
            "node_name": headers_lower.get(cls.CORRELATION_HEADERS["node_name"], ""),
            "node_sequence": headers_lower.get(
                cls.CORRELATION_HEADERS["node_sequence"], ""
            ),
            "node_type": headers_lower.get(cls.CORRELATION_HEADERS["node_type"], ""),
        }

        logger.info(
            f"Extracted AlignX correlation context: trace_id={trace_id_header}, "
            f"org_id={org_id}, workflow_id={correlation_info['workflow_id']}, "
            f"node={correlation_info['node_name']}, sequence={correlation_info['node_sequence']}"
        )
        return correlation_info

    @classmethod
    def create_trace_context_from_correlation(
        cls, correlation_info: Dict[str, Any], context: Optional[Context] = None
    ) -> Context:
        """Create a trace context from AlignX correlation information.

        This method creates a trace context using the correlation trace ID as the actual
        trace ID for the current request. This enables unified workflow tracing where:
        - One workflow execution = One trace ID
        - Each node execution = One span within that trace
        - Grafana shows hierarchical view: Workflow -> Node1, Node2, Node3, etc.

        Args:
            correlation_info: Correlation info extracted from headers
            context: Existing context to extend (optional)

        Returns:
            Context with the correlated trace ID set as the active trace ID for unified workflow tracing
        """
        if not correlation_info or not correlation_info.get("trace_id"):
            logger.warning("No valid correlation info provided")
            return context or Context()

        trace_id_str = correlation_info["trace_id"]

        try:
            # Convert hex string to int
            trace_id_int = int(trace_id_str, 16)

            # Generate a new span ID for this node execution
            from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

            id_generator = RandomIdGenerator()
            span_id = id_generator.generate_span_id()

            # Create span context with the workflow trace ID
            # This creates a child span within the unified workflow trace
            span_context = SpanContext(
                trace_id=trace_id_int,  # Same trace ID for entire workflow
                span_id=span_id,  # Unique span ID for this node execution
                is_remote=True,  # This comes from remote backend workflow
                trace_flags=TraceFlags(0x01),  # Sampled
            )

            # Create a non-recording span to carry the context
            span = trace.NonRecordingSpan(span_context)

            # Set span in context - this makes all spans in this agent execution
            # part of the unified workflow trace
            new_context = set_span_in_context(span, context or Context())

            logger.info(
                f"✅ Created child span in workflow trace: {trace_id_str} (unified workflow tracing enabled)"
            )
            return new_context

        except (ValueError, OverflowError) as e:
            logger.error(
                f"Failed to create trace context from correlation trace_id {trace_id_str}: {e}"
            )
            return context or Context()

    @classmethod
    def process_incoming_request(
        cls, headers: Dict[str, str], context: Optional[Context] = None
    ) -> Context:
        """Process incoming request headers for AlignX correlation.

        ⚠️  WARNING: This method should only be used for MANUAL header processing.
        For automatic middleware handling, external requests should bypass correlation
        processing entirely to avoid interfering with normal OpenTelemetry instrumentation.

        This is the main entry point for processing incoming requests when manually handling
        correlation headers (e.g., via TelemetryManager.process_correlation_headers()).

        It intelligently handles two types of requests:

        1. **AlignX Backend Requests**: Contains correlation headers → Uses unified workflow tracing
        2. **External/Regular Requests**: No correlation headers → Uses standard OpenTelemetry behavior

        Args:
            headers: HTTP headers from incoming request
            context: Existing context (optional)

        Returns:
            Context with appropriate trace information:
            - AlignX requests: Context with correlated trace ID (unified workflow tracing)
            - Regular requests: Context from standard OpenTelemetry propagation (separate traces)
        """
        # First, try to extract AlignX correlation
        correlation_info = cls.extract_correlation_context(headers)

        if correlation_info:
            # ✅ AlignX Backend Request: Use unified workflow tracing
            logger.info(
                f"🔗 AlignX backend request detected - using unified workflow tracing (trace_id: {correlation_info['trace_id']})"
            )
            return cls.create_trace_context_from_correlation(correlation_info, context)
        else:
            # ✅ External/Regular Request: Use standard OpenTelemetry behavior
            logger.debug(
                "🔄 External request detected - using standard OpenTelemetry propagation (separate trace)"
            )
            propagator = get_global_textmap()
            return propagator.extract(headers, context or Context())

    @classmethod
    def _is_valid_trace_id(cls, trace_id: str) -> bool:
        """Validate trace ID format.

        Args:
            trace_id: Trace ID string to validate

        Returns:
            True if valid W3C trace ID format, False otherwise
        """
        if not trace_id or len(trace_id) != 32:
            return False

        try:
            # Check if it's valid hex
            int(trace_id, 16)
            # Check if it's not all zeros (invalid per W3C spec)
            return trace_id != "00000000000000000000000000000000"
        except ValueError:
            return False


def get_alignx_context_processor() -> AlignXContextProcessor:
    """Get the AlignX context processor instance.

    Returns:
        AlignXContextProcessor instance
    """
    return AlignXContextProcessor()
