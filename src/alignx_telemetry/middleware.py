"""ASGI middleware for AlignX trace correlation.

This middleware processes AlignX correlation headers before OpenTelemetry instrumentation
runs, ensuring that spans are created with the correct correlated trace ID.
"""

import logging
from typing import Dict, Any, Callable

# Optional starlette import for ASGI middleware
try:
    from starlette.types import ASGIApp, Receive, Scope, Send

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    # Type hints for when starlette is not available
    ASGIApp = Any
    Receive = Callable
    Scope = Dict[str, Any]
    Send = Callable

from opentelemetry.context import attach, detach
from alignx_telemetry.context_processor import AlignXContextProcessor

logger = logging.getLogger(__name__)


class AlignXCorrelationMiddleware:
    """ASGI middleware that processes AlignX correlation headers and sets trace context.

    This middleware must be added BEFORE any OpenTelemetry instrumentation middleware
    to ensure that correlation context is properly set before spans are created.

    CRITICAL: This middleware only interferes with tracing when correlation headers
    are present. For external/regular requests, it completely delegates to OpenTelemetry.
    """

    def __init__(self, app: ASGIApp):
        """Initialize the correlation middleware.

        Args:
            app: The ASGI application to wrap
        """
        if not STARLETTE_AVAILABLE:
            raise ImportError(
                "Starlette is required for ASGI middleware. Install with: pip install starlette"
            )

        self.app = app
        self.context_processor = AlignXContextProcessor()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the ASGI request and handle correlation.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        # Only process HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract headers from ASGI scope
        headers = self._extract_headers_from_scope(scope)

        # Check for AlignX correlation
        correlation_info = self.context_processor.extract_correlation_context(headers)

        if correlation_info and correlation_info.get("trace_id"):
            # AlignX workflow request - set correlation context BEFORE OpenTelemetry runs
            logger.debug(
                f"🔗 Processing AlignX workflow request with trace_id: {correlation_info['trace_id']}"
            )
            await self._handle_workflow_request(scope, receive, send, correlation_info)
        else:
            # Let OpenTelemetry instrumentation handle this completely
            logger.debug(
                "🔄 External request detected - bypassing AlignX correlation (normal OTel instrumentation)"
            )
            await self.app(scope, receive, send)

    async def _handle_workflow_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        correlation_info: Dict[str, Any],
    ) -> None:
        """Handle AlignX workflow request with enhanced span hierarchy.

        CRITICAL FIX: Follow the old SDK pattern - create a workflow node parent span
        with all the workflow attributes. All child spans will inherit this context.
        """
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        # Get tracer
        tracer = trace.get_tracer(__name__)

        # Create workflow node parent span
        node_name = correlation_info.get("node_name", "unknown-node")
        node_sequence = correlation_info.get("node_sequence", "0")
        workflow_id = correlation_info.get("workflow_id", "unknown-workflow")

        span_name = f"🔀 Workflow Node: {node_name}"
        if node_sequence != "0":
            span_name = f"🔀 [{node_sequence}] {node_name}"

        # Set up correlation context first
        correlation_context = (
            self.context_processor.create_trace_context_from_correlation(
                correlation_info
            )
        )

        # Start workflow node span - THIS IS THE KEY DIFFERENCE FROM CURRENT APPROACH
        with tracer.start_as_current_span(
            span_name, context=correlation_context, kind=trace.SpanKind.SERVER
        ) as workflow_span:
            try:
                # Add comprehensive workflow attributes - EXACTLY like the old SDK
                if workflow_span.is_recording():
                    # Core workflow attributes
                    workflow_span.set_attribute("alignx.workflow.id", workflow_id)
                    workflow_span.set_attribute("alignx.workflow.node.name", node_name)
                    workflow_span.set_attribute(
                        "alignx.workflow.node.sequence", node_sequence
                    )
                    workflow_span.set_attribute(
                        "alignx.workflow.node.type",
                        correlation_info.get("node_type", "agent"),
                    )

                    # Correlation attributes
                    workflow_span.set_attribute("alignx.correlation.enabled", True)
                    workflow_span.set_attribute(
                        "alignx.correlation.org_id", correlation_info.get("org_id", "")
                    )
                    workflow_span.set_attribute(
                        "alignx.correlation.source", "backend_workflow"
                    )
                    workflow_span.set_attribute(
                        "alignx.correlation.trace_id", correlation_info["trace_id"]
                    )

                    # Mandatory OpenTelemetry semantic conventions for dashboard queries
                    # Get service name and environment from telemetry manager
                    try:
                        from alignx_telemetry.telemetry import _telemetry_manager

                        metrics_config = _telemetry_manager.get_metrics_config()
                        if metrics_config:
                            workflow_span.set_attribute(
                                "gen_ai.application_name",
                                metrics_config.service_name or "unknown-service",
                            )
                            workflow_span.set_attribute(
                                "gen_ai.environment",
                                metrics_config.environment or "unknown",
                            )
                        else:
                            workflow_span.set_attribute(
                                "gen_ai.application_name", "unknown-service"
                            )
                            workflow_span.set_attribute("gen_ai.environment", "unknown")
                    except Exception as e:
                        logger.debug(f"Failed to get telemetry config: {e}")
                        workflow_span.set_attribute(
                            "gen_ai.application_name", "unknown-service"
                        )
                        workflow_span.set_attribute("gen_ai.environment", "unknown")

                    # Visual helpers for Grafana
                    workflow_span.set_attribute(
                        "workflow.node.display_name", f"[{node_sequence}] {node_name}"
                    )
                    workflow_span.set_attribute(
                        "service.workflow.context", f"{workflow_id}::{node_name}"
                    )

                # Process the request within this workflow context
                current_context = trace.set_span_in_context(workflow_span)
                token = attach(current_context)

                try:
                    await self.app(scope, receive, send)
                    # Mark as successful
                    workflow_span.set_status(
                        Status(StatusCode.OK, "Workflow node completed successfully")
                    )
                except Exception as e:
                    # Mark as error and re-raise
                    workflow_span.set_status(
                        Status(StatusCode.ERROR, f"Workflow node failed: {str(e)}")
                    )
                    workflow_span.record_exception(e)
                    raise
                finally:
                    detach(token)

            except Exception as e:
                logger.error(f"Error in workflow node processing: {e}", exc_info=True)
                raise

    def _extract_headers_from_scope(self, scope: Scope) -> Dict[str, str]:
        """Extract headers from ASGI scope.

        Args:
            scope: ASGI scope containing headers

        Returns:
            Dictionary of headers (case-insensitive)
        """
        headers = {}

        if "headers" in scope:
            for header_name, header_value in scope["headers"]:
                if isinstance(header_name, bytes):
                    header_name = header_name.decode("latin-1")
                if isinstance(header_value, bytes):
                    header_value = header_value.decode("latin-1")
                headers[header_name.lower()] = header_value

        return headers


def add_alignx_correlation_middleware(app: ASGIApp) -> ASGIApp:
    """Add AlignX correlation middleware to an ASGI application.

    This is a convenience function to add the correlation middleware.
    The middleware should be added BEFORE any OpenTelemetry instrumentation.

    Args:
        app: The ASGI application

    Returns:
        The wrapped application with correlation middleware
    """
    return AlignXCorrelationMiddleware(app)
