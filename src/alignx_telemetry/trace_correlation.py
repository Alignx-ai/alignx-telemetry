"""AlignX trace correlation for unified workflow tracing.

This module handles correlation of traces for requests originating from AlignX backend
workflow engine. When the backend generates a pre_generated_trace_id, all follow-up
requests in that particular workflow will show under a single trace.

CRITICAL: This module only adds span attributes for correlation. Context manipulation
is handled by the middleware to avoid interfering with OpenTelemetry span hierarchy.
"""

import logging
from typing import Any, Dict, Optional, Callable

from alignx_telemetry.context_processor import AlignXContextProcessor

# Logger for trace correlation
logger = logging.getLogger(__name__)


class AlignXTraceCorrelator:
    """Handles trace correlation for AlignX workflow requests."""

    def __init__(self):
        self._context_processor = AlignXContextProcessor()

    def create_correlation_request_hook(
        self, original_hook: Optional[Callable] = None
    ) -> Callable:
        """Create a request hook that automatically processes AlignX correlation headers.

        Args:
            original_hook: Original request hook to chain (optional)

        Returns:
            Enhanced request hook function
        """

        def correlation_request_hook(span, scope, **kwargs):
            """Request hook that intelligently handles AlignX correlation vs regular requests.

            CRITICAL FIX: Only adds span attributes for correlation. Does NOT manipulate
            context to avoid interfering with natural OpenTelemetry span hierarchy.

            Behavior:
            - AlignX backend requests (with correlation headers) → Add correlation attributes
            - External/regular requests (no correlation headers) → Add external attributes
            - Context manipulation is handled by middleware, not here
            """
            try:
                # Extract headers from different scope formats
                headers = self._extract_headers_from_scope(scope, **kwargs)

                if headers:
                    # Check if this was an AlignX backend request with correlation
                    correlation_info = (
                        self._context_processor.extract_correlation_context(headers)
                    )
                    if correlation_info:
                        # 🔗 AlignX Backend Request: Add correlation attributes only
                        logger.debug(
                            f"✅ AlignX workflow request - adding correlation attributes for trace: {correlation_info['trace_id']}"
                        )

                        # CRITICAL FIX: Only add span attributes, do NOT manipulate context!
                        # Context manipulation is handled by middleware to preserve span hierarchy

                        # Add metadata as span attributes for debugging/monitoring
                        span.set_attribute("alignx.correlation.enabled", True)
                        span.set_attribute(
                            "alignx.correlation.org_id",
                            correlation_info.get("org_id", ""),
                        )
                        span.set_attribute(
                            "alignx.correlation.source", "backend_workflow"
                        )
                        span.set_attribute("alignx.request.type", "workflow_node")
                        # Add the correlation trace ID for easy filtering in Grafana
                        span.set_attribute(
                            "alignx.correlation.trace_id", correlation_info["trace_id"]
                        )

                        # Add workflow context attributes
                        if correlation_info.get("workflow_id"):
                            span.set_attribute(
                                "alignx.workflow.id", correlation_info["workflow_id"]
                            )
                        if correlation_info.get("node_name"):
                            span.set_attribute(
                                "alignx.workflow.node_name",
                                correlation_info["node_name"],
                            )
                        if correlation_info.get("node_sequence"):
                            span.set_attribute(
                                "alignx.workflow.node_sequence",
                                correlation_info["node_sequence"],
                            )
                        if correlation_info.get("node_type"):
                            span.set_attribute(
                                "alignx.workflow.node_type",
                                correlation_info["node_type"],
                            )
                    else:
                        # 🔄 External/Regular Request: Add external attributes
                        logger.debug(
                            "🔄 External request - adding external request attributes"
                        )
                        span.set_attribute("alignx.correlation.enabled", False)
                        span.set_attribute("alignx.request.type", "external")

                # Add mandatory OpenTelemetry semantic conventions for dashboard queries
                try:
                    # Try to get telemetry manager instance
                    from alignx_telemetry.telemetry import _telemetry_manager

                    metrics_config = (
                        _telemetry_manager.get_metrics_config()
                        if _telemetry_manager
                        else None
                    )
                    if metrics_config:
                        span.set_attribute(
                            "gen_ai.application_name",
                            metrics_config.service_name or "unknown-service",
                        )
                        span.set_attribute(
                            "gen_ai.environment",
                            metrics_config.environment or "unknown",
                        )
                    else:
                        span.set_attribute("gen_ai.application_name", "unknown-service")
                        span.set_attribute("gen_ai.environment", "unknown")
                except Exception as e:
                    logger.debug(f"Failed to get metrics config: {e}")
                    span.set_attribute("gen_ai.application_name", "unknown-service")
                    span.set_attribute("gen_ai.environment", "unknown")

                # Call original hook if provided
                if original_hook:
                    original_hook(span, scope, **kwargs)

            except Exception as e:
                logger.error(
                    f"Error in AlignX correlation processing: {e}", exc_info=True
                )
                # Call original hook even if our processing failed
                if original_hook:
                    try:
                        original_hook(span, scope, **kwargs)
                    except Exception:
                        pass

        return correlation_request_hook

    def create_correlation_response_hook(
        self, original_hook: Optional[Callable] = None
    ) -> Callable:
        """Create a response hook that adds correlation metadata.

        Args:
            original_hook: Original response hook to chain (optional)

        Returns:
            Enhanced response hook function
        """

        def correlation_response_hook(span, scope, response, **kwargs):
            """Response hook that adds correlation metadata to spans."""
            try:
                # Check if this span has correlation info
                if span.get_attribute("alignx.correlation.enabled"):
                    # Add response metadata for workflow requests
                    span.set_attribute("alignx.workflow.response_processed", True)

                    # Add status code if available
                    if hasattr(response, "status_code"):
                        span.set_attribute(
                            "alignx.workflow.status_code", response.status_code
                        )

                # Call original hook if provided
                if original_hook:
                    original_hook(span, scope, response, **kwargs)

            except Exception as e:
                logger.error(
                    f"Error in AlignX correlation response processing: {e}",
                    exc_info=True,
                )
                # Call original hook even if our processing failed
                if original_hook:
                    try:
                        original_hook(span, scope, response, **kwargs)
                    except Exception:
                        pass

        return correlation_response_hook

    def _extract_headers_from_scope(self, scope, **kwargs) -> Dict[str, str]:
        """Extract headers from various scope formats.

        Args:
            scope: ASGI scope or similar context
            **kwargs: Additional context (e.g., request object)

        Returns:
            Dictionary of headers (case-insensitive)
        """
        headers = {}

        # Try ASGI scope format (FastAPI, Starlette)
        if isinstance(scope, dict) and "headers" in scope:
            for header_name, header_value in scope["headers"]:
                if isinstance(header_name, bytes):
                    header_name = header_name.decode("latin-1")
                if isinstance(header_value, bytes):
                    header_value = header_value.decode("latin-1")
                headers[header_name.lower()] = header_value

        # Try Flask request format
        elif hasattr(scope, "headers"):
            for header_name, header_value in scope.headers.items():
                headers[header_name.lower()] = header_value

        # Try extracting from kwargs (request object)
        request = kwargs.get("request")
        if request and hasattr(request, "headers"):
            for header_name, header_value in request.headers.items():
                headers[header_name.lower()] = header_value

        logger.debug(f"🔍 Extracted headers: {list(headers.keys())}")
        return headers
