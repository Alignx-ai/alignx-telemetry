"""Workflow context detection for unified tracing."""

import logging
from typing import Dict, Any, Optional
from opentelemetry import trace, context
from opentelemetry.sdk.trace import ReadableSpan

from .semantic_conventions import AlignXSemanticConventions

logger = logging.getLogger(__name__)


class WorkflowDetector:
    """Detects when code is running in an AlignX workflow context.

    This detector examines the current OpenTelemetry context to determine if
    the current execution is part of an AlignX workflow. It looks for workflow
    attributes set by the correlation middleware.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def is_workflow_context(self) -> bool:
        """Check if current execution is in a workflow context.

        Returns:
            True if running in workflow context, False otherwise
        """
        try:
            current_context = context.get_current()
            current_span = trace.get_current_span(current_context)

            if not current_span or not hasattr(current_span, "attributes"):
                return False

            attributes = current_span.attributes or {}

            # Check for correlation trace ID (primary indicator)
            correlation_trace_id = attributes.get(
                AlignXSemanticConventions.ALIGNX_CORRELATION_TRACE_ID
            )
            correlation_enabled = attributes.get(
                AlignXSemanticConventions.ALIGNX_CORRELATION_ENABLED
            )

            return bool(correlation_trace_id and correlation_enabled)

        except Exception as e:
            self.logger.debug(f"Error detecting workflow context: {e}")
            return False

    def get_workflow_context(self) -> Optional[Dict[str, Any]]:
        """Extract workflow context information from current span.

        Returns:
            Dictionary with workflow context if available, None otherwise
        """
        try:
            current_context = context.get_current()
            current_span = trace.get_current_span(current_context)

            if not current_span or not hasattr(current_span, "attributes"):
                return None

            attributes = current_span.attributes or {}

            # Check if we're in workflow context
            if not self.is_workflow_context():
                return None

            # Extract workflow information
            workflow_context = {
                "correlation_trace_id": attributes.get(
                    AlignXSemanticConventions.ALIGNX_CORRELATION_TRACE_ID
                ),
                "workflow_id": attributes.get(
                    AlignXSemanticConventions.ALIGNX_WORKFLOW_ID
                ),
                "node_name": attributes.get(
                    AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_NAME
                ),
                "node_sequence": attributes.get(
                    AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_SEQUENCE
                ),
                "node_type": attributes.get(
                    AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_TYPE
                ),
                "org_id": attributes.get(
                    AlignXSemanticConventions.ALIGNX_CORRELATION_ORG_ID
                ),
                "correlation_enabled": attributes.get(
                    AlignXSemanticConventions.ALIGNX_CORRELATION_ENABLED, False
                ),
            }

            self.logger.debug(f"Extracted workflow context: {workflow_context}")
            return workflow_context

        except Exception as e:
            self.logger.debug(f"Error extracting workflow context: {e}")
            return None

    def add_workflow_attributes_to_span(self, span: ReadableSpan) -> None:
        """Add workflow attributes to a span if in workflow context.

        Args:
            span: Span to enhance with workflow attributes
        """
        if not span or not hasattr(span, "set_attribute"):
            return

        workflow_context = self.get_workflow_context()
        if not workflow_context:
            # Not in workflow context - mark as external
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_CORRELATION_ENABLED, False
            )
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_REQUEST_TYPE,
                AlignXSemanticConventions.REQUEST_TYPE_EXTERNAL,
            )
            return

        # In workflow context - add all workflow attributes
        span.set_attribute(AlignXSemanticConventions.ALIGNX_CORRELATION_ENABLED, True)
        span.set_attribute(
            AlignXSemanticConventions.ALIGNX_REQUEST_TYPE,
            AlignXSemanticConventions.REQUEST_TYPE_WORKFLOW_NODE,
        )

        if workflow_context.get("correlation_trace_id"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_CORRELATION_TRACE_ID,
                workflow_context["correlation_trace_id"],
            )

        if workflow_context.get("workflow_id"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_WORKFLOW_ID,
                workflow_context["workflow_id"],
            )

        if workflow_context.get("node_name"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_NAME,
                workflow_context["node_name"],
            )

        if workflow_context.get("node_sequence"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_SEQUENCE,
                workflow_context["node_sequence"],
            )

        if workflow_context.get("node_type"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_WORKFLOW_NODE_TYPE,
                workflow_context["node_type"],
            )

        if workflow_context.get("org_id"):
            span.set_attribute(
                AlignXSemanticConventions.ALIGNX_CORRELATION_ORG_ID,
                workflow_context["org_id"],
            )

        span.set_attribute(
            AlignXSemanticConventions.ALIGNX_CORRELATION_SOURCE,
            AlignXSemanticConventions.CORRELATION_SOURCE_BACKEND_WORKFLOW,
        )

        self.logger.debug(f"Added workflow attributes to span: {span.name}")

    def get_service_metadata(self, telemetry_manager=None) -> Dict[str, str]:
        """Get service metadata for dashboard attribution.

        Args:
            telemetry_manager: Optional telemetry manager instance

        Returns:
            Dictionary with service name and environment
        """
        try:
            if telemetry_manager:
                metrics_config = telemetry_manager.get_metrics_config()
                if metrics_config:
                    return {
                        "service_name": metrics_config.service_name
                        or "unknown-service",
                        "environment": metrics_config.environment or "unknown",
                    }
        except Exception as e:
            self.logger.debug(f"Failed to get service metadata: {e}")

        return {"service_name": "unknown-service", "environment": "unknown"}


# Global instance for easy access
_global_detector = None


def get_workflow_detector() -> WorkflowDetector:
    """Get the global workflow detector instance.

    Returns:
        Global WorkflowDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = WorkflowDetector()
    return _global_detector
