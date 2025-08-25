"""Base provider instrumentation infrastructure for AlignX.

This module provides the abstract base class and core patterns for implementing
provider-level LLM instrumentation, inspired by OpenInference and OpenLLMetry patterns.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

# Avoid circular import - use string annotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alignx_telemetry.telemetry import TelemetryManager
from alignx_telemetry.unified_tracing import (
    WorkflowDetector,
    MetricsBridge,
    AlignXSemanticConventions,
)
from alignx_telemetry.observability.metrics import (
    AlignXEnhancedProviderMetrics,
    record_enhanced_llm_metrics,
    ErrorType,
)

logger = logging.getLogger(__name__)


@dataclass
class AlignXProviderCallData:
    """Standardized call data structure for provider instrumentation."""

    # Request information
    method: str  # "chat.completions.create", "embeddings.create"
    model: str  # "gpt-4", "claude-3-sonnet"
    inputs: Dict[str, Any]  # Request parameters

    # Response information (populated after call)
    outputs: Optional[Dict[str, Any]] = None
    response_metadata: Optional[Dict[str, Any]] = None

    # Timing information
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    time_to_first_token: Optional[float] = None

    # Error information
    error: Optional[Exception] = None
    error_type: Optional[str] = None

    # Provider-specific metadata
    provider_metadata: Optional[Dict[str, Any]] = None


class AlignXBaseProviderInstrumentation(ABC):
    """Abstract base class for all AlignX provider instrumentations.

    This class defines the interface and common patterns for instrumenting
    LLM providers at the lowest level to capture all usage patterns.

    Inspired by:
    - OpenInference BaseInstrumentor pattern
    - OpenLLMetry instrumentation wrappers
    - Opik provider tracking approach
    """

    def __init__(self, telemetry_manager: "TelemetryManager"):
        """Initialize provider instrumentation.

        Args:
            telemetry_manager: AlignX telemetry manager instance
        """
        self.telemetry_manager = telemetry_manager
        self.tracer = telemetry_manager.tracer
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Track instrumentation state
        self._instrumented = False
        self._original_methods = {}

        # Import metrics emitter
        from .metrics import AlignXMetricsEmitter

        self.metrics_emitter = AlignXMetricsEmitter(telemetry_manager)

        # Initialize unified tracing components
        self.workflow_detector = WorkflowDetector()
        self.metrics_bridge = MetricsBridge()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def instrumentation_targets(self) -> Dict[str, str]:
        """Dictionary of module.method -> wrapper_method mappings to instrument.

        Example:
            {
                "openai.OpenAI.request": "_wrap_openai_request",
                "openai.AsyncOpenAI.request": "_wrap_async_openai_request"
            }
        """
        pass

    def instrument(self) -> bool:
        """Instrument the provider SDK.

        Returns:
            bool: True if instrumentation succeeded, False otherwise
        """
        if self._instrumented:
            self.logger.debug(f"{self.provider_name} provider already instrumented")
            return True

        try:
            self.logger.info(f"Instrumenting {self.provider_name} provider...")

            # Perform provider-specific instrumentation
            success = self._perform_instrumentation()

            if success:
                self._instrumented = True
                self.logger.info(
                    f"Successfully instrumented {self.provider_name} provider"
                )
            else:
                self.logger.error(f"Failed to instrument {self.provider_name} provider")

            return success

        except Exception as e:
            self.logger.error(
                f"Error instrumenting {self.provider_name} provider: {e}", exc_info=True
            )
            return False

    def uninstrument(self) -> bool:
        """Remove instrumentation from the provider SDK.

        Returns:
            bool: True if uninstrumentation succeeded, False otherwise
        """
        if not self._instrumented:
            self.logger.debug(f"{self.provider_name} provider not instrumented")
            return True

        try:
            self.logger.info(f"Uninstrumenting {self.provider_name} provider...")

            # Restore original methods
            success = self._restore_original_methods()

            if success:
                self._instrumented = False
                self._original_methods.clear()
                self.logger.info(
                    f"Successfully uninstrumented {self.provider_name} provider"
                )
            else:
                self.logger.error(
                    f"Failed to uninstrument {self.provider_name} provider"
                )

            return success

        except Exception as e:
            self.logger.error(
                f"Error uninstrumenting {self.provider_name} provider: {e}",
                exc_info=True,
            )
            return False

    @abstractmethod
    def _perform_instrumentation(self) -> bool:
        """Perform the actual provider instrumentation.

        This method should:
        1. Import the provider SDK
        2. Wrap the appropriate methods using wrapt
        3. Store original methods for restoration

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def extract_call_data(
        self, method: str, args: tuple, kwargs: dict
    ) -> AlignXProviderCallData:
        """Extract call data from method arguments.

        Args:
            method: Method name being called
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            AlignXProviderCallData: Extracted call data
        """
        pass

    @abstractmethod
    def extract_response_data(
        self, call_data: AlignXProviderCallData, response: Any
    ) -> None:
        """Extract response data and update call_data.

        Args:
            call_data: Call data to update
            response: Provider response object
        """
        pass

    @abstractmethod
    def calculate_metrics(self, call_data: AlignXProviderCallData) -> Any:
        """Calculate metrics from call data.

        Args:
            call_data: Complete call data

        Returns:
            AlignXProviderMetrics: Calculated metrics (import handled by concrete implementations)
        """
        pass

    def create_provider_span(self, call_data: AlignXProviderCallData) -> trace.Span:
        """Create an OpenTelemetry span for the provider call.

        Args:
            call_data: Call data for the span

        Returns:
            trace.Span: Created span
        """
        # Import context here to avoid circular imports
        from .context import AlignXFrameworkContext

        # Get framework context if available
        framework_context = AlignXFrameworkContext.get_current_context()

        # Create span name
        span_name = f"{self.provider_name}.{call_data.method}"

        # Create span with standard attributes
        span = self.tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes={
                AlignXSemanticConventions.GEN_AI_SYSTEM: self.provider_name,
                AlignXSemanticConventions.GEN_AI_REQUEST_MODEL: call_data.model,
                "gen_ai.operation.name": call_data.method,
                AlignXSemanticConventions.ALIGNX_PROVIDER: self.provider_name,
                "telemetry.sdk.name": "alignx",
            },
        )

        # Add workflow context using the unified tracing detector
        self.workflow_detector.add_workflow_attributes_to_span(span)

        # Add service metadata for dashboard queries
        service_metadata = self.workflow_detector.get_service_metadata(
            self.telemetry_manager
        )
        span.set_attribute(
            AlignXSemanticConventions.GEN_AI_APPLICATION_NAME,
            service_metadata["service_name"],
        )
        span.set_attribute(
            AlignXSemanticConventions.GEN_AI_ENVIRONMENT,
            service_metadata["environment"],
        )

        # Add framework context if available (for framework-specific context)
        if framework_context.get("framework"):
            span.set_attribute("alignx.framework", framework_context["framework"])
        if framework_context.get("workflow_id"):
            span.set_attribute("alignx.workflow_id", framework_context["workflow_id"])
        if framework_context.get("node_name"):
            span.set_attribute("alignx.node_name", framework_context["node_name"])

        return span

    def finalize_span(
        self, span: trace.Span, call_data: AlignXProviderCallData
    ) -> None:
        """Finalize span with response data and status.

        Args:
            span: Span to finalize
            call_data: Complete call data
        """
        try:
            if call_data.error:
                # Set error status
                span.set_status(Status(StatusCode.ERROR, str(call_data.error)))
                span.set_attribute(
                    "error.type", call_data.error_type or type(call_data.error).__name__
                )
                span.record_exception(call_data.error)
            else:
                # Set success status
                span.set_status(Status(StatusCode.OK))

                # Add response attributes
                if call_data.outputs:
                    # Add any provider-specific response attributes
                    self._add_response_attributes(span, call_data)

            # Record enhanced metrics (new comprehensive system)
            self._record_enhanced_metrics(span, call_data)

            # Record metrics using the bridge (connects to existing AI metrics system)
            self._record_metrics_via_bridge(span, call_data)

        except Exception as e:
            self.logger.warning(f"Error finalizing span: {e}")
        finally:
            span.end()

    def _record_metrics_via_bridge(
        self, span: trace.Span, call_data: AlignXProviderCallData
    ) -> None:
        """Record metrics via the metrics bridge to existing AI metrics system.

        Args:
            span: OpenTelemetry span with context
            call_data: Complete call data
        """
        try:
            # Calculate provider metrics
            provider_metrics = self.calculate_metrics(call_data)

            # Record via metrics bridge (connects to existing ai_metrics.record_llm_request)
            self.metrics_bridge.record_provider_metrics_from_span(
                telemetry_manager=self.telemetry_manager,
                span=span,
                provider_metrics=provider_metrics,
            )

        except Exception as e:
            self.logger.debug(f"Error recording metrics via bridge: {e}")

    def _record_enhanced_metrics(
        self, span: trace.Span, call_data: AlignXProviderCallData
    ) -> None:
        """Record enhanced metrics with industry-standard observability data."""
        try:
            # Calculate basic provider metrics
            provider_metrics = self.calculate_metrics(call_data)

            # Extract timing data from span
            span_start_time = getattr(span, "_start_time", None)
            span_end_time = getattr(span, "_end_time", None)
            total_latency_ms = 0.0

            if span_start_time and span_end_time:
                total_latency_ms = (
                    span_end_time - span_start_time
                ) / 1_000_000  # Convert nanoseconds to milliseconds

            # Get service metadata from workflow detector
            service_metadata = self.workflow_detector.get_service_metadata(
                self.telemetry_manager
            )

            # Create enhanced metrics data
            enhanced_metrics = AlignXEnhancedProviderMetrics(
                # Basic identification
                provider=self.provider_name,
                model=call_data.model or "unknown",
                operation=call_data.method or "unknown",
                # Core usage metrics
                input_tokens=getattr(provider_metrics, "input_tokens", 0),
                output_tokens=getattr(provider_metrics, "output_tokens", 0),
                total_tokens=getattr(provider_metrics, "total_tokens", 0),
                cost_usd=getattr(provider_metrics, "cost_usd", 0.0),
                # Latency metrics
                total_latency_ms=total_latency_ms,
                # Extract streaming info from call data
                is_streaming=getattr(call_data, "is_streaming", False),
                time_to_first_token_ms=getattr(
                    call_data, "time_to_first_token_ms", None
                ),
                tokens_per_second=getattr(call_data, "tokens_per_second", None),
                stream_chunks_count=getattr(call_data, "stream_chunks", 0),
                # Cost optimization
                cost_per_request=getattr(provider_metrics, "cost_usd", 0.0),
                # Quality metrics
                success=not bool(call_data.error),
                error_type=(
                    self._map_error_to_type(call_data.error)
                    if call_data.error
                    else None
                ),
                # Service context
                application_name=service_metadata.get("service_name", "unknown"),
                environment=service_metadata.get("environment", "unknown"),
                service_name=service_metadata.get("service_name", "unknown"),
                # Additional metadata
                metadata={
                    "provider_name": self.provider_name,
                    "span_id": span.get_span_context().span_id,
                    "trace_id": span.get_span_context().trace_id,
                },
            )

            # Record the enhanced metrics
            record_enhanced_llm_metrics(enhanced_metrics)

        except Exception as e:
            self.logger.debug(f"Error recording enhanced metrics: {e}")

    def _map_error_to_type(self, error: Exception) -> ErrorType:
        """Map exception to standardized error type."""
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()

        if "timeout" in error_name or "timeout" in error_message:
            return ErrorType.TIMEOUT
        elif "rate" in error_message and "limit" in error_message:
            return ErrorType.RATE_LIMIT
        elif "auth" in error_name or "unauthorized" in error_message:
            return ErrorType.AUTHENTICATION
        elif "quota" in error_message or "exceeded" in error_message:
            return ErrorType.QUOTA_EXCEEDED
        elif "context" in error_message and "length" in error_message:
            return ErrorType.CONTEXT_LENGTH
        elif "model" in error_message and (
            "unavailable" in error_message or "not found" in error_message
        ):
            return ErrorType.MODEL_UNAVAILABLE
        elif "network" in error_name or "connection" in error_message:
            return ErrorType.NETWORK
        elif "validation" in error_name or "invalid" in error_message:
            return ErrorType.VALIDATION
        else:
            return ErrorType.UNKNOWN

    def _add_response_attributes(
        self, span: trace.Span, call_data: AlignXProviderCallData
    ) -> None:
        """Add provider-specific response attributes to span.

        Override this method in provider implementations to add specific attributes.

        Args:
            span: Span to add attributes to
            call_data: Call data with response information
        """
        pass

    def _restore_original_methods(self) -> bool:
        """Restore original methods that were wrapped.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for target, original_method in self._original_methods.items():
                # Parse module and method
                module_path, method_name = target.rsplit(".", 1)

                # Import module and restore method
                module = self._import_module(module_path)
                if module:
                    setattr(module, method_name, original_method)

            return True
        except Exception as e:
            self.logger.error(f"Error restoring original methods: {e}")
            return False

    def _import_module(self, module_path: str) -> Optional[Any]:
        """Safely import a module.

        Args:
            module_path: Module path to import

        Returns:
            Module object or None if import fails
        """
        try:
            from importlib import import_module

            return import_module(module_path)
        except ImportError as e:
            self.logger.debug(f"Could not import {module_path}: {e}")
            return None

    def should_suppress_instrumentation(self) -> bool:
        """Check if instrumentation should be suppressed.

        This implements the suppression pattern from OpenLLMetry to avoid
        duplicate spans when multiple instrumentation layers are active.

        Returns:
            bool: True if instrumentation should be suppressed
        """
        from .context import AlignXFrameworkContext

        return AlignXFrameworkContext.should_suppress_provider_instrumentation(
            self.provider_name
        )

    @property
    def is_instrumented(self) -> bool:
        """Check if provider is currently instrumented."""
        return self._instrumented
