"""AlignX LLM Span Processor for unified tracing."""

import logging
import threading
from typing import Optional
from weakref import WeakSet

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import StatusCode

from .config import AlignXLLMConfig
from .workflow_detector import WorkflowDetector
from .semantic_conventions import AlignXSemanticConventions
from .metrics_bridge import MetricsBridge

logger = logging.getLogger(__name__)


class AlignXLLMSpanProcessor(SpanProcessor):
    """Enhanced span processor that adds AlignX workflow context to LLM spans.

    This processor is the critical bridge between the AlignX workflow system
    and LLM operations. It:

    1. Detects when running in workflow context
    2. Adds workflow attributes to LLM spans for correlation
    3. Records metrics for dashboard visualization
    4. Handles cost calculation and tracking

    This ensures that LLM operations are properly linked to their originating
    workflow executions for unified tracing in Grafana.
    """

    def __init__(
        self,
        telemetry_manager=None,
        config: Optional[AlignXLLMConfig] = None,
    ):
        """Initialize the AlignX LLM span processor.

        Args:
            telemetry_manager: AlignX telemetry manager instance
            config: Configuration for LLM processing
        """
        self.telemetry_manager = telemetry_manager
        self.config = config or AlignXLLMConfig()
        self.workflow_detector = WorkflowDetector()
        self.metrics_bridge = MetricsBridge()
        self.logger = logging.getLogger(__name__)
        self._processed_spans = (
            WeakSet() if self.config.enable_memory_management else set()
        )
        self._lock = threading.Lock()

    def on_start(self, span: ReadableSpan, parent_context=None):
        """Called when a span starts - add workflow attributes if in workflow context.

        Args:
            span: The span that is starting
            parent_context: Parent context (optional)
        """
        if not self.config.enabled:
            return

        try:
            # Add workflow attributes using the detector
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

            # Call pre-process hook if configured
            if self.config.pre_process_hook:
                self.config.pre_process_hook(span, "start")

        except Exception as e:
            if self.config.log_errors:
                self.logger.debug(f"Error in span start processing: {e}")

    def on_end(self, span: ReadableSpan):
        """Called when a span ends - record metrics if this is an LLM span.

        Args:
            span: The span that is ending
        """
        if not self.config.enabled:
            return

        # Prevent duplicate processing
        with self._lock:
            if span in self._processed_spans:
                return
            self._processed_spans.add(span)

        try:
            # Only process spans that look like LLM operations
            if not self._is_llm_span(span):
                return

            self.logger.debug(f"Processing LLM span: {span.name}")

            # Process the span for metrics and cost calculation
            self._process_llm_span(span)

            # Call post-process hook if configured
            if self.config.post_process_hook:
                self.config.post_process_hook(span, "end")

        except Exception as e:
            if self.config.log_errors:
                self.logger.error(f"Error processing LLM span: {e}", exc_info=True)
            if not self.config.fail_silently:
                raise

    def _is_llm_span(self, span: ReadableSpan) -> bool:
        """Check if span represents an LLM operation.

        Args:
            span: Span to check

        Returns:
            True if this is an LLM span, False otherwise
        """
        if not span.attributes:
            return False

        attributes = span.attributes

        # Check for standard LLM semantic conventions
        gen_ai_system = attributes.get(AlignXSemanticConventions.GEN_AI_SYSTEM)
        gen_ai_model = attributes.get(AlignXSemanticConventions.GEN_AI_REQUEST_MODEL)

        # Check for common LLM span patterns
        llm_patterns = [
            "openai",
            "anthropic",
            "google",
            "bedrock",
            "cohere",
            "mistral",
            "chat",
            "completion",
            "embedding",
            "generation",
        ]

        span_name_lower = span.name.lower() if span.name else ""

        return bool(gen_ai_system or gen_ai_model) or any(
            pattern in span_name_lower for pattern in llm_patterns
        )

    def _process_llm_span(self, span: ReadableSpan):
        """Process LLM span for metrics and cost calculation.

        Args:
            span: LLM span to process
        """
        if not self.telemetry_manager:
            self.logger.debug("No telemetry manager, skipping metrics recording")
            return

        attributes = span.attributes or {}

        # Extract LLM operation data
        model = attributes.get(
            AlignXSemanticConventions.GEN_AI_REQUEST_MODEL, "unknown"
        )
        provider = self._extract_provider_from_span(span)
        input_tokens = attributes.get("gen_ai.usage.prompt_tokens", 0)
        output_tokens = attributes.get("gen_ai.usage.completion_tokens", 0)
        total_tokens = attributes.get(
            "llm.usage.total_tokens", input_tokens + output_tokens
        )

        # Calculate duration
        duration = 0.0
        if span.end_time and span.start_time:
            duration = (span.end_time - span.start_time) / 1_000_000_000

        # Calculate cost if enabled
        cost_usd = None
        if self.config.capture_cost:
            cost_usd = self._calculate_cost(
                provider, model, input_tokens, output_tokens
            )

        # Use metrics bridge to record via existing AI metrics system
        try:
            self.metrics_bridge.record_llm_metrics(
                telemetry_manager=self.telemetry_manager,
                provider=provider,
                model=model,
                operation_type="chat.completion",  # Default operation type
                success=span.status.status_code != StatusCode.ERROR,
                duration_seconds=duration,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )
            self.logger.debug(f"Successfully recorded metrics for {provider} span")

        except Exception as e:
            self.logger.error(f"Failed to record LLM metrics: {e}", exc_info=True)

    def _extract_provider_from_span(self, span: ReadableSpan) -> str:
        """Extract provider name from span.

        Args:
            span: Span to extract provider from

        Returns:
            Provider name string
        """
        if not span.attributes:
            return "unknown"

        attributes = span.attributes

        # Check for explicit provider attribute
        provider = attributes.get(AlignXSemanticConventions.ALIGNX_PROVIDER)
        if provider:
            return provider

        # Check gen_ai.system
        gen_ai_system = attributes.get(AlignXSemanticConventions.GEN_AI_SYSTEM, "")
        if gen_ai_system:
            return gen_ai_system.lower()

        # Infer from span name
        span_name_lower = span.name.lower() if span.name else ""

        if "openai" in span_name_lower:
            return "openai"
        elif "anthropic" in span_name_lower:
            return "anthropic"
        elif "google" in span_name_lower or "gemini" in span_name_lower:
            return "google_genai"
        elif "bedrock" in span_name_lower:
            return "bedrock"
        elif "cohere" in span_name_lower:
            return "cohere"
        elif "mistral" in span_name_lower:
            return "mistral"

        return "unknown"

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        """Calculate cost using AlignX pricing.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD, or None if calculation fails
        """
        try:
            if not self.telemetry_manager:
                return None

            metrics_config = self.telemetry_manager.get_metrics_config()
            if not metrics_config or not metrics_config.enable_cost_tracking:
                return None

            from alignx_telemetry.pricing import get_llm_cost

            return get_llm_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            self.logger.debug(f"Cost calculation failed: {e}")
            return None

    def shutdown(self):
        """Shutdown the processor."""
        self._processed_spans.clear()
        self.logger.debug("AlignX LLM span processor shutdown")

    def force_flush(self, timeout_millis: float = 30000):
        """Force flush any pending spans.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if flush was successful
        """
        return True
