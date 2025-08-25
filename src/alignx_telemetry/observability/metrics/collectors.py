"""
Enhanced metrics collectors for AlignX observability.

This module provides the metrics collection infrastructure for capturing
industry-standard LLM observability metrics including latency percentiles,
streaming performance, cost optimization, and reliability metrics.
"""

import logging
from typing import Optional, Dict, Any

from opentelemetry import metrics
from opentelemetry.metrics import Histogram, Counter, Gauge

from .schemas import AlignXEnhancedProviderMetrics, StreamingEvent

logger = logging.getLogger(__name__)


class AlignXEnhancedMetricsCollector:
    """Enhanced metrics collector with industry-standard LLM observability metrics.

    This collector captures comprehensive metrics including latency percentiles,
    streaming performance, cost optimization, and reliability metrics that are
    essential for production LLM monitoring and optimization.

    Key Features:
    - Latency percentiles (P50, P95, P99) with optimized buckets
    - Streaming metrics (TTFT, tokens/sec, completion rates)
    - Cost optimization (cached tokens, efficiency scores)
    - Error classification and reliability tracking
    - Framework and provider breakdown
    """

    def __init__(self, meter_name: str = "alignx.enhanced.metrics"):
        """Initialize the enhanced metrics collector.

        Args:
            meter_name: Name of the OpenTelemetry meter for metric attribution
        """
        self.meter = metrics.get_meter(meter_name)
        self.logger = logging.getLogger(__name__)
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all metrics instruments with industry-standard configurations."""

        # === CORE REQUEST METRICS ===
        self.requests_total = self.meter.create_counter(
            "gen_ai_requests_total",
            description="Total LLM requests by provider, model, and operation",
            unit="1",
        )

        # === LATENCY METRICS (with percentile-friendly buckets) ===
        # Buckets optimized for LLM response times (milliseconds to minutes)
        latency_buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
        ]

        self.request_duration = self.meter.create_histogram(
            "gen_ai_request_duration_seconds",
            description="Request duration with percentile buckets (p50, p95, p99)",
            unit="s",
        )

        self.queue_time = self.meter.create_histogram(
            "gen_ai_queue_time_seconds",
            description="Time spent waiting in provider queue",
            unit="s",
        )

        self.processing_time = self.meter.create_histogram(
            "gen_ai_processing_time_seconds",
            description="Actual model processing time",
            unit="s",
        )

        # === STREAMING METRICS (critical for real-time applications) ===
        self.time_to_first_token = self.meter.create_histogram(
            "gen_ai_time_to_first_token_seconds",
            description="Time to first token in streaming responses (TTFT)",
            unit="s",
        )

        self.tokens_per_second = self.meter.create_histogram(
            "gen_ai_tokens_per_second",
            description="Token generation rate for streaming responses",
            unit="tokens/s",
        )

        self.stream_completion_ratio = self.meter.create_histogram(
            "gen_ai_stream_completion_ratio",
            description="Percentage of stream completed successfully",
            unit="ratio",
        )

        self.stream_chunks = self.meter.create_histogram(
            "gen_ai_stream_chunks_total",
            description="Number of chunks in streaming response",
            unit="1",
        )

        # === TOKEN USAGE METRICS ===
        self.input_tokens_total = self.meter.create_counter(
            "gen_ai_input_tokens_total",
            description="Total input/prompt tokens",
            unit="1",
        )

        self.output_tokens_total = self.meter.create_counter(
            "gen_ai_output_tokens_total",
            description="Total output/completion tokens",
            unit="1",
        )

        self.cached_tokens_total = self.meter.create_counter(
            "gen_ai_cached_tokens_total",
            description="Total cached input tokens (cost optimization)",
            unit="1",
        )

        self.reasoning_tokens_total = self.meter.create_counter(
            "gen_ai_reasoning_tokens_total",
            description="Total reasoning tokens (O1 models)",
            unit="1",
        )

        # === CONTEXT WINDOW METRICS ===
        self.context_utilization = self.meter.create_histogram(
            "gen_ai_context_utilization_ratio",
            description="Context window utilization ratio",
            unit="ratio",
        )

        # === COST METRICS ===
        self.cost_usd_total = self.meter.create_counter(
            "gen_ai_cost_usd_total", description="Total cost in USD", unit="USD"
        )

        self.cost_per_request = self.meter.create_histogram(
            "gen_ai_cost_per_request_usd",
            description="Cost per request distribution",
            unit="USD",
        )

        self.cost_savings_cached = self.meter.create_counter(
            "gen_ai_cost_savings_cached_usd",
            description="Cost savings from cached tokens",
            unit="USD",
        )

        self.cost_efficiency = self.meter.create_histogram(
            "gen_ai_cost_efficiency_tokens_per_usd",
            description="Cost efficiency (tokens per dollar)",
            unit="tokens/USD",
        )

        # === ERROR AND RELIABILITY METRICS ===
        self.errors_total = self.meter.create_counter(
            "gen_ai_errors_total",
            description="Total errors by type and provider",
            unit="1",
        )

        self.retry_attempts = self.meter.create_counter(
            "gen_ai_retry_attempts_total", description="Total retry attempts", unit="1"
        )

        self.model_availability = self.meter.create_gauge(
            "gen_ai_model_availability_ratio",
            description="Model availability ratio (0-1)",
            unit="ratio",
        )

        # === FRAMEWORK METRICS ===
        self.framework_requests = self.meter.create_counter(
            "gen_ai_framework_requests_total",
            description="Requests by framework type",
            unit="1",
        )

    def record_metrics(self, metrics_data: AlignXEnhancedProviderMetrics) -> None:
        """Record comprehensive metrics from enhanced provider data.

        This method processes and records all available metrics from the
        enhanced provider metrics data structure, ensuring comprehensive
        observability coverage.

        Args:
            metrics_data: Enhanced provider metrics data
        """
        try:
            labels = metrics_data.to_labels()

            # Core request metrics
            self.requests_total.add(1, labels)

            # Latency metrics (convert milliseconds to seconds for OpenTelemetry)
            if metrics_data.total_latency_ms > 0:
                self.request_duration.record(
                    metrics_data.total_latency_ms / 1000.0, labels
                )

            if metrics_data.queue_time_ms:
                self.queue_time.record(metrics_data.queue_time_ms / 1000.0, labels)

            if metrics_data.processing_time_ms:
                self.processing_time.record(
                    metrics_data.processing_time_ms / 1000.0, labels
                )

            # Streaming metrics (only for streaming requests)
            if metrics_data.is_streaming:
                if metrics_data.time_to_first_token_ms:
                    self.time_to_first_token.record(
                        metrics_data.time_to_first_token_ms / 1000.0, labels
                    )

                if metrics_data.tokens_per_second:
                    self.tokens_per_second.record(
                        metrics_data.tokens_per_second, labels
                    )

                self.stream_completion_ratio.record(
                    metrics_data.stream_completion_ratio, labels
                )
                self.stream_chunks.record(metrics_data.stream_chunks_count, labels)

            # Token metrics
            if metrics_data.input_tokens > 0:
                self.input_tokens_total.add(metrics_data.input_tokens, labels)

            if metrics_data.output_tokens > 0:
                self.output_tokens_total.add(metrics_data.output_tokens, labels)

            if metrics_data.cached_input_tokens > 0:
                self.cached_tokens_total.add(metrics_data.cached_input_tokens, labels)

            if metrics_data.reasoning_tokens > 0:
                self.reasoning_tokens_total.add(metrics_data.reasoning_tokens, labels)

            # Context utilization
            if metrics_data.context_utilization_ratio > 0:
                self.context_utilization.record(
                    metrics_data.context_utilization_ratio, labels
                )

            # Cost metrics
            if metrics_data.cost_usd > 0:
                self.cost_usd_total.add(metrics_data.cost_usd, labels)
                self.cost_per_request.record(metrics_data.cost_per_request, labels)

            if metrics_data.cost_savings_from_cache > 0:
                self.cost_savings_cached.add(
                    metrics_data.cost_savings_from_cache, labels
                )

            if metrics_data.cost_efficiency_score:
                self.cost_efficiency.record(metrics_data.cost_efficiency_score, labels)

            # Error and reliability metrics
            if not metrics_data.success:
                error_labels = metrics_data.get_error_labels()
                self.errors_total.add(1, error_labels)

            if metrics_data.retry_count > 0:
                self.retry_attempts.add(metrics_data.retry_count, labels)

            # Framework-specific metrics
            if metrics_data.framework:
                framework_labels = {**labels, "framework": metrics_data.framework}
                self.framework_requests.add(1, framework_labels)

        except Exception as e:
            self.logger.error(f"Error recording enhanced metrics: {e}", exc_info=True)

    def record_streaming_event(
        self,
        event: StreamingEvent,
        metrics_data: AlignXEnhancedProviderMetrics,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record streaming-specific events for detailed streaming analysis.

        Args:
            event: Type of streaming event
            metrics_data: Base metrics data for context
            event_data: Additional event-specific data
        """
        try:
            labels = metrics_data.to_labels()
            labels["event_type"] = event.value

            # Create streaming event counter if not exists (lazy initialization)
            if not hasattr(self, "streaming_events"):
                self.streaming_events = self.meter.create_counter(
                    "gen_ai_streaming_events_total",
                    description="Streaming events by type",
                    unit="1",
                )

            self.streaming_events.add(1, labels)

        except Exception as e:
            self.logger.error(f"Error recording streaming event: {e}", exc_info=True)

    def record_model_availability(
        self, provider: str, model: str, is_available: bool
    ) -> None:
        """Record model availability status for monitoring provider health.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            is_available: Whether the model is currently available
        """
        try:
            labels = {
                "gen_ai_system": provider,
                "gen_ai_request_model": model,
                "telemetry_sdk_name": "alignx",
            }

            availability_value = 1.0 if is_available else 0.0
            self.model_availability.set(availability_value, labels)

        except Exception as e:
            self.logger.error(f"Error recording model availability: {e}", exc_info=True)


# Global enhanced metrics collector instance
_enhanced_metrics_collector: Optional[AlignXEnhancedMetricsCollector] = None


def get_enhanced_metrics_collector() -> AlignXEnhancedMetricsCollector:
    """Get or create the global enhanced metrics collector.

    This function implements the singleton pattern to ensure consistent
    metrics collection across the entire AlignX SDK.

    Returns:
        Global enhanced metrics collector instance
    """
    global _enhanced_metrics_collector
    if _enhanced_metrics_collector is None:
        _enhanced_metrics_collector = AlignXEnhancedMetricsCollector()
    return _enhanced_metrics_collector


__all__ = [
    "AlignXEnhancedMetricsCollector",
    "get_enhanced_metrics_collector",
]
