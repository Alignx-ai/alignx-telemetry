"""Metrics data structures and emission for AlignX provider instrumentation.

This module defines standardized metrics structures and emission logic that
integrates with the existing AlignX Grafana dashboard schema.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

# Avoid circular import - use string annotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alignx_telemetry.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


@dataclass
class AlignXProviderMetrics:
    """Standardized metrics structure for all LLM provider operations.

    This structure is designed to work with the existing AlignX Grafana dashboard
    schema while providing comprehensive observability data.
    """

    # Provider identification
    provider: str  # "openai", "anthropic", "google", etc.
    model: str  # "gpt-4", "claude-3-sonnet", etc.
    operation: str  # "chat.completion", "embedding", etc.

    # Usage metrics (core dashboard metrics)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None

    # Framework context (enhanced observability)
    framework: Optional[str] = None  # "langchain", "llamaindex", "crewai", etc.
    workflow_id: Optional[str] = None  # Unique workflow identifier
    node_name: Optional[str] = None  # Specific component/node name
    agent_name: Optional[str] = None  # Agent name (for multi-agent frameworks)

    # AlignX-specific labels (required by dashboard)
    application_name: str = "unknown"
    environment: str = "unknown"

    # Quality metrics
    error_type: Optional[str] = None
    success: bool = True

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-init validation and calculations."""
        # Calculate total tokens if not provided
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "framework": self.framework,
            "workflow_id": self.workflow_id,
            "node_name": self.node_name,
            "agent_name": self.agent_name,
            "application_name": self.application_name,
            "environment": self.environment,
            "error_type": self.error_type,
            "success": self.success,
            "metadata": self.metadata,
        }


class AlignXMetricsEmitter:
    """Metrics emitter that integrates with existing AlignX Grafana dashboard.

    This class emits metrics in the format expected by the current dashboard
    schema while adding enhanced observability capabilities.
    """

    def __init__(self, telemetry_manager: "TelemetryManager"):
        """Initialize metrics emitter.

        Args:
            telemetry_manager: AlignX telemetry manager instance
        """
        self.telemetry_manager = telemetry_manager
        self.logger = logging.getLogger(__name__)

        # Get metrics from telemetry manager
        try:
            self.registry = telemetry_manager.metric_registry
        except AttributeError:
            self.logger.warning("Metric registry not available in telemetry manager")
            self.registry = None

    def emit_llm_metrics(self, metrics: AlignXProviderMetrics) -> None:
        """Emit comprehensive LLM metrics for dashboard consumption.

        Args:
            metrics: Provider metrics to emit
        """
        if not self.registry:
            self.logger.debug("Metric registry not available, skipping metric emission")
            return

        try:
            # Create base labels for all metrics
            labels = self._create_base_labels(metrics)

            # Emit core dashboard metrics (existing schema)
            self._emit_core_metrics(labels, metrics)

            # Emit enhanced metrics (new capabilities)
            self._emit_enhanced_metrics(labels, metrics)

            # Emit error metrics if applicable
            if not metrics.success:
                self._emit_error_metrics(labels, metrics)

        except Exception as e:
            self.logger.error(f"Error emitting metrics: {e}", exc_info=True)

    def _create_base_labels(self, metrics: AlignXProviderMetrics) -> Dict[str, str]:
        """Create base labels for all metrics.

        Args:
            metrics: Provider metrics

        Returns:
            Dictionary of labels
        """
        labels = {
            # Core dashboard labels (existing schema)
            "gen_ai_system": metrics.provider,
            "gen_ai_request_model": metrics.model,
            "gen_ai_operation_name": metrics.operation,
            "gen_ai_application_name": metrics.application_name,
            "gen_ai_environment": metrics.environment,
            "telemetry_sdk_name": "alignx",
        }

        # Add framework context labels (enhanced observability)
        if metrics.framework:
            labels["gen_ai_framework"] = metrics.framework

        if metrics.workflow_id:
            labels["gen_ai_workflow_id"] = metrics.workflow_id[:50]  # Limit length

        if metrics.node_name:
            labels["gen_ai_node_name"] = metrics.node_name[:50]  # Limit length

        if metrics.agent_name:
            labels["gen_ai_agent_name"] = metrics.agent_name[:50]  # Limit length

        return labels

    def _emit_core_metrics(
        self, labels: Dict[str, str], metrics: AlignXProviderMetrics
    ) -> None:
        """Emit core metrics that match existing dashboard schema.

        Args:
            labels: Base labels for metrics
            metrics: Provider metrics
        """
        try:
            # Request counter (existing dashboard metric)
            if hasattr(self.registry, "gen_ai_requests_total"):
                self.registry.gen_ai_requests_total.labels(**labels).inc()

            # Token usage metrics (existing dashboard metrics)
            if hasattr(self.registry, "gen_ai_usage_input_tokens_total"):
                self.registry.gen_ai_usage_input_tokens_total.labels(**labels).inc(
                    metrics.input_tokens
                )

            if hasattr(self.registry, "gen_ai_usage_completion_tokens_total"):
                self.registry.gen_ai_usage_completion_tokens_total.labels(**labels).inc(
                    metrics.output_tokens
                )

            # Cost metrics (existing dashboard metric)
            if hasattr(self.registry, "gen_ai_usage_cost_usd_total"):
                self.registry.gen_ai_usage_cost_usd_total.labels(**labels).inc(
                    metrics.cost_usd
                )

        except Exception as e:
            self.logger.error(f"Error emitting core metrics: {e}")

    def _emit_enhanced_metrics(
        self, labels: Dict[str, str], metrics: AlignXProviderMetrics
    ) -> None:
        """Emit enhanced metrics for new observability capabilities.

        Args:
            labels: Base labels for metrics
            metrics: Provider metrics
        """
        try:
            # Performance metrics (new capabilities)
            if hasattr(self.registry, "gen_ai_request_duration_seconds"):
                self.registry.gen_ai_request_duration_seconds.labels(**labels).observe(
                    metrics.latency_ms / 1000.0  # Convert to seconds
                )

            # Time to first token (streaming performance)
            if metrics.time_to_first_token_ms is not None:
                if hasattr(self.registry, "gen_ai_time_to_first_token_seconds"):
                    self.registry.gen_ai_time_to_first_token_seconds.labels(
                        **labels
                    ).observe(
                        metrics.time_to_first_token_ms / 1000.0  # Convert to seconds
                    )

            # Framework-specific metrics
            if metrics.framework:
                framework_labels = {**labels, "framework": metrics.framework}

                if hasattr(self.registry, "gen_ai_framework_requests_total"):
                    self.registry.gen_ai_framework_requests_total.labels(
                        **framework_labels
                    ).inc()

            # Provider breakdown metrics
            provider_labels = {**labels, "provider": metrics.provider}
            if hasattr(self.registry, "gen_ai_provider_requests_total"):
                self.registry.gen_ai_provider_requests_total.labels(
                    **provider_labels
                ).inc()

        except Exception as e:
            self.logger.error(f"Error emitting enhanced metrics: {e}")

    def _emit_error_metrics(
        self, labels: Dict[str, str], metrics: AlignXProviderMetrics
    ) -> None:
        """Emit error-specific metrics.

        Args:
            labels: Base labels for metrics
            metrics: Provider metrics with error information
        """
        try:
            if metrics.error_type:
                error_labels = {**labels, "error_type": metrics.error_type}

                if hasattr(self.registry, "gen_ai_errors_total"):
                    self.registry.gen_ai_errors_total.labels(**error_labels).inc()

        except Exception as e:
            self.logger.error(f"Error emitting error metrics: {e}")

    def emit_provider_health_metrics(
        self, provider: str, is_healthy: bool, response_time_ms: float
    ) -> None:
        """Emit provider health and availability metrics.

        Args:
            provider: Provider name
            is_healthy: Whether provider is responding successfully
            response_time_ms: Response time in milliseconds
        """
        try:
            if not self.registry:
                return

            labels = {
                "provider": provider,
                "telemetry_sdk_name": "alignx",
            }

            # Health status
            if hasattr(self.registry, "gen_ai_provider_health"):
                health_value = 1.0 if is_healthy else 0.0
                self.registry.gen_ai_provider_health.labels(**labels).set(health_value)

            # Response time
            if hasattr(self.registry, "gen_ai_provider_response_time_seconds"):
                self.registry.gen_ai_provider_response_time_seconds.labels(
                    **labels
                ).observe(response_time_ms / 1000.0)

        except Exception as e:
            self.logger.error(f"Error emitting provider health metrics: {e}")


class AlignXCostCalculator:
    """Cost calculation utilities for different providers.

    This class provides standardized cost calculation across providers
    using up-to-date pricing models.
    """

    # Provider pricing models (can be extended with external pricing APIs)
    PRICING_MODELS = {
        "openai": {
            "gpt-4": {"input": 0.00003, "output": 0.00006},  # per token
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
            "text-embedding-ada-002": {"input": 0.0000001, "output": 0.0},
            "text-embedding-3-small": {"input": 0.00000002, "output": 0.0},
            "text-embedding-3-large": {"input": 0.00000013, "output": 0.0},
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
        },
        "google": {
            "gemini-pro": {"input": 0.0000005, "output": 0.0000015},
            "gemini-pro-vision": {"input": 0.00000025, "output": 0.0000005},
        },
    }

    @classmethod
    def calculate_cost(
        cls, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a provider operation.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        try:
            provider_pricing = cls.PRICING_MODELS.get(provider.lower(), {})
            model_pricing = provider_pricing.get(model.lower(), {})

            if not model_pricing:
                logger.debug(f"No pricing available for {provider}/{model}")
                return 0.0

            input_cost = input_tokens * model_pricing.get("input", 0.0)
            output_cost = output_tokens * model_pricing.get("output", 0.0)

            return input_cost + output_cost

        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0

    @classmethod
    def update_pricing_model(
        cls, provider: str, model: str, input_price: float, output_price: float
    ) -> None:
        """Update pricing model for a provider/model combination.

        Args:
            provider: Provider name
            model: Model name
            input_price: Price per input token
            output_price: Price per output token
        """
        if provider.lower() not in cls.PRICING_MODELS:
            cls.PRICING_MODELS[provider.lower()] = {}

        cls.PRICING_MODELS[provider.lower()][model.lower()] = {
            "input": input_price,
            "output": output_price,
        }
