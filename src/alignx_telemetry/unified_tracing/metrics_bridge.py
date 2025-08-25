"""Metrics bridge between new provider system and existing AI metrics."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsBridge:
    """Bridge between new provider metrics and existing AI metrics system.

    This class ensures that metrics from the new provider system are properly
    recorded in the existing AI metrics infrastructure, maintaining backward
    compatibility with Grafana dashboards and metric aggregation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def record_llm_metrics(
        self,
        telemetry_manager,
        provider: str,
        model: str,
        operation_type: str = "chat.completion",
        success: bool = True,
        duration_seconds: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Record LLM metrics via existing AI metrics system.

        This method bridges the gap between the new provider metrics format
        and the existing ai_metrics.record_llm_request() method.

        Args:
            telemetry_manager: AlignX telemetry manager instance
            provider: LLM provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            operation_type: Type of operation (e.g., "chat.completion")
            success: Whether the operation was successful
            duration_seconds: Duration of the operation in seconds
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens
            total_tokens: Total number of tokens
            cost_usd: Cost in USD (optional)
            **kwargs: Additional metrics data

        Returns:
            True if metrics were recorded successfully, False otherwise
        """
        try:
            if not telemetry_manager:
                self.logger.debug("No telemetry manager provided")
                return False

            ai_metrics = telemetry_manager.get_ai_metrics()
            if not ai_metrics:
                self.logger.debug("No AI metrics instance available")
                return False

            # Record via existing AI metrics system
            ai_metrics.record_llm_request(
                provider=provider,
                model=model,
                operation_type=operation_type,
                success=success,
                duration_seconds=duration_seconds,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )

            self.logger.debug(
                f"Successfully recorded metrics: {provider}/{model}, "
                f"tokens={total_tokens}, duration={duration_seconds:.3f}s"
                + (f", cost=${cost_usd:.6f}" if cost_usd else "")
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to record LLM metrics: {e}", exc_info=True)
            return False

    def bridge_provider_metrics(
        self, telemetry_manager, provider_metrics, **kwargs
    ) -> bool:
        """Bridge metrics from new provider system to existing AI metrics.

        This method takes metrics in the new AlignXProviderMetrics format
        and converts them to the format expected by the existing AI metrics system.

        Args:
            telemetry_manager: AlignX telemetry manager instance
            provider_metrics: Metrics from new provider system
            **kwargs: Additional context

        Returns:
            True if metrics were bridged successfully, False otherwise
        """
        try:
            # Import here to avoid circular dependencies
            from ..providers.metrics import AlignXProviderMetrics

            if not isinstance(provider_metrics, AlignXProviderMetrics):
                self.logger.warning(
                    f"Invalid provider metrics type: {type(provider_metrics)}"
                )
                return False

            # Convert provider metrics to AI metrics format
            return self.record_llm_metrics(
                telemetry_manager=telemetry_manager,
                provider=provider_metrics.provider,
                model=provider_metrics.model,
                operation_type=provider_metrics.operation,
                success=provider_metrics.success,
                duration_seconds=provider_metrics.duration_seconds,
                prompt_tokens=provider_metrics.input_tokens,
                completion_tokens=provider_metrics.output_tokens,
                total_tokens=provider_metrics.total_tokens,
                cost_usd=provider_metrics.cost_usd,
                **kwargs,
            )

        except Exception as e:
            self.logger.error(f"Failed to bridge provider metrics: {e}", exc_info=True)
            return False

    def record_provider_metrics_from_span(
        self, telemetry_manager, span, provider_metrics, **kwargs
    ) -> bool:
        """Record provider metrics with span context.

        This method adds span-specific context to provider metrics before
        recording them via the existing AI metrics system.

        Args:
            telemetry_manager: AlignX telemetry manager instance
            span: OpenTelemetry span for context
            provider_metrics: Metrics from provider system
            **kwargs: Additional context

        Returns:
            True if metrics were recorded successfully, False otherwise
        """
        try:
            # Extract additional context from span if available
            additional_context = {}

            if span and hasattr(span, "attributes") and span.attributes:
                attributes = span.attributes

                # Add workflow context if available
                if attributes.get("alignx.correlation.enabled"):
                    additional_context["workflow_context"] = True
                    additional_context["workflow_id"] = attributes.get(
                        "alignx.workflow.id"
                    )
                    additional_context["node_name"] = attributes.get(
                        "alignx.workflow.node.name"
                    )
                else:
                    additional_context["workflow_context"] = False

                # Add service context
                additional_context["service_name"] = attributes.get(
                    "gen_ai.application_name"
                )
                additional_context["environment"] = attributes.get("gen_ai.environment")

            # Merge with provided kwargs
            merged_kwargs = {**additional_context, **kwargs}

            # Bridge the metrics
            return self.bridge_provider_metrics(
                telemetry_manager=telemetry_manager,
                provider_metrics=provider_metrics,
                **merged_kwargs,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to record provider metrics from span: {e}", exc_info=True
            )
            return False


# Global instance for easy access
_global_bridge = None


def get_metrics_bridge() -> MetricsBridge:
    """Get the global metrics bridge instance.

    Returns:
        Global MetricsBridge instance
    """
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = MetricsBridge()
    return _global_bridge
