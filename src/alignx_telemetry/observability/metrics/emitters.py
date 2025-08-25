"""
Enhanced metrics emitters for AlignX observability.

This module provides convenience functions and utilities for emitting
enhanced metrics data to the collection system.
"""

import logging
from typing import Optional

from .schemas import AlignXEnhancedProviderMetrics, StreamingEvent
from .collectors import get_enhanced_metrics_collector

logger = logging.getLogger(__name__)


def record_enhanced_llm_metrics(metrics_data: AlignXEnhancedProviderMetrics) -> None:
    """Convenience function to record enhanced LLM metrics.

    This is the primary function for recording comprehensive LLM metrics
    from provider implementations. It handles all the complexity of metric
    collection and ensures consistent data recording across the SDK.

    Args:
        metrics_data: Enhanced provider metrics data containing all observability information

    Example:
        ```python
        from alignx_telemetry.observability.metrics import record_enhanced_llm_metrics
        from alignx_telemetry.observability.metrics.schemas import AlignXEnhancedProviderMetrics

        metrics = AlignXEnhancedProviderMetrics(
            provider="openai",
            model="gpt-4",
            operation="chat.completion",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.003,
            total_latency_ms=1250.0,
            time_to_first_token_ms=200.0,
            is_streaming=True,
        )

        record_enhanced_llm_metrics(metrics)
        ```
    """
    try:
        collector = get_enhanced_metrics_collector()
        collector.record_metrics(metrics_data)

        logger.debug(
            f"Recorded enhanced metrics for {metrics_data.provider}/{metrics_data.model}: "
            f"{metrics_data.total_tokens} tokens, ${metrics_data.cost_usd:.6f}, "
            f"{metrics_data.total_latency_ms:.1f}ms"
        )

    except Exception as e:
        logger.error(f"Failed to record enhanced LLM metrics: {e}", exc_info=True)


def record_streaming_event(
    event: StreamingEvent,
    provider: str,
    model: str,
    operation: str = "chat.completion",
    **additional_context,
) -> None:
    """Record a streaming event for detailed streaming analysis.

    This function provides a simplified interface for recording streaming events
    without requiring a full metrics data structure.

    Args:
        event: Type of streaming event
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-sonnet")
        operation: Operation type (default: "chat.completion")
        **additional_context: Additional context data

    Example:
        ```python
        from alignx_telemetry.observability.metrics import record_streaming_event
        from alignx_telemetry.observability.metrics.schemas import StreamingEvent

        record_streaming_event(
            StreamingEvent.FIRST_TOKEN,
            provider="openai",
            model="gpt-4",
            operation="chat.completion",
            time_to_first_token_ms=150.0
        )
        ```
    """
    try:
        # Create minimal metrics data for the streaming event
        metrics_data = AlignXEnhancedProviderMetrics(
            provider=provider,
            model=model,
            operation=operation,
            is_streaming=True,
            **additional_context,
        )

        collector = get_enhanced_metrics_collector()
        collector.record_streaming_event(event, metrics_data)

        logger.debug(f"Recorded streaming event {event.value} for {provider}/{model}")

    except Exception as e:
        logger.error(f"Failed to record streaming event: {e}", exc_info=True)


def record_model_availability(provider: str, model: str, is_available: bool) -> None:
    """Record model availability status for provider health monitoring.

    This function is used to track the availability of specific models from
    different providers, which is crucial for reliability monitoring.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-sonnet")
        is_available: Whether the model is currently available

    Example:
        ```python
        from alignx_telemetry.observability.metrics import record_model_availability

        # Record that GPT-4 is available
        record_model_availability("openai", "gpt-4", True)

        # Record that a model is temporarily unavailable
        record_model_availability("anthropic", "claude-3-opus", False)
        ```
    """
    try:
        collector = get_enhanced_metrics_collector()
        collector.record_model_availability(provider, model, is_available)

        status = "available" if is_available else "unavailable"
        logger.debug(f"Recorded model availability: {provider}/{model} is {status}")

    except Exception as e:
        logger.error(f"Failed to record model availability: {e}", exc_info=True)


def create_metrics_from_provider_data(
    provider: str,
    model: str,
    operation: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    success: bool = True,
    **kwargs,
) -> AlignXEnhancedProviderMetrics:
    """Create enhanced metrics data structure from basic provider data.

    This utility function helps create properly structured metrics data
    from basic provider information, applying intelligent defaults and
    calculating derived metrics.

    Args:
        provider: Provider name
        model: Model name
        operation: Operation type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
        latency_ms: Total latency in milliseconds
        success: Whether the operation was successful
        **kwargs: Additional metrics data

    Returns:
        AlignXEnhancedProviderMetrics: Structured metrics data

    Example:
        ```python
        from alignx_telemetry.observability.metrics import create_metrics_from_provider_data

        metrics = create_metrics_from_provider_data(
            provider="openai",
            model="gpt-4",
            operation="chat.completion",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.003,
            latency_ms=1250.0,
            is_streaming=True,
            time_to_first_token_ms=200.0
        )
        ```
    """
    return AlignXEnhancedProviderMetrics(
        provider=provider,
        model=model,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        total_latency_ms=latency_ms,
        success=success,
        **kwargs,
    )


__all__ = [
    "record_enhanced_llm_metrics",
    "record_streaming_event",
    "record_model_availability",
    "create_metrics_from_provider_data",
]
