"""
Enhanced metrics schemas and data structures for AlignX observability.

This module defines the data structures used throughout the enhanced metrics system,
including provider metrics, event types, and error classifications.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class StreamingEvent(Enum):
    """Streaming event types for metrics tracking."""

    STREAM_START = "stream_start"
    FIRST_TOKEN = "first_token"
    TOKEN_RECEIVED = "token_received"
    STREAM_COMPLETE = "stream_complete"
    STREAM_ERROR = "stream_error"
    STREAM_INTERRUPTED = "stream_interrupted"


class ErrorType(Enum):
    """Standardized error types for LLM operations."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    NETWORK = "network"
    PROVIDER_ERROR = "provider_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTEXT_LENGTH = "context_length"
    UNKNOWN = "unknown"


@dataclass
class AlignXEnhancedProviderMetrics:
    """Enhanced metrics structure with industry-standard LLM observability data.

    This extends the basic provider metrics with advanced performance,
    cost optimization, and reliability metrics that are essential for
    production LLM monitoring.

    Key Features:
    - Latency breakdown (queue, processing, network)
    - Streaming performance (TTFT, tokens/sec)
    - Cost optimization (cached tokens, efficiency scores)
    - Quality metrics (error types, retry counts)
    - Context utilization tracking
    """

    # === BASIC IDENTIFICATION ===
    provider: str  # "openai", "anthropic", "google", etc.
    model: str  # "gpt-4", "claude-3-sonnet", etc.
    operation: str  # "chat.completion", "embedding", etc.

    # === CORE USAGE METRICS ===
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # === ADVANCED LATENCY METRICS (industry standard) ===
    total_latency_ms: float = 0.0
    queue_time_ms: Optional[float] = None  # Time waiting in provider queue
    processing_time_ms: Optional[float] = None  # Actual model processing time
    network_time_ms: Optional[float] = None  # Network overhead

    # === STREAMING PERFORMANCE METRICS (critical for real-time apps) ===
    is_streaming: bool = False
    time_to_first_token_ms: Optional[float] = None  # TTFT - crucial metric
    time_between_tokens_ms: Optional[float] = None  # Token generation rate
    stream_chunks_count: int = 0
    stream_completion_ratio: float = 1.0  # % of stream completed successfully
    tokens_per_second: Optional[float] = None

    # === ADVANCED TOKEN METRICS (cost optimization) ===
    cached_input_tokens: int = 0  # Tokens served from cache
    reasoning_tokens: int = 0  # O1 model reasoning tokens
    context_window_size: int = 0  # Model's context window
    context_utilization_ratio: float = 0.0  # How much context was used

    # === COST OPTIMIZATION METRICS ===
    cost_per_token: float = 0.0
    cost_per_request: float = 0.0
    cost_savings_from_cache: float = 0.0  # Savings from cached tokens
    cost_efficiency_score: Optional[float] = None  # Custom efficiency metric

    # === QUALITY AND RELIABILITY METRICS ===
    success: bool = True
    error_type: Optional[ErrorType] = None
    retry_count: int = 0
    is_cached_response: bool = False
    model_confidence: Optional[float] = None  # If provided by model

    # === FRAMEWORK AND WORKFLOW CONTEXT ===
    framework: Optional[str] = None  # "langchain", "llamaindex", etc.
    workflow_id: Optional[str] = None  # Unique workflow identifier
    node_name: Optional[str] = None  # Specific component/node name
    agent_name: Optional[str] = None  # Agent name (for multi-agent frameworks)

    # === SERVICE CONTEXT ===
    application_name: str = "unknown"
    environment: str = "unknown"
    service_name: str = "unknown"

    # === METADATA ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Post-init calculations and validations.

        This method calculates derived metrics and validates data consistency
        to ensure accurate observability data.
        """
        # Calculate total tokens if not provided
        if self.total_tokens == 0:
            self.total_tokens = (
                self.input_tokens + self.output_tokens + self.cached_input_tokens
            )

        # Calculate cost per token
        if self.total_tokens > 0 and self.cost_usd > 0:
            self.cost_per_token = self.cost_usd / self.total_tokens

        # Calculate context utilization ratio
        if self.context_window_size > 0:
            self.context_utilization_ratio = min(
                (self.input_tokens + self.output_tokens) / self.context_window_size, 1.0
            )

        # Calculate streaming metrics
        if self.is_streaming and self.time_to_first_token_ms and self.total_latency_ms:
            generation_time = self.total_latency_ms - self.time_to_first_token_ms
            if generation_time > 0 and self.output_tokens > 0:
                self.tokens_per_second = (self.output_tokens * 1000) / generation_time

        # Calculate cost efficiency (tokens per dollar)
        if self.cost_usd > 0:
            self.cost_efficiency_score = self.total_tokens / self.cost_usd

        # Set cost per request
        self.cost_per_request = self.cost_usd

    def to_labels(self) -> Dict[str, str]:
        """Convert metrics to OpenTelemetry labels.

        Returns:
            Dictionary of string labels for metric attribution
        """
        return {
            "gen_ai_system": self.provider,
            "gen_ai_request_model": self.model,
            "gen_ai_operation_name": self.operation,
            "gen_ai_application_name": self.application_name,
            "gen_ai_environment": self.environment,
            "service_name": self.service_name,
            "telemetry_sdk_name": "alignx",
            "framework": self.framework or "unknown",
            "streaming": str(self.is_streaming),
            "cached": str(self.is_cached_response),
        }

    def get_error_labels(self) -> Dict[str, str]:
        """Get labels for error metrics.

        Returns:
            Dictionary of labels including error type information
        """
        labels = self.to_labels()
        if self.error_type:
            labels["error_type"] = self.error_type.value
        return labels

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation.

        Returns:
            Dictionary representation of all metrics data
        """
        return {
            # Basic identification
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            # Usage metrics
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            # Performance metrics
            "total_latency_ms": self.total_latency_ms,
            "queue_time_ms": self.queue_time_ms,
            "processing_time_ms": self.processing_time_ms,
            "network_time_ms": self.network_time_ms,
            # Streaming metrics
            "is_streaming": self.is_streaming,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "time_between_tokens_ms": self.time_between_tokens_ms,
            "stream_chunks_count": self.stream_chunks_count,
            "stream_completion_ratio": self.stream_completion_ratio,
            "tokens_per_second": self.tokens_per_second,
            # Advanced token metrics
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "context_window_size": self.context_window_size,
            "context_utilization_ratio": self.context_utilization_ratio,
            # Cost metrics
            "cost_per_token": self.cost_per_token,
            "cost_per_request": self.cost_per_request,
            "cost_savings_from_cache": self.cost_savings_from_cache,
            "cost_efficiency_score": self.cost_efficiency_score,
            # Quality metrics
            "success": self.success,
            "error_type": self.error_type.value if self.error_type else None,
            "retry_count": self.retry_count,
            "is_cached_response": self.is_cached_response,
            "model_confidence": self.model_confidence,
            # Context
            "framework": self.framework,
            "workflow_id": self.workflow_id,
            "node_name": self.node_name,
            "agent_name": self.agent_name,
            "application_name": self.application_name,
            "environment": self.environment,
            "service_name": self.service_name,
            # Metadata
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


__all__ = [
    "AlignXEnhancedProviderMetrics",
    "StreamingEvent",
    "ErrorType",
]
