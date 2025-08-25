"""Metrics collection and reporting for AI operations and general application metrics."""

import os
import time
from typing import Optional

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource


# Standard operation types for consistency across providers
class OperationType:
    """Standardized operation types for AlignX telemetry."""

    # GenAI Operations
    CHAT_COMPLETION = "chat.completion"
    TEXT_COMPLETION = "text.completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image.generation"
    AUDIO_GENERATION = "audio.generation"
    AUDIO_TRANSCRIPTION = "audio.transcription"
    VISION = "vision"
    FINE_TUNING = "fine_tuning"

    # VectorDB Operations
    DB_ADD = "add"
    DB_DELETE = "delete"
    DB_QUERY = "query"
    DB_UPSERT = "upsert"
    DB_UPDATE = "update"

    # AI Framework Operations
    CHAIN_INVOKE = "chain.invoke"
    AGENT_RUN = "agent.run"
    TOOL_CALL = "tool.call"


class MetricsConfig:
    """Configuration for AlignX metrics collection."""

    def __init__(
        self,
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        enable_cost_tracking: bool = True,
        enable_gpu_monitoring: bool = True,
        enable_content_capture: bool = True,
    ):
        """Initialize metrics configuration.

        Args:
            service_name: Name of the application
            environment: Environment name
            enable_cost_tracking: Whether to track LLM usage costs
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_content_capture: Whether to capture prompt/completion content
        """
        self.service_name = service_name
        self.environment = environment
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_content_capture = enable_content_capture


class MetricsInstrumentor:
    """Instrumentor for metrics collection."""

    def __init__(self):
        """Initialize the metrics instrumentor."""
        self._meter_provider = None
        self._instrumented = False

    def instrument(
        self,
        resource: Resource,
        otlp_endpoint: Optional[str] = None,
        export_interval_millis: int = 60000,
    ) -> None:
        """Instrument the metrics collection system.

        Args:
            resource: Existing OpenTelemetry resource to use.
            otlp_endpoint: OTLP exporter endpoint (defaults to OTEL_EXPORTER_OTLP_ENDPOINT_METRICS env var)
            export_interval_millis: How often to export metrics in milliseconds
        """
        if self._instrumented:
            return

        # Get otlp_endpoint from env vars if not provided
        otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT_METRICS"
        )

        # Configure metrics exporter
        if otlp_endpoint:
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint),
                export_interval_millis=export_interval_millis,
            )
            self._meter_provider = MeterProvider(
                resource=resource, metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(self._meter_provider)
            self._instrumented = True
        else:
            # No endpoint, no metrics
            self._instrumented = False
            return

    def uninstrument(self) -> None:
        """Stop metrics collection."""
        if self._meter_provider:
            self._meter_provider.shutdown()
            self._instrumented = False


class AIMetrics:
    """Enhanced metrics collection for AI operations with AlignX-style dimensional metrics."""

    def __init__(
        self,
        meter_name: str = "alignx.ai.metrics",
        config: Optional[MetricsConfig] = None,
    ):
        """Initialize the AI metrics collector.

        Args:
            meter_name: Name of the meter for OpenTelemetry metrics
            config: Metrics configuration (defaults to MetricsConfig())
        """
        self.meter = metrics.get_meter(meter_name)
        self.config = config or MetricsConfig()
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize the metrics instruments with AlignX-style naming."""

        # GenAI Request Metrics
        self.gen_ai_requests_total = self.meter.create_counter(
            "gen_ai_requests_total",
            description="Total number of GenAI requests by provider, model, and operation",
            unit="1",
        )

        self.gen_ai_request_duration_seconds = self.meter.create_histogram(
            "gen_ai_request_duration_seconds",
            description="Duration of GenAI requests in seconds",
            unit="s",
        )

        # Token Usage Metrics
        self.gen_ai_usage_tokens_total = self.meter.create_counter(
            "gen_ai_usage_tokens_total",
            description="Total number of tokens used across all requests",
            unit="1",
        )

        self.gen_ai_usage_input_tokens_total = self.meter.create_counter(
            "gen_ai_usage_input_tokens_total",
            description="Total number of input/prompt tokens used",
            unit="1",
        )

        self.gen_ai_usage_completion_tokens_total = self.meter.create_counter(
            "gen_ai_usage_completion_tokens_total",
            description="Total number of completion/output tokens used",
            unit="1",
        )

        # Cost Tracking Metrics (only if enabled)
        if self.config.enable_cost_tracking:
            self.gen_ai_usage_cost_usd_total = self.meter.create_counter(
                "gen_ai_usage_cost_usd_total",
                description="Total cost of GenAI usage in USD",
            )

            self.gen_ai_usage_cost_usd_bucket = self.meter.create_histogram(
                "gen_ai_usage_cost_usd_bucket",
                description="Distribution of GenAI usage costs in USD",
            )

        # VectorDB Metrics
        self.db_requests_total = self.meter.create_counter(
            "db_requests_total",
            description="Total number of database requests by system and operation",
            unit="1",
        )

        self.db_request_duration_seconds = self.meter.create_histogram(
            "db_request_duration_seconds",
            description="Duration of database requests in seconds",
            unit="s",
        )

        # Legacy metrics for backward compatibility
        self.prompt_tokens = self.meter.create_counter(
            "ai.tokens.prompt",
            description="Number of prompt tokens used",
            unit="tokens",
        )

        self.completion_tokens = self.meter.create_counter(
            "ai.tokens.completion",
            description="Number of completion tokens used",
            unit="tokens",
        )

        self.total_tokens = self.meter.create_counter(
            "ai.tokens.total",
            description="Total number of tokens used",
            unit="tokens",
        )

        self.llm_latency = self.meter.create_histogram(
            "ai.latency",
            description="Latency of AI operations",
            unit="ms",
        )

        self.errors = self.meter.create_counter(
            "ai.errors",
            description="Number of errors in AI operations",
            unit="errors",
        )

        self.requests = self.meter.create_counter(
            "ai.requests",
            description="Number of AI requests",
            unit="requests",
        )

    def _get_standard_attributes(
        self,
        provider: str,
        model: str = "unknown",
        operation_type: str = "unknown",
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        **additional_attributes,
    ) -> dict:
        """Get standardized attributes for AlignX metrics.

        Args:
            provider: AI provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            operation_type: Standardized operation type
            application_name: Override application name
            environment: Override environment name
            **additional_attributes: Additional custom attributes

        Returns:
            Standardized attributes dictionary
        """
        attributes = {
            "gen_ai_system": provider,
            "gen_ai_request_model": model,
            "gen_ai_response_model": model,  # Assuming same model for request/response
            "gen_ai_operation_name": operation_type,
            "gen_ai_application_name": service_name,
            "gen_ai_environment": environment,
            "telemetry_sdk_name": "alignx",
            "telemetry_sdk_version": "1.0.0",
        }

        # Add any additional attributes
        attributes.update(additional_attributes)

        return attributes

    def _get_db_attributes(
        self,
        db_system: str,
        operation: str,
        **additional_attributes,
    ) -> dict:
        """Get standardized attributes for database metrics.

        Args:
            db_system: Database system name (e.g., "pinecone", "chroma")
            operation: Database operation type (add, delete, query, etc.)
            application_name: Override application name
            environment: Override environment name
            **additional_attributes: Additional custom attributes

        Returns:
            Standardized attributes dictionary
        """
        attributes = {
            "db_system": db_system,
            "db_operation": operation,
            "gen_ai_application_name": self.config.service_name,
            "gen_ai_environment": self.config.environment,
            "telemetry_sdk_name": "alignx",
            "telemetry_sdk_version": "1.0.0",
        }

        # Add any additional attributes
        attributes.update(additional_attributes)

        return attributes

    def record_llm_request(
        self,
        provider: str,
        model: str = "unknown",
        operation_type: str = OperationType.CHAT_COMPLETION,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        **additional_attributes,
    ) -> None:
        """Record a complete LLM request with enhanced dimensional metrics.

        Args:
            provider: AI provider name (e.g., "openai", "anthropic", "cohere")
            model: Model name (e.g., "gpt-4", "claude-3-sonnet-20240229")
            operation_type: Standardized operation type (use OperationType constants)
            success: Whether the request was successful
            duration_seconds: Request duration in seconds
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total number of tokens used
            cost_usd: Cost of the request in USD
            **additional_attributes: Additional attributes to include in metrics
        """
        # Get standardized attributes
        attributes = self._get_standard_attributes(
            provider=provider,
            model=model,
            operation_type=operation_type,
            service_name=self.config.service_name,
            environment=self.config.environment,
            success=success,
            **additional_attributes,
        )

        # Record request count
        self.gen_ai_requests_total.add(1, attributes)

        # Record duration if provided
        if duration_seconds is not None:
            self.gen_ai_request_duration_seconds.record(duration_seconds, attributes)

        # Record token usage
        if prompt_tokens is not None:
            self.gen_ai_usage_input_tokens_total.add(prompt_tokens, attributes)
        if completion_tokens is not None:
            self.gen_ai_usage_completion_tokens_total.add(completion_tokens, attributes)
        if total_tokens is not None:
            self.gen_ai_usage_tokens_total.add(total_tokens, attributes)

        # Record cost (only if cost tracking is enabled)
        if self.config.enable_cost_tracking and cost_usd is not None and cost_usd > 0:
            self.gen_ai_usage_cost_usd_total.add(cost_usd, attributes)
            self.gen_ai_usage_cost_usd_bucket.record(cost_usd, attributes)

    def record_db_request(
        self,
        db_system: str,
        operation: str = OperationType.DB_QUERY,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        **additional_attributes,
    ) -> None:
        """Record a database/VectorDB request.

        Args:
            db_system: Database system name (e.g., "pinecone", "chroma", "qdrant")
            operation: Database operation (use OperationType.DB_* constants)
            success: Whether the request was successful
            duration_seconds: Request duration in seconds
            **additional_attributes: Additional attributes to include in metrics
        """
        # Get standardized attributes
        attributes = self._get_db_attributes(
            db_system=db_system,
            operation=operation,
            service_name=self.config.service_name,
            environment=self.config.environment,
            success=success,
            **additional_attributes,
        )

        # Record request count
        self.db_requests_total.add(1, attributes)

        # Record duration if provided
        if duration_seconds is not None:
            self.db_request_duration_seconds.record(duration_seconds, attributes)

    def get_config(self) -> MetricsConfig:
        """Get the current metrics configuration."""
        return self.config

    def update_config(self, **config_updates) -> None:
        """Update metrics configuration.

        Args:
            **config_updates: Configuration parameters to update
        """
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


class ApplicationMetrics:
    """General-purpose metrics collection for application operations."""

    def __init__(self, meter_name: str = "alignx.application"):
        """Initialize the application metrics collector.

        Args:
            meter_name: Name of the meter for OpenTelemetry metrics
        """
        self.meter = metrics.get_meter(meter_name)
        self._counters = {}
        self._histograms = {}
        self._gauges = {}

    def create_counter(
        self, name: str, description: str = "", unit: str = "count"
    ) -> metrics.Counter:
        """Create or get a counter.

        Args:
            name: Name of the counter
            description: Description of the counter
            unit: Unit of the counter

        Returns:
            A counter instance
        """
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name, description=description, unit=unit
            )
        return self._counters[name]

    def create_histogram(
        self, name: str, description: str = "", unit: str = "ms"
    ) -> metrics.Histogram:
        """Create or get a histogram.

        Args:
            name: Name of the histogram
            description: Description of the histogram
            unit: Unit of the histogram

        Returns:
            A histogram instance
        """
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name, description=description, unit=unit
            )
        return self._histograms[name]


class MetricsTimer:
    """Context manager for measuring and recording operation duration."""

    def __init__(
        self,
        ai_metrics: AIMetrics,
        provider: str = "unknown",
        model: str = "unknown",
        operation_type: str = OperationType.CHAT_COMPLETION,
    ):
        """Initialize the metrics timer.

        Args:
            ai_metrics: AIMetrics instance for recording metrics
            provider: AI provider name
            model: Model name
            operation_type: Type of operation
        """
        self.ai_metrics = ai_metrics
        self.provider = provider
        self.model = model
        self.operation_type = operation_type
        self.start_time = None
        self.end_time = None
        self.success = True
        self.prompt_tokens = None
        self.completion_tokens = None
        self.cost_usd = None

    def __enter__(self) -> "MetricsTimer":
        """Start the timer.

        Returns:
            Self for method chaining
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the timer and record metrics.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.end_time = time.time()
        self.success = exc_type is None

        # Calculate duration in seconds
        duration_seconds = self.end_time - self.start_time

        # Record complete request metrics
        self.ai_metrics.record_llm_request(
            provider=self.provider,
            model=self.model,
            operation_type=self.operation_type,
            success=self.success,
            duration_seconds=duration_seconds,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
        )

        # Record error if there was one
        if not self.success:
            self.ai_metrics.record_error(
                error_type=exc_type.__name__,
                provider=self.provider,
                model=self.model,
                operation_type=self.operation_type,
            )

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def record_cost(self, cost_usd: float) -> None:
        """Record cost for the operation.

        Args:
            cost_usd: Cost in USD
        """
        self.cost_usd = cost_usd
