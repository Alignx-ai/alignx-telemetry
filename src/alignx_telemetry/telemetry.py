"""Main telemetry manager for AlignX Telemetry SDK."""

import logging
import os
from typing import Optional, Dict, Any, Union, List

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from alignx_telemetry.tracing import TracingInstrumentor
from alignx_telemetry.logging import LoggingInstrumentor
from alignx_telemetry.metrics import (
    MetricsInstrumentor,
    AIMetrics,
    ApplicationMetrics,
    MetricsTimer,
    MetricsConfig,
)
from alignx_telemetry.gpu_metrics import (
    start_gpu_metrics_collection,
    stop_gpu_metrics_collection,
    get_gpu_info,
    is_gpu_available,
)
from alignx_telemetry.instrumentation import (
    instrument_all,
    instrument_specific,
    get_instrumentors,
    is_instrumentation_supported,
)
from alignx_telemetry.context_processor import AlignXContextProcessor
from alignx_telemetry.unified_tracing import AlignXLLMSpanProcessor, AlignXLLMConfig

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Main telemetry manager for AlignX that provides a unified interface for all telemetry components."""

    def __init__(self):
        """Initialize the telemetry manager."""
        self.tracing = TracingInstrumentor()
        self.logging = LoggingInstrumentor()
        self.metrics = MetricsInstrumentor()
        self._initialized = False
        self._ai_metrics = None
        self._app_metrics = None
        self._resource = None
        self._instrumented_libraries = {}
        self._context_processor = AlignXContextProcessor()
        self._gpu_metrics_enabled = False
        self._metrics_config = None
        self._llm_span_processor = None

    def initialize(
        self,
        service_name: Optional[str] = None,
        service_namespace: Optional[str] = None,
        service_version: Optional[str] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        console_export: bool = False,
        log_level: Optional[Union[int, str]] = None,
        metrics_export_interval_millis: int = 60000,
        environment: Optional[str] = None,
        enable_cost_tracking: bool = True,
        enable_gpu_monitoring: bool = True,
        enable_content_capture: bool = True,
        gpu_collection_interval: float = 10.0,
    ) -> None:
        """Initialize all telemetry components with common configuration.

        Args:
            service_name: Name of the service (defaults to OTEL_SERVICE_NAME env var or 'unknown-service')
            service_namespace: Namespace of the service (defaults to OTEL_SERVICE_NAMESPACE env var)
            service_version: Version of the service (defaults to OTEL_SERVICE_VERSION env var)
            resource_attributes: Additional resource attributes
            console_export: Whether to export to console for debugging
            log_level: Log level for the root logger
            metrics_export_interval_millis: How often to export metrics in milliseconds
            environment: Environment name (dev/staging/prod)
            enable_cost_tracking: Whether to track LLM usage costs
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_content_capture: Whether to capture prompt/completion content
            gpu_collection_interval: How often to collect GPU metrics (seconds)
        """
        if self._initialized:
            return

        # Normalize parameters from environment variables
        service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "unknown-service")
        service_namespace = service_namespace or os.getenv("OTEL_SERVICE_NAMESPACE")
        service_version = service_version or os.getenv("OTEL_SERVICE_VERSION")

        # Create metrics configuration
        self._metrics_config = MetricsConfig(
            service_name=service_name,
            environment=environment,
            enable_cost_tracking=enable_cost_tracking,
            enable_gpu_monitoring=enable_gpu_monitoring,
            enable_content_capture=enable_content_capture,
        )

        tracing_disabled = os.getenv("ALIGNX_TRACING_ENABLED", "true").lower() in (
            "false",
            "0",
            "no",
        )
        logging_disabled = os.getenv("ALIGNX_LOGGING_ENABLED", "true").lower() in (
            "false",
            "0",
            "no",
        )
        metrics_disabled = os.getenv("ALIGNX_METRICS_ENABLED", "true").lower() in (
            "false",
            "0",
            "no",
        )

        # Create a single Resource to share across all components
        attrs = {
            "service.name": service_name,
        }

        # Add optional attributes if they exist
        if service_namespace:
            attrs["service.namespace"] = service_namespace
        if service_version:
            attrs["service.version"] = service_version

        # Add metrics configuration to resource attributes
        attrs.update(
            {
                "telemetry_sdk_name": "alignx",
                "telemetry_sdk_version": "0.1.0",
                "alignx.environment": self._metrics_config.environment,
                "alignx.cost_tracking_enabled": str(
                    self._metrics_config.enable_cost_tracking
                ),
                "alignx.gpu_monitoring_enabled": str(
                    self._metrics_config.enable_gpu_monitoring
                ),
            }
        )

        # Add custom resource attributes
        if resource_attributes:
            attrs.update(resource_attributes)

        # Create the shared resource
        self._resource = Resource.create(attrs)
        logger.info(f"Resource: {self._resource}")
        logger.info(f"Tracing enabled: {not tracing_disabled}")
        logger.info(f"Logging enabled: {not logging_disabled}")
        logger.info(f"Metrics enabled: {not metrics_disabled}")
        logger.info(f"Metrics Config: {self._metrics_config.__dict__}")

        if not tracing_disabled:
            # Initialize tracing
            self.tracing.instrument(
                resource=self._resource,
                console_export=console_export,
            )

            # Add AlignX LLM span processor for unified tracing
            self._add_llm_span_processor()

        if not logging_disabled:
            # Initialize logging
            self.logging.instrument(
                resource=self._resource,
                log_level=log_level,
                console_export=console_export,
            )

        if not metrics_disabled:
            # Initialize metrics
            self.metrics.instrument(
                resource=self._resource,
                export_interval_millis=metrics_export_interval_millis,
            )

            # Initialize AI metrics with configuration
            self._ai_metrics = AIMetrics(config=self._metrics_config)

            # Initialize application metrics
            self._app_metrics = ApplicationMetrics()

            # Start GPU monitoring if enabled
            if self._metrics_config.enable_gpu_monitoring:
                gpu_started = start_gpu_metrics_collection(
                    collection_interval=gpu_collection_interval,
                    config=self._metrics_config,
                )
                self._gpu_metrics_enabled = gpu_started
                if gpu_started:
                    logger.info(
                        f"GPU metrics collection started (interval: {gpu_collection_interval}s)"
                    )
                else:
                    logger.info("GPU metrics collection not available or disabled")

        self._initialized = True

    def get_ai_metrics(self) -> Optional[AIMetrics]:
        """Get the AI metrics instance.

        Returns:
            AIMetrics instance or None if not initialized
        """
        return self._ai_metrics

    def get_app_metrics(self) -> Optional[ApplicationMetrics]:
        """Get the application metrics instance.

        Returns:
            ApplicationMetrics instance or None if not initialized
        """
        return self._app_metrics

    def get_metrics_config(self) -> Optional[MetricsConfig]:
        """Get the current metrics configuration.

        Returns:
            MetricsConfig instance or None if not initialized
        """
        return self._metrics_config

    @property
    def tracer(self) -> Optional[trace.Tracer]:
        """Get a tracer instance for creating spans.

        Returns:
            OpenTelemetry tracer instance or None if not initialized
        """
        if not self._initialized:
            return None
        return self.tracing.get_tracer("alignx_telemetry")

    def update_metrics_config(self, **config_updates) -> None:
        """Update metrics configuration.

        Args:
            **config_updates: Configuration parameters to update
        """
        if self._metrics_config:
            for key, value in config_updates.items():
                if hasattr(self._metrics_config, key):
                    setattr(self._metrics_config, key, value)
            logger.info(f"Updated metrics config: {config_updates}")

    def create_metrics_timer(
        self,
        provider: str = "unknown",
        model: str = "unknown",
        operation_type: str = "unknown",
    ) -> Optional[MetricsTimer]:
        """Create a metrics timer for tracking operation duration.

        Args:
            provider: AI provider name
            model: Model name
            operation_type: Type of operation

        Returns:
            MetricsTimer instance or None if AI metrics not available
        """
        if self._ai_metrics is None:
            return None

        return MetricsTimer(
            ai_metrics=self._ai_metrics,
            provider=provider,
            model=model,
            operation_type=operation_type,
        )

    def instrument_library(
        self, library_name: str, enabled: Optional[bool] = None, **kwargs
    ) -> bool:
        """Instrument a specific library.

        Args:
            library_name: Name of the library to instrument
            enabled: Override whether to enable instrumentation
            **kwargs: Additional configuration for the library

        Returns:
            True if instrumentation was successful, False otherwise
        """
        # Pass telemetry manager to instrumentors for AI metrics access
        kwargs["telemetry_manager"] = self

        success = instrument_specific(library_name, enabled=enabled, **kwargs)
        self._instrumented_libraries[library_name] = success

        if success:
            logger.info(f"Successfully instrumented {library_name}")
        else:
            logger.warning(f"Failed to instrument {library_name}")

        return success

    def instrument_all_libraries(
        self,
        excluded_libraries: Optional[List[str]] = None,
        included_libraries: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        """Instrument all available libraries.

        Args:
            excluded_libraries: List of libraries to exclude
            included_libraries: List of libraries to specifically include
            **kwargs: Additional configuration for libraries

        Returns:
            Dictionary mapping library names to success status
        """
        # Pass telemetry manager to instrumentors for AI metrics access
        kwargs["telemetry_manager"] = self

        results = instrument_all(
            excluded_libraries=excluded_libraries,
            included_libraries=included_libraries,
            **kwargs,
        )

        self._instrumented_libraries.update(results)

        # Log results
        successful = [lib for lib, success in results.items() if success]
        failed = [lib for lib, success in results.items() if not success]

        if successful:
            logger.info(f"Successfully instrumented: {', '.join(successful)}")
        if failed:
            logger.info(f"Failed to instrument: {', '.join(failed)}")

        return results

    def get_instrumented_libraries(self) -> Dict[str, bool]:
        """Get the status of all instrumented libraries.

        Returns:
            Dictionary mapping library names to instrumentation status
        """
        return self._instrumented_libraries.copy()

    def is_library_instrumented(self, library_name: str) -> bool:
        """Check if a library is successfully instrumented.

        Args:
            library_name: Name of the library to check

        Returns:
            True if library is instrumented, False otherwise
        """
        return self._instrumented_libraries.get(library_name, False)

    def get_available_instrumentors(self) -> List[str]:
        """Get list of available instrumentor names.

        Returns:
            List of available instrumentor names
        """
        return get_instrumentors()

    def is_instrumentation_supported(self, library_name: str) -> bool:
        """Check if instrumentation is supported for a library.

        Args:
            library_name: Name of the library

        Returns:
            True if instrumentation is supported
        """
        return is_instrumentation_supported(library_name)

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPUs.

        Returns:
            List of GPU information dictionaries
        """
        return get_gpu_info()

    def is_gpu_monitoring_available(self) -> bool:
        """Check if GPU monitoring is available.

        Returns:
            True if GPU monitoring is available
        """
        return is_gpu_available() and self._metrics_config.enable_gpu_monitoring

    def is_gpu_monitoring_enabled(self) -> bool:
        """Check if GPU monitoring is currently enabled.

        Returns:
            True if GPU monitoring is enabled and running
        """
        return self._gpu_metrics_enabled

    def _add_llm_span_processor(self) -> None:
        """Add AlignXLLMSpanProcessor to the tracer provider for unified tracing.

        This processor is responsible for:
        1. Detecting workflow context from middleware
        2. Adding workflow attributes to LLM spans
        3. Recording metrics via the existing AI metrics system
        4. Enabling unified tracing for SaaS workflow correlation
        """
        try:
            if not self.tracing._tracer_provider:
                logger.warning("No tracer provider available for LLM span processor")
                return

            # Create LLM configuration from environment
            llm_config = AlignXLLMConfig.from_env()

            # Create and add the AlignX LLM span processor
            self._llm_span_processor = AlignXLLMSpanProcessor(
                telemetry_manager=self, config=llm_config
            )

            # Add to tracer provider
            self.tracing._tracer_provider.add_span_processor(self._llm_span_processor)

            logger.info("✅ AlignX LLM span processor added for unified tracing")

        except Exception as e:
            logger.error(f"Failed to add LLM span processor: {e}", exc_info=True)

    def shutdown(self) -> None:
        """Shutdown all telemetry components."""
        if not self._initialized:
            return

        logger.info("Shutting down telemetry manager...")

        # Stop GPU metrics collection
        if self._gpu_metrics_enabled:
            stop_gpu_metrics_collection()
            self._gpu_metrics_enabled = False

        # Shutdown LLM span processor
        if self._llm_span_processor:
            self._llm_span_processor.shutdown()
            self._llm_span_processor = None

        # Shutdown metrics
        self.metrics.uninstrument()

        # Shutdown tracing
        self.tracing.shutdown()

        # Shutdown logging
        self.logging.shutdown()

        self._initialized = False
        logger.info("Telemetry manager shutdown complete")
