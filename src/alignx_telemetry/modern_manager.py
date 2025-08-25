"""
Modern AlignX Telemetry Manager with improved developer experience.

This module provides a modern, fluent API for configuring and managing telemetry
while maintaining backward compatibility with the existing TelemetryManager.
"""

import logging
import os
from typing import Optional, Dict, Any, Union, List, Callable
from contextlib import contextmanager

from .telemetry import TelemetryManager
from .providers import (
    instrument_all_providers,
    list_providers,
    get_instrumentation_status,
)

logger = logging.getLogger(__name__)


class AlignXTelemetryBuilder:
    """Fluent builder for AlignX telemetry configuration.

    This builder provides a modern, chainable API for configuring telemetry
    components with intelligent defaults and validation.

    Example:
        ```python
        telemetry = (
            AlignXTelemetryBuilder()
            .service("my-ai-agent", "1.0.0")
            .environment("production")
            .enable_cost_tracking()
            .enable_gpu_monitoring()
            .enable_unified_tracing()
            .build()
        )
        ```
    """

    def __init__(self):
        """Initialize the builder with default values."""
        self._config = {
            # Service identity
            "service_name": None,
            "service_namespace": None,
            "service_version": None,
            "environment": None,
            # Core telemetry settings
            "console_export": False,
            "log_level": None,
            "metrics_export_interval_millis": 60000,
            # Feature flags
            "enable_cost_tracking": True,
            "enable_gpu_monitoring": True,
            "enable_content_capture": True,
            "enable_unified_tracing": True,
            # Performance settings
            "gpu_collection_interval": 10.0,
            "batch_span_processor_queue_size": 2048,
            # Custom attributes and hooks
            "resource_attributes": {},
            "pre_init_hooks": [],
            "post_init_hooks": [],
            # Provider instrumentation
            "auto_instrument_providers": True,
            "provider_config": {},
        }

    def service(
        self, name: str, version: Optional[str] = None, namespace: Optional[str] = None
    ) -> "AlignXTelemetryBuilder":
        """Configure service identity.

        Args:
            name: Service name (required)
            version: Service version (optional)
            namespace: Service namespace (optional)

        Returns:
            Self for method chaining
        """
        self._config["service_name"] = name
        if version:
            self._config["service_version"] = version
        if namespace:
            self._config["service_namespace"] = namespace
        return self

    def environment(self, env: str) -> "AlignXTelemetryBuilder":
        """Set the deployment environment.

        Args:
            env: Environment name (e.g., "dev", "staging", "production")

        Returns:
            Self for method chaining
        """
        self._config["environment"] = env
        return self

    def console_export(self, enabled: bool = True) -> "AlignXTelemetryBuilder":
        """Enable/disable console export for debugging.

        Args:
            enabled: Whether to export to console

        Returns:
            Self for method chaining
        """
        self._config["console_export"] = enabled
        return self

    def log_level(self, level: Union[int, str]) -> "AlignXTelemetryBuilder":
        """Set the logging level.

        Args:
            level: Log level (e.g., logging.INFO, "INFO")

        Returns:
            Self for method chaining
        """
        self._config["log_level"] = level
        return self

    def metrics_interval(self, millis: int) -> "AlignXTelemetryBuilder":
        """Set metrics export interval.

        Args:
            millis: Export interval in milliseconds

        Returns:
            Self for method chaining
        """
        self._config["metrics_export_interval_millis"] = millis
        return self

    def enable_cost_tracking(self, enabled: bool = True) -> "AlignXTelemetryBuilder":
        """Enable/disable LLM cost tracking.

        Args:
            enabled: Whether to track LLM costs

        Returns:
            Self for method chaining
        """
        self._config["enable_cost_tracking"] = enabled
        return self

    def enable_gpu_monitoring(
        self, enabled: bool = True, interval: float = 10.0
    ) -> "AlignXTelemetryBuilder":
        """Enable/disable GPU monitoring.

        Args:
            enabled: Whether to monitor GPU metrics
            interval: Collection interval in seconds

        Returns:
            Self for method chaining
        """
        self._config["enable_gpu_monitoring"] = enabled
        self._config["gpu_collection_interval"] = interval
        return self

    def enable_content_capture(self, enabled: bool = True) -> "AlignXTelemetryBuilder":
        """Enable/disable LLM content capture.

        Args:
            enabled: Whether to capture prompts/completions

        Returns:
            Self for method chaining
        """
        self._config["enable_content_capture"] = enabled
        return self

    def enable_unified_tracing(self, enabled: bool = True) -> "AlignXTelemetryBuilder":
        """Enable/disable unified workflow tracing.

        Args:
            enabled: Whether to enable SaaS workflow correlation

        Returns:
            Self for method chaining
        """
        self._config["enable_unified_tracing"] = enabled
        return self

    def batch_queue_size(self, size: int) -> "AlignXTelemetryBuilder":
        """Set batch span processor queue size.

        Args:
            size: Maximum queue size for span processor

        Returns:
            Self for method chaining
        """
        self._config["batch_span_processor_queue_size"] = size
        return self

    def resource_attributes(self, **attributes) -> "AlignXTelemetryBuilder":
        """Add custom resource attributes.

        Args:
            **attributes: Custom attributes to add to resource

        Returns:
            Self for method chaining
        """
        self._config["resource_attributes"].update(attributes)
        return self

    def provider_instrumentation(
        self, enabled: bool = True, **provider_configs
    ) -> "AlignXTelemetryBuilder":
        """Configure provider instrumentation.

        Args:
            enabled: Whether to auto-instrument providers
            **provider_configs: Provider-specific configurations

        Returns:
            Self for method chaining
        """
        self._config["auto_instrument_providers"] = enabled
        self._config["provider_config"].update(provider_configs)
        return self

    def add_hook(
        self, hook: Callable, when: str = "post_init"
    ) -> "AlignXTelemetryBuilder":
        """Add initialization hooks.

        Args:
            hook: Function to call during initialization
            when: When to call ("pre_init" or "post_init")

        Returns:
            Self for method chaining
        """
        if when == "pre_init":
            self._config["pre_init_hooks"].append(hook)
        elif when == "post_init":
            self._config["post_init_hooks"].append(hook)
        else:
            raise ValueError(f"Invalid hook timing: {when}")
        return self

    def from_env(self) -> "AlignXTelemetryBuilder":
        """Configure from environment variables.

        This method reads standard AlignX environment variables and applies
        them to the configuration.

        Returns:
            Self for method chaining
        """
        # Service identity from environment
        if service_name := os.getenv("ALIGNX_SERVICE_NAME"):
            self._config["service_name"] = service_name
        if service_version := os.getenv("ALIGNX_SERVICE_VERSION"):
            self._config["service_version"] = service_version
        if service_namespace := os.getenv("ALIGNX_SERVICE_NAMESPACE"):
            self._config["service_namespace"] = service_namespace
        if environment := os.getenv("ALIGNX_ENVIRONMENT"):
            self._config["environment"] = environment

        # Feature flags from environment
        self._config["enable_cost_tracking"] = (
            os.getenv("ALIGNX_COST_TRACKING_ENABLED", "true").lower() == "true"
        )
        self._config["enable_gpu_monitoring"] = (
            os.getenv("ALIGNX_GPU_MONITORING_ENABLED", "true").lower() == "true"
        )
        self._config["enable_content_capture"] = (
            os.getenv("ALIGNX_CONTENT_CAPTURE_ENABLED", "true").lower() == "true"
        )
        self._config["enable_unified_tracing"] = (
            os.getenv("ALIGNX_UNIFIED_TRACING_ENABLED", "true").lower() == "true"
        )

        # Debug settings from environment
        self._config["console_export"] = (
            os.getenv("ALIGNX_CONSOLE_EXPORT", "false").lower() == "true"
        )

        # Performance settings from environment
        if gpu_interval := os.getenv("ALIGNX_GPU_COLLECTION_INTERVAL"):
            try:
                self._config["gpu_collection_interval"] = float(gpu_interval)
            except ValueError:
                logger.warning(f"Invalid GPU collection interval: {gpu_interval}")

        if queue_size := os.getenv("ALIGNX_BATCH_QUEUE_SIZE"):
            try:
                self._config["batch_span_processor_queue_size"] = int(queue_size)
            except ValueError:
                logger.warning(f"Invalid batch queue size: {queue_size}")

        return self

    def validate(self) -> Dict[str, List[str]]:
        """Validate the current configuration.

        Returns:
            Dictionary with validation results:
            - "errors": List of validation errors
            - "warnings": List of validation warnings
        """
        errors = []
        warnings = []

        # Required fields validation
        if not self._config["service_name"]:
            errors.append("service_name is required")

        # Range validation
        if self._config["gpu_collection_interval"] <= 0:
            errors.append("gpu_collection_interval must be positive")

        if self._config["metrics_export_interval_millis"] <= 0:
            errors.append("metrics_export_interval_millis must be positive")

        if self._config["batch_span_processor_queue_size"] <= 0:
            errors.append("batch_span_processor_queue_size must be positive")

        # Warnings for common issues
        if not self._config["environment"]:
            warnings.append(
                "environment not set - consider setting for better observability"
            )

        if (
            self._config["console_export"]
            and self._config["environment"] == "production"
        ):
            warnings.append(
                "console_export enabled in production - may impact performance"
            )

        return {"errors": errors, "warnings": warnings}

    def build(self) -> "AlignXTelemetryManager":
        """Build and initialize the telemetry manager.

        Returns:
            Configured and initialized AlignXTelemetryManager

        Raises:
            ValueError: If configuration validation fails
        """
        # Validate configuration
        validation = self.validate()
        if validation["errors"]:
            raise ValueError(f"Configuration errors: {', '.join(validation['errors'])}")

        # Log warnings
        for warning in validation["warnings"]:
            logger.warning(f"Configuration warning: {warning}")

        # Create the telemetry manager
        manager = AlignXTelemetryManager(self._config)

        # Execute pre-init hooks
        for hook in self._config["pre_init_hooks"]:
            try:
                hook(manager)
            except Exception as e:
                logger.error(f"Pre-init hook failed: {e}", exc_info=True)

        # Initialize the manager
        manager.initialize()

        # Execute post-init hooks
        for hook in self._config["post_init_hooks"]:
            try:
                hook(manager)
            except Exception as e:
                logger.error(f"Post-init hook failed: {e}", exc_info=True)

        return manager


class AlignXTelemetryManager:
    """Modern telemetry manager with enhanced developer experience.

    This manager wraps the existing TelemetryManager while providing:
    - Context manager support for automatic cleanup
    - Enhanced configuration options
    - Better integration with provider system
    - Modern API patterns
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the modern telemetry manager.

        Args:
            config: Configuration dictionary from builder
        """
        self._config = config
        self._telemetry_manager = TelemetryManager()
        self._initialized = False
        self._provider_instrumentations = {}

    def initialize(self) -> "AlignXTelemetryManager":
        """Initialize all telemetry components.

        Returns:
            Self for method chaining
        """
        if self._initialized:
            logger.debug("AlignX telemetry already initialized")
            return self

        # Initialize the underlying telemetry manager
        self._telemetry_manager.initialize(
            service_name=self._config["service_name"],
            service_namespace=self._config["service_namespace"],
            service_version=self._config["service_version"],
            environment=self._config["environment"],
            resource_attributes=self._config["resource_attributes"],
            console_export=self._config["console_export"],
            log_level=self._config["log_level"],
            metrics_export_interval_millis=self._config[
                "metrics_export_interval_millis"
            ],
            enable_cost_tracking=self._config["enable_cost_tracking"],
            enable_gpu_monitoring=self._config["enable_gpu_monitoring"],
            enable_content_capture=self._config["enable_content_capture"],
            gpu_collection_interval=self._config["gpu_collection_interval"],
        )

        # Auto-instrument providers if enabled
        if self._config["auto_instrument_providers"]:
            self._setup_provider_instrumentation()

        self._initialized = True
        logger.info("✅ AlignX telemetry initialization complete")

        return self

    def _setup_provider_instrumentation(self) -> None:
        """Set up provider instrumentation with enhanced configuration."""
        try:
            # Instrument all available providers
            results = instrument_all_providers(
                telemetry_manager=self._telemetry_manager,
                **self._config["provider_config"],
            )

            self._provider_instrumentations = results

            # Log results
            successful = sum(1 for success in results.values() if success)
            total = len(results)

            logger.info(f"✅ Provider instrumentation: {successful}/{total} successful")

            if successful < total:
                failed_providers = [
                    name for name, success in results.items() if not success
                ]
                logger.warning(f"Failed to instrument providers: {failed_providers}")

        except Exception as e:
            logger.error(f"Provider instrumentation failed: {e}", exc_info=True)

    @property
    def telemetry_manager(self) -> TelemetryManager:
        """Get the underlying TelemetryManager for backward compatibility."""
        return self._telemetry_manager

    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

    @property
    def provider_status(self) -> Dict[str, Any]:
        """Get detailed provider instrumentation status."""
        return {
            "available_providers": list_providers(),
            "instrumentation_status": get_instrumentation_status(),
            "instrumentation_results": self._provider_instrumentations.copy(),
        }

    # Client requirement #3: AI Metrics access
    def get_ai_metrics(self):
        """Get the AI metrics instance for recording custom metrics.
        
        Returns:
            AIMetrics instance or None if not initialized
            
        Example:
            telemetry = alignx_telemetry.init({"service_name": "my-service"})
            ai_metrics = telemetry.get_ai_metrics()
            ai_metrics.record_db_request(
                db_system="lancedb",
                operation="search_insurance_plans",
                success=True,
                duration_seconds=0.1
            )
        """
        return self._telemetry_manager.get_ai_metrics()

    # Client requirement #4: Tracer access  
    @property
    def tracer(self):
        """Get a tracer instance for creating custom spans.
        
        Returns:
            OpenTelemetry tracer instance or None if not initialized
            
        Example:
            telemetry = alignx_telemetry.init({"service_name": "my-service"})
            tracer = telemetry.tracer
            with tracer.start_as_current_span("business_logic") as span:
                span.set_attribute("operation_type", "data_processing")
                # Your business logic here
        """
        return self._telemetry_manager.tracer

    # Client requirement #2: Library instrumentation
    def instrument_library(self, library_name: str, app=None, **kwargs) -> bool:
        """Instrument a specific library.
        
        Args:
            library_name: Name of the library to instrument  
            app: Application instance (required for FastAPI, Flask)
            **kwargs: Additional configuration for the library
            
        Returns:
            True if instrumentation was successful, False otherwise
            
        Example:
            # FastAPI
            from fastapi import FastAPI
            app = FastAPI()
            telemetry.instrument_library("fastapi", app=app)
            
            # Flask  
            from flask import Flask
            app = Flask(__name__)
            telemetry.instrument_library("flask", app=app)
        """
        return self._telemetry_manager.instrument_library(
            library_name=library_name, 
            app=app, 
            **kwargs
        )

    def __enter__(self) -> "AlignXTelemetryManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown all telemetry components."""
        if not self._initialized:
            return

        logger.info("Shutting down AlignX telemetry...")

        # Shutdown the underlying telemetry manager
        self._telemetry_manager.shutdown()

        self._initialized = False
        logger.info("✅ AlignX telemetry shutdown complete")


# Convenience factory functions
def create_telemetry() -> AlignXTelemetryBuilder:
    """Create a new telemetry builder.

    Returns:
        New AlignXTelemetryBuilder instance
    """
    return AlignXTelemetryBuilder()


def quick_setup(
    service_name: str, environment: str = "development", **kwargs
) -> AlignXTelemetryManager:
    """Quick setup for common use cases.

    Args:
        service_name: Name of the service
        environment: Deployment environment
        **kwargs: Additional configuration options

    Returns:
        Initialized AlignXTelemetryManager
    """
    return (
        create_telemetry()
        .from_env()
        .service(service_name)
        .environment(environment)
        .build()
    )


@contextmanager
def telemetry_context(service_name: str, environment: str = "development", **kwargs):
    """Context manager for automatic telemetry setup and cleanup.

    Args:
        service_name: Name of the service
        environment: Deployment environment
        **kwargs: Additional configuration options

    Yields:
        Initialized AlignXTelemetryManager

    Example:
        ```python
        with telemetry_context("my-agent", "production") as telemetry:
            # Your code here
            pass
        # Automatic cleanup
        ```
    """
    telemetry = quick_setup(service_name, environment, **kwargs)
    try:
        yield telemetry
    finally:
        telemetry.shutdown()


# Export the modern API
__all__ = [
    "AlignXTelemetryBuilder",
    "AlignXTelemetryManager",
    "create_telemetry",
    "quick_setup",
    "telemetry_context",
]
