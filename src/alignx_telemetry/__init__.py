"""AlignX Telemetry SDK for AI applications."""

import os
from typing import Dict, Optional, Any, Union, List

# Core telemetry components
from alignx_telemetry.auth import AuthManager
from alignx_telemetry.telemetry import TelemetryManager

# Modern API (Phase 2)
from alignx_telemetry.modern_manager import (
    AlignXTelemetryBuilder,
    create_telemetry,
    quick_setup,
)

# Configuration system
from alignx_telemetry.configuration import AlignXConfig

# Decorators for custom tracing
from alignx_telemetry.decorators import (
    trace,
    trace_function, 
    trace_async,
    trace_class,
    span,  # Alternative alias for trace
)

# Unified tracing system
from alignx_telemetry.evaluation import (
    get_evaluator,
)

# Global telemetry manager instance
_global_telemetry_manager: Optional[TelemetryManager] = None

__all__ = [
    # Primary API (Simple - 80% of users)
    "init",                      # Simple initialization
    "instrument_library",        # FastAPI/Flask instrumentation
    "trace",                     # @alignx.trace() decorator
    
    # Modern API (Advanced - 20% of users)  
    "create_telemetry",          # Builder pattern
    "quick_setup",               # Convenience setup
    "AlignXTelemetryBuilder",    # Full builder class
    
    # Advanced decorators
    "trace_function",            # Explicit trace decorator  
    "trace_async",               # Async trace decorator
    "trace_class",               # Class decorator
    "span",                      # Alternative alias for trace
    
    # Direct class access (Advanced users)
    "TelemetryManager",          # Legacy manager
    "AlignXConfig",              # Configuration class
]


def init(
    config: Union[Dict[str, Any], AlignXConfig, str, None] = None,
    service_name: Optional[str] = None,
    license_key: Optional[str] = None,
    service_namespace: Optional[str] = None,
    service_version: Optional[str] = None,
    resource_attributes: Optional[Dict[str, Any]] = None,
    environment: Optional[str] = None,
    enable_cost_tracking: bool = True,
    enable_gpu_monitoring: bool = True,
    enable_content_capture: bool = True,
    gpu_collection_interval: float = 10.0,
    console_export: bool = False,
    log_level: Optional[Union[int, str]] = None,
    metrics_export_interval_millis: int = 60000,
    auto_instrument: bool = True,
    excluded_libraries: Optional[List[str]] = None,
    included_libraries: Optional[List[str]] = None,
    instrumentation_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TelemetryManager:
    """Initialize the AlignX Telemetry SDK.

    This is the main entry point for using the AlignX Telemetry SDK. It initializes
    all telemetry components (tracing, metrics, and logging) with the provided configuration.

    Args:
        config: Configuration dict, AlignXConfig instance, or service name string
        service_name: Name of the service (defaults to OTEL_SERVICE_NAME env var or 'unknown-service')
        license_key: AlignX license key for authentication (optional for OSS usage)
        service_namespace: Namespace of the service (defaults to OTEL_SERVICE_NAMESPACE env var)
        service_version: Version of the service (defaults to OTEL_SERVICE_VERSION env var)
        resource_attributes: Additional resource attributes
        console_export: Whether to export to console for debugging
        log_level: Log level for the root logger
        metrics_export_interval_millis: How often to export metrics in milliseconds
        auto_instrument: Whether to automatically instrument all available libraries
        excluded_libraries: List of libraries to exclude from auto-instrumentation
        included_libraries: List of libraries to specifically include in auto-instrumentation
        instrumentation_config: Configuration for specific instrumentors (library_name -> config dict)

    Returns:
        The initialized TelemetryManager instance

    Raises:
        ValueError: If license key validation fails
    """
    global _global_telemetry_manager
    
    # Handle different config input types
    if isinstance(config, str):
        # Treat string as service name
        service_name = service_name or config
    elif isinstance(config, dict):
        # Extract values from config dict
        service_name = service_name or config.get("service_name")
        license_key = license_key or config.get("license_key")
        service_namespace = service_namespace or config.get("service_namespace")
        service_version = service_version or config.get("service_version")
        environment = environment or config.get("environment")
        # Add more config mappings as needed
    elif isinstance(config, AlignXConfig):
        # Use AlignXConfig instance
        service_name = service_name or config.service_name
        license_key = license_key or config.license_key
        service_namespace = service_namespace or config.service_namespace
        service_version = service_version or config.service_version
        environment = environment or config.environment
    
    # License key is optional - allow OSS usage without license
    license_key = license_key or os.getenv("ALIGNX_LICENSE_KEY")
    
    # Create and initialize telemetry manager
    telemetry_manager = TelemetryManager()
    
    # Handle license key validation if provided
    if license_key:
        try:
            auth_manager = AuthManager()
            auth_result = auth_manager.validate_key(license_key)
            if not auth_result:
                print("WARNING: Invalid license key - continuing in OSS mode")
                license_key = None
            else:
                print(f"SUCCESS: Validated license key for org_id: {auth_result['org_id']}")
                resource_attributes = resource_attributes or {}
                resource_attributes.update({"org_id": auth_result["org_id"]})
        except Exception as e:
            print(f"WARNING: License validation failed: {e} - continuing in OSS mode")
            license_key = None

    # Initialize telemetry manager
    telemetry_manager.initialize(
        service_name=service_name,
        service_namespace=service_namespace,
        service_version=service_version,
        resource_attributes=resource_attributes,
        console_export=console_export,
        log_level=log_level,
        metrics_export_interval_millis=metrics_export_interval_millis,
        environment=environment,
        enable_cost_tracking=enable_cost_tracking,
        enable_gpu_monitoring=enable_gpu_monitoring,
        enable_content_capture=enable_content_capture,
        gpu_collection_interval=gpu_collection_interval,
    )

    # Auto-instrument libraries if enabled
    if auto_instrument:
        telemetry_manager.instrument_all_libraries(
            excluded_libraries=excluded_libraries,
            included_libraries=included_libraries,
            **(instrumentation_config or {}),
        )

    # Initialize evaluator if available
    try:
        get_evaluator()
    except Exception as e:
        # Evaluator is optional, don't fail initialization
        print(f"WARNING: Evaluator initialization failed: {e}")

    # Set as global instance for instrument_library() function
    _global_telemetry_manager = telemetry_manager
    
    return telemetry_manager


def instrument_library(
    library_name: str, 
    app: Optional[Any] = None,
    enabled: Optional[bool] = None,
    **config
) -> bool:
    """Instrument a specific library requiring app instance (FastAPI, Flask).
    
    This function provides global access to library instrumentation,
    particularly for web frameworks that need the application instance.
    
    Args:
        library_name: Name of the library to instrument ("fastapi", "flask", etc.)
        app: Application instance (required for web frameworks)
        enabled: Override whether to enable instrumentation
        **config: Additional configuration for the library
    
    Returns:
        True if instrumentation was successful, False otherwise
        
    Raises:
        RuntimeError: If telemetry not initialized
        
    Examples:
        # Initialize telemetry first
        telemetry = alignx_telemetry.init({"service_name": "my-service"})
        
        # Instrument FastAPI app
        from fastapi import FastAPI
        app = FastAPI()
        alignx_telemetry.instrument_library("fastapi", app=app)
        
        # Instrument Flask app
        from flask import Flask
        app = Flask(__name__)
        alignx_telemetry.instrument_library("flask", app=app)
    """
    global _global_telemetry_manager
    
    if _global_telemetry_manager is None:
        raise RuntimeError(
            "Telemetry not initialized. Call alignx_telemetry.init() first.\n"
            "Example:\n"
            "  import alignx_telemetry\n"
            "  telemetry = alignx_telemetry.init({'service_name': 'my-service'})\n"
            "  alignx_telemetry.instrument_library('fastapi', app=app)"
        )
    
    # Pass app parameter and other config to the telemetry manager
    return _global_telemetry_manager.instrument_library(
        library_name=library_name,
        app=app,
        enabled=enabled,
        **config
    )


def get_global_telemetry() -> Optional[TelemetryManager]:
    """Get the global telemetry manager instance.
    
    Returns:
        The global TelemetryManager instance or None if not initialized
    """
    return _global_telemetry_manager


__version__ = "0.1.0"
