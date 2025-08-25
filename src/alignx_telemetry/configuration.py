"""
Centralized configuration management for AlignX Telemetry SDK.

This module provides standardized configuration patterns, environment variable
handling, and validation across the entire SDK.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standard log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """Standard deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AlignXConfig:
    """Centralized configuration for AlignX Telemetry SDK.

    This configuration class consolidates all settings and provides
    standardized environment variable patterns.
    """

    # Service Identity
    service_name: str = "unknown-service"
    service_version: Optional[str] = None
    service_namespace: Optional[str] = None
    environment: str = "development"

    # Core Telemetry Settings
    console_export: bool = False
    log_level: Optional[str] = None
    metrics_export_interval_millis: int = 60000
    batch_span_processor_queue_size: int = 2048

    # Feature Flags
    enable_cost_tracking: bool = True
    enable_gpu_monitoring: bool = True
    enable_content_capture: bool = True
    enable_unified_tracing: bool = True
    enable_auto_instrumentation: bool = True

    # Performance Settings
    gpu_collection_interval: float = 10.0
    content_capture_max_length: int = 1024

    # OTLP Export Settings
    otlp_endpoint_traces: Optional[str] = None
    otlp_endpoint_metrics: Optional[str] = None
    otlp_endpoint_logs: Optional[str] = None
    otlp_headers: Dict[str, str] = field(default_factory=dict)
    otlp_timeout: int = 10

    # SaaS/Enterprise Settings (optional)
    license_key: Optional[str] = None
    organization_id: Optional[str] = None
    correlation_enabled: bool = False

    # Provider-Specific Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None

    # Advanced Settings
    fail_silently: bool = True
    log_errors: bool = True
    enable_memory_management: bool = True
    debug_mode: bool = False

    # Custom attributes
    custom_resource_attributes: Dict[str, Any] = field(default_factory=dict)
    custom_tags: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AlignXConfig":
        """Create configuration from environment variables.

        Uses standardized ALIGNX_* environment variable patterns.

        Returns:
            AlignXConfig instance populated from environment
        """
        return cls(
            # Service Identity
            service_name=os.getenv("ALIGNX_SERVICE_NAME", "unknown-service"),
            service_version=os.getenv("ALIGNX_SERVICE_VERSION"),
            service_namespace=os.getenv("ALIGNX_SERVICE_NAMESPACE"),
            environment=os.getenv("ALIGNX_ENVIRONMENT", "development"),
            # Core Telemetry Settings
            console_export=_parse_bool(os.getenv("ALIGNX_CONSOLE_EXPORT", "false")),
            log_level=os.getenv("ALIGNX_LOG_LEVEL"),
            metrics_export_interval_millis=_parse_int(
                os.getenv("ALIGNX_METRICS_EXPORT_INTERVAL", "60000")
            ),
            batch_span_processor_queue_size=_parse_int(
                os.getenv("ALIGNX_BATCH_QUEUE_SIZE", "2048")
            ),
            # Feature Flags
            enable_cost_tracking=_parse_bool(
                os.getenv("ALIGNX_COST_TRACKING_ENABLED", "true")
            ),
            enable_gpu_monitoring=_parse_bool(
                os.getenv("ALIGNX_GPU_MONITORING_ENABLED", "true")
            ),
            enable_content_capture=_parse_bool(
                os.getenv("ALIGNX_CONTENT_CAPTURE_ENABLED", "true")
            ),
            enable_unified_tracing=_parse_bool(
                os.getenv("ALIGNX_UNIFIED_TRACING_ENABLED", "true")
            ),
            enable_auto_instrumentation=_parse_bool(
                os.getenv("ALIGNX_AUTO_INSTRUMENTATION_ENABLED", "true")
            ),
            # Performance Settings
            gpu_collection_interval=_parse_float(
                os.getenv("ALIGNX_GPU_COLLECTION_INTERVAL", "10.0")
            ),
            content_capture_max_length=_parse_int(
                os.getenv("ALIGNX_CONTENT_CAPTURE_MAX_LENGTH", "1024")
            ),
            # OTLP Export Settings
            otlp_endpoint_traces=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otlp_endpoint_metrics=os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otlp_endpoint_logs=os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otlp_headers=_parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")),
            otlp_timeout=_parse_int(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "10")),
            # SaaS/Enterprise Settings
            license_key=os.getenv("ALIGNX_LICENSE_KEY"),
            organization_id=os.getenv("ALIGNX_ORGANIZATION_ID"),
            correlation_enabled=_parse_bool(
                os.getenv("ALIGNX_CORRELATION_ENABLED", "false")
            ),
            # Provider-Specific Settings
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION"),
            # Advanced Settings
            fail_silently=_parse_bool(os.getenv("ALIGNX_FAIL_SILENTLY", "true")),
            log_errors=_parse_bool(os.getenv("ALIGNX_LOG_ERRORS", "true")),
            enable_memory_management=_parse_bool(
                os.getenv("ALIGNX_ENABLE_MEMORY_MANAGEMENT", "true")
            ),
            debug_mode=_parse_bool(os.getenv("ALIGNX_DEBUG_MODE", "false")),
            # Custom attributes and tags
            custom_resource_attributes=_parse_custom_attributes(
                os.getenv("ALIGNX_CUSTOM_ATTRIBUTES", "")
            ),
            custom_tags=_parse_custom_tags(os.getenv("ALIGNX_CUSTOM_TAGS", "")),
        )

    def validate(self) -> Dict[str, List[str]]:
        """Validate the configuration.

        Returns:
            Dictionary with validation results:
            - "errors": List of validation errors
            - "warnings": List of validation warnings
        """
        errors = []
        warnings = []

        # Required field validation
        if not self.service_name or self.service_name == "unknown-service":
            warnings.append("service_name should be set to a meaningful value")

        # Range validation
        if self.metrics_export_interval_millis <= 0:
            errors.append("metrics_export_interval_millis must be positive")

        if self.batch_span_processor_queue_size <= 0:
            errors.append("batch_span_processor_queue_size must be positive")

        if self.gpu_collection_interval <= 0:
            errors.append("gpu_collection_interval must be positive")

        if self.content_capture_max_length <= 0:
            errors.append("content_capture_max_length must be positive")

        # Environment validation
        valid_environments = [env.value for env in Environment]
        if self.environment not in valid_environments:
            warnings.append(
                f"environment '{self.environment}' not in recommended values: {valid_environments}"
            )

        # Feature consistency validation
        if self.enable_unified_tracing and not self.license_key:
            warnings.append(
                "unified_tracing enabled but no license_key provided - will work in OSS mode only"
            )

        if self.enable_gpu_monitoring and self.environment == "production":
            warnings.append(
                "GPU monitoring enabled in production - ensure this is intended"
            )

        if self.console_export and self.environment == "production":
            warnings.append(
                "console_export enabled in production - may impact performance"
            )

        # API key validation
        if self.enable_auto_instrumentation:
            missing_keys = []
            if not self.openai_api_key:
                missing_keys.append("OPENAI_API_KEY")
            if not self.anthropic_api_key:
                missing_keys.append("ANTHROPIC_API_KEY")
            if not self.google_api_key:
                missing_keys.append("GOOGLE_API_KEY")

            if missing_keys:
                warnings.append(
                    f"Auto-instrumentation enabled but missing API keys: {missing_keys}"
                )

        return {"errors": errors, "warnings": warnings}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def to_env_vars(self) -> Dict[str, str]:
        """Convert configuration to environment variable format.

        Returns:
            Dictionary of environment variables
        """
        env_vars = {}

        # Service Identity
        env_vars["ALIGNX_SERVICE_NAME"] = self.service_name
        if self.service_version:
            env_vars["ALIGNX_SERVICE_VERSION"] = self.service_version
        if self.service_namespace:
            env_vars["ALIGNX_SERVICE_NAMESPACE"] = self.service_namespace
        env_vars["ALIGNX_ENVIRONMENT"] = self.environment

        # Feature Flags
        env_vars["ALIGNX_COST_TRACKING_ENABLED"] = str(
            self.enable_cost_tracking
        ).lower()
        env_vars["ALIGNX_GPU_MONITORING_ENABLED"] = str(
            self.enable_gpu_monitoring
        ).lower()
        env_vars["ALIGNX_CONTENT_CAPTURE_ENABLED"] = str(
            self.enable_content_capture
        ).lower()
        env_vars["ALIGNX_UNIFIED_TRACING_ENABLED"] = str(
            self.enable_unified_tracing
        ).lower()
        env_vars["ALIGNX_AUTO_INSTRUMENTATION_ENABLED"] = str(
            self.enable_auto_instrumentation
        ).lower()

        # Debug Settings
        env_vars["ALIGNX_CONSOLE_EXPORT"] = str(self.console_export).lower()
        env_vars["ALIGNX_DEBUG_MODE"] = str(self.debug_mode).lower()

        # Performance Settings
        env_vars["ALIGNX_GPU_COLLECTION_INTERVAL"] = str(self.gpu_collection_interval)
        env_vars["ALIGNX_CONTENT_CAPTURE_MAX_LENGTH"] = str(
            self.content_capture_max_length
        )
        env_vars["ALIGNX_METRICS_EXPORT_INTERVAL"] = str(
            self.metrics_export_interval_millis
        )
        env_vars["ALIGNX_BATCH_QUEUE_SIZE"] = str(self.batch_span_processor_queue_size)

        # Optional settings
        if self.license_key:
            env_vars["ALIGNX_LICENSE_KEY"] = self.license_key
        if self.organization_id:
            env_vars["ALIGNX_ORGANIZATION_ID"] = self.organization_id
        if self.log_level:
            env_vars["ALIGNX_LOG_LEVEL"] = self.log_level

        return env_vars

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ["development", "dev"]

    def has_license(self) -> bool:
        """Check if license key is available."""
        return bool(self.license_key)

    def has_provider_credentials(self) -> bool:
        """Check if any provider credentials are available."""
        return any(
            [
                self.openai_api_key,
                self.anthropic_api_key,
                self.google_api_key,
                self.aws_access_key_id,
            ]
        )


# Utility functions for parsing environment variables
def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: str) -> int:
    """Parse integer from string with error handling."""
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value: {value}")
        return 0


def _parse_float(value: str) -> float:
    """Parse float from string with error handling."""
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value: {value}")
        return 0.0


def _parse_headers(value: str) -> Dict[str, str]:
    """Parse OTLP headers from string format."""
    headers = {}
    if not value:
        return headers

    try:
        for header in value.split(","):
            if "=" in header:
                key, val = header.split("=", 1)
                headers[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f"Failed to parse headers '{value}': {e}")

    return headers


def _parse_custom_attributes(value: str) -> Dict[str, Any]:
    """Parse custom attributes from string format."""
    attributes = {}
    if not value:
        return attributes

    try:
        for attr in value.split(","):
            if "=" in attr:
                key, val = attr.split("=", 1)
                attributes[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f"Failed to parse custom attributes '{value}': {e}")

    return attributes


def _parse_custom_tags(value: str) -> List[str]:
    """Parse custom tags from string format."""
    if not value:
        return []

    return [tag.strip() for tag in value.split(",") if tag.strip()]


# Configuration factory functions
def create_config(**overrides) -> AlignXConfig:
    """Create configuration with optional overrides.

    Args:
        **overrides: Configuration field overrides

    Returns:
        AlignXConfig instance
    """
    config = AlignXConfig.from_env()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration field: {key}")

    return config


def get_default_config() -> AlignXConfig:
    """Get default configuration for development.

    Returns:
        AlignXConfig with development defaults
    """
    return AlignXConfig(
        service_name="alignx-development",
        environment="development",
        console_export=True,
        debug_mode=True,
        enable_unified_tracing=False,  # Disabled by default in dev
    )


def get_production_config() -> AlignXConfig:
    """Get recommended configuration for production.

    Returns:
        AlignXConfig with production defaults
    """
    return AlignXConfig.from_env()


# Export configuration classes and functions
__all__ = [
    "AlignXConfig",
    "LogLevel",
    "Environment",
    "create_config",
    "get_default_config",
    "get_production_config",
]
