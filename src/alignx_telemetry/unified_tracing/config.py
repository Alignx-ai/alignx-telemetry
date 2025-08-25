"""Configuration for AlignX unified tracing capabilities."""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class AlignXLLMConfig:
    """Configuration for AlignX LLM instrumentation enhancements.

    This configuration controls how LLM spans are enhanced with AlignX-specific
    context, metrics, and correlation information for unified workflow tracing.
    """

    # Core settings
    enabled: bool = True

    # Cost calculation settings
    calculate_costs: bool = True
    capture_cost: bool = True  # Alias for calculate_costs for compatibility
    pricing_model: str = "openlit"  # 'openlit' or 'custom'
    custom_pricing_file: Optional[str] = None

    # Content capture settings
    capture_content: bool = True
    capture_max_length: int = 1024

    # Metric collection settings
    collect_metrics: bool = True
    metric_dimensions: Dict[str, str] = field(default_factory=dict)

    # Enhanced span attributes
    add_enhanced_attributes: bool = True

    # Unified tracing settings
    enable_workflow_correlation: bool = True
    add_workflow_attributes: bool = True

    # Performance and reliability settings
    fail_silently: bool = True
    log_errors: bool = True
    enable_memory_management: bool = True

    # Custom hooks
    pre_process_hook: Optional[Callable] = None
    post_process_hook: Optional[Callable] = None

    @classmethod
    def from_env(cls) -> "AlignXLLMConfig":
        """Create configuration from environment variables.

        Returns:
            AlignXLLMConfig instance configured from environment variables
        """
        return cls(
            enabled=os.getenv("ALIGNX_LLM_ENABLED", "true").lower() == "true",
            calculate_costs=os.getenv("ALIGNX_LLM_CALCULATE_COSTS", "true").lower()
            == "true",
            capture_cost=os.getenv("ALIGNX_LLM_CAPTURE_COST", "true").lower() == "true",
            pricing_model=os.getenv("ALIGNX_LLM_PRICING_MODEL", "openlit"),
            custom_pricing_file=os.getenv("ALIGNX_LLM_CUSTOM_PRICING_FILE"),
            capture_content=os.getenv("ALIGNX_LLM_CAPTURE_CONTENT", "true").lower()
            == "true",
            capture_max_length=int(os.getenv("ALIGNX_LLM_CAPTURE_MAX_LENGTH", "1024")),
            collect_metrics=os.getenv("ALIGNX_LLM_COLLECT_METRICS", "true").lower()
            == "true",
            add_enhanced_attributes=os.getenv(
                "ALIGNX_LLM_ADD_ENHANCED_ATTRIBUTES", "true"
            ).lower()
            == "true",
            enable_workflow_correlation=os.getenv(
                "ALIGNX_ENABLE_WORKFLOW_CORRELATION", "true"
            ).lower()
            == "true",
            add_workflow_attributes=os.getenv(
                "ALIGNX_ADD_WORKFLOW_ATTRIBUTES", "true"
            ).lower()
            == "true",
            fail_silently=os.getenv("ALIGNX_LLM_FAIL_SILENTLY", "true").lower()
            == "true",
            log_errors=os.getenv("ALIGNX_LLM_LOG_ERRORS", "true").lower() == "true",
            enable_memory_management=os.getenv(
                "ALIGNX_LLM_ENABLE_MEMORY_MANAGEMENT", "true"
            ).lower()
            == "true",
        )


# AlignX-specific config parameters that should be filtered out when passing to other systems
ALIGNX_CONFIG_PARAMS = {
    "enabled",
    "telemetry_manager",
    "alignx_correlation_enabled",
    "alignx_workflow_id",
    "alignx_node_name",
    "app",  # Remove app to avoid duplicate parameter
}


def filter_otel_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out AlignX-specific parameters from config before passing to OpenTelemetry.

    Args:
        config: Raw config dictionary

    Returns:
        Filtered config safe for OpenTelemetry instrumentors
    """
    return {k: v for k, v in config.items() if k not in ALIGNX_CONFIG_PARAMS}
