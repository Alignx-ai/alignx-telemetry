"""
AlignX Observability Module

Enhanced observability features for comprehensive monitoring and analysis.
This module provides advanced metrics collection, dashboard integration,
and evaluation capabilities.
"""

from .metrics.collectors import (
    AlignXEnhancedMetricsCollector,
    get_enhanced_metrics_collector,
)
from .metrics.schemas import (
    AlignXEnhancedProviderMetrics,
    StreamingEvent,
    ErrorType,
)
from .metrics.emitters import record_enhanced_llm_metrics

# Import evaluation module
try:
    from ..evaluation import (
        AsyncEvaluator,
        EvaluationPolicy,
        PolicyTitle,
        EvaluationConfig,
        EvaluationConfigurationError,
        configure_evaluation,
        get_evaluator,
        evaluate_agent_response,
    )
except ImportError:
    # Evaluation module not available
    pass

__all__ = [
    # Enhanced metrics system
    "AlignXEnhancedMetricsCollector",
    "get_enhanced_metrics_collector",
    "AlignXEnhancedProviderMetrics",
    "StreamingEvent",
    "ErrorType",
    "record_enhanced_llm_metrics",
    # Evaluation system (if available)
    "AsyncEvaluator",
    "EvaluationPolicy",
    "PolicyTitle",
    "EvaluationConfig",
    "EvaluationConfigurationError",
    "configure_evaluation",
    "get_evaluator",
    "evaluate_agent_response",
]
