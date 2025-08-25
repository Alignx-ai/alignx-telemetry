"""
AlignX Unified Tracing Module

This module provides unified tracing capabilities that link LLM operations
to SaaS workflow executions. It bridges the gap between the AlignX backend
workflow engine and customer AI agents.

Key Components:
- WorkflowDetector: Detects when running in workflow context
- AlignXLLMSpanProcessor: Adds workflow context to LLM spans
- Config: Configuration for unified tracing features
- MetricsBridge: Bridges new provider metrics to existing AI metrics

This module replaces the legacy llm_provider_strategy.py while preserving
all critical unified tracing functionality.
"""

from .config import AlignXLLMConfig
from .span_processor import AlignXLLMSpanProcessor
from .workflow_detector import WorkflowDetector
from .metrics_bridge import MetricsBridge
from .semantic_conventions import AlignXSemanticConventions

__all__ = [
    "AlignXLLMConfig",
    "AlignXLLMSpanProcessor",
    "WorkflowDetector",
    "MetricsBridge",
    "AlignXSemanticConventions",
]
