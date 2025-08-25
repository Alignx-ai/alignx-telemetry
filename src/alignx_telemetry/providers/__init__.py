"""AlignX Provider Instrumentation Module

This module provides provider-first instrumentation for various LLM providers,
ensuring universal coverage regardless of the framework used.

Supported Providers:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google GenAI (Gemini models)
- AWS Bedrock (multiple foundation models)

The provider-first approach ensures that all LLM calls are captured at the lowest
level, with framework-specific context layered on top when available.
"""

from .registry import (
    register_provider,
    instrument_provider,
    uninstrument_provider,
    instrument_all_providers,
    uninstrument_all_providers,
    get_registered_providers,
    get_instrumented_providers,
    list_providers,  # Alias for get_registered_providers
    get_instrumentation_status,
    discover_available_providers,
    get_provider_capabilities,
    get_comprehensive_status,
)

# Import base classes for custom provider development
from .base import AlignXBaseProviderInstrumentation, AlignXProviderCallData
from .metrics import AlignXProviderMetrics, AlignXMetricsEmitter
from .context import AlignXFrameworkContext, AlignXTraceContext

# Auto-register all available providers
from .registry import _auto_register_providers

_auto_register_providers()

__all__ = [
    # Registry functions
    "register_provider",
    "instrument_provider",
    "uninstrument_provider",
    "instrument_all_providers",
    "uninstrument_all_providers",
    "list_providers",
    "get_instrumentation_status",
    "discover_available_providers",
    "get_provider_capabilities",
    "get_comprehensive_status",
    # Base classes
    "AlignXBaseProviderInstrumentation",
    "AlignXProviderCallData",
    "AlignXProviderMetrics",
    "AlignXMetricsEmitter",
    "AlignXFrameworkContext",
    "AlignXTraceContext",
]
