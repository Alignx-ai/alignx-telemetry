"""AlignX OpenAI Provider Instrumentation.

This module provides comprehensive instrumentation for OpenAI's Python SDK,
capturing ALL OpenAI API calls regardless of how they're invoked (vanilla SDK,
LangChain, LlamaIndex, CrewAI, AutoGen, etc.).

Features:
- Universal coverage via lowest-level request instrumentation
- Streaming response support with real-time metrics
- Cost calculation using up-to-date pricing models
- Custom base URL support (Azure OpenAI, OpenAI-compatible APIs)
- Framework context integration
- Zero duplicate spans

Example Usage:
    from alignx_telemetry.providers.openai import AlignXOpenAIInstrumentation

    # Instrument OpenAI globally
    instrumentation = AlignXOpenAIInstrumentation(telemetry_manager)
    success = instrumentation.instrument()

    # Now all OpenAI calls will be traced
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from .instrumentation import AlignXOpenAIInstrumentation
from .metrics_extractor import OpenAIMetricsExtractor
from .response_parser import OpenAIResponseParser
from .cost_calculator import OpenAICostCalculator

__all__ = [
    "AlignXOpenAIInstrumentation",
    "OpenAIMetricsExtractor",
    "OpenAIResponseParser",
    "OpenAICostCalculator",
]
