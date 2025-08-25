"""AlignX Bedrock Provider Instrumentation

This module provides comprehensive instrumentation for AWS Bedrock models,
capturing LLM calls, usage metrics, and costs in a provider-first approach.

Supports multiple foundation models available through Bedrock including:
- Anthropic Claude models
- Amazon Titan models
- Cohere Command models
- Meta Llama models
- Mistral AI models
"""

from .instrumentation import AlignXBedrockInstrumentation

__all__ = ["AlignXBedrockInstrumentation"]
