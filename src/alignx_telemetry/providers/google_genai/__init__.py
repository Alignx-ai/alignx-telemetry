"""AlignX Google GenAI Provider Instrumentation

This module provides comprehensive instrumentation for Google's Generative AI models,
capturing LLM calls, usage metrics, and costs in a provider-first approach.

Supports both google-generativeai and google-genai packages.
"""

from .instrumentation import AlignXGoogleGenAIInstrumentation

__all__ = ["AlignXGoogleGenAIInstrumentation"]
