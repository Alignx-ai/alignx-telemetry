"""AlignX Anthropic Metrics Extractor

Extracts structured data and metadata from Anthropic API requests for tracing and metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Union

try:
    from anthropic.types import MessageParam

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    MessageParam = None

from ..base import AlignXProviderCallData

logger = logging.getLogger(__name__)


class AnthropicMetricsExtractor:
    """Extracts metrics and metadata from Anthropic API calls."""

    def extract_call_data(
        self, args: tuple, kwargs: dict, operation: str = "unknown"
    ) -> AlignXProviderCallData:
        """Extract structured call data from Anthropic API arguments."""

        # Determine the model being used
        model = self._extract_model(kwargs)

        # Extract operation-specific inputs
        inputs = self._extract_inputs(kwargs, operation)

        # Extract sanitized arguments for debugging
        sanitized_args = self._sanitize_args(kwargs)

        return AlignXProviderCallData(
            provider="anthropic",
            model=model,
            operation=operation,
            inputs=inputs,
            args=sanitized_args,
        )

    def _extract_model(self, kwargs: dict) -> str:
        """Extract the model name from API call arguments."""
        model = kwargs.get("model", "unknown")

        # Normalize model names for consistency
        model_mapping = {
            "claude-3-opus-20240229": "claude-3-opus",
            "claude-3-sonnet-20240229": "claude-3-sonnet",
            "claude-3-haiku-20240307": "claude-3-haiku",
            "claude-2.1": "claude-2.1",
            "claude-2.0": "claude-2.0",
            "claude-instant-1.2": "claude-instant-1.2",
        }

        return model_mapping.get(model, model)

    def _extract_inputs(self, kwargs: dict, operation: str) -> Dict[str, Any]:
        """Extract input data from API call arguments."""

        if operation in ["messages.create", "messages.stream"]:
            return self._extract_messages_inputs(kwargs)
        elif operation == "completions.create":
            return self._extract_completions_inputs(kwargs)
        else:
            return {}

    def _extract_messages_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from Messages API call."""
        inputs = {}

        # Extract messages
        messages = kwargs.get("messages", [])
        if messages:
            inputs["message_count"] = len(messages)
            inputs["system_message"] = kwargs.get("system")

            # Extract content types and lengths
            total_length = 0
            content_types = set()

            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        total_length += len(content)
                        content_types.add("text")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    total_length += len(text)
                                    content_types.add("text")
                                elif item.get("type") == "image":
                                    content_types.add("image")

            inputs["total_input_length"] = total_length
            inputs["content_types"] = list(content_types)

        # Extract parameters
        for param in ["max_tokens", "temperature", "top_p", "top_k", "stop_sequences"]:
            if param in kwargs:
                inputs[param] = kwargs[param]

        # Extract streaming flag
        inputs["stream"] = kwargs.get("stream", False)

        return inputs

    def _extract_completions_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from Completions API call (legacy)."""
        inputs = {}

        # Extract prompt
        prompt = kwargs.get("prompt", "")
        if prompt:
            inputs["prompt_length"] = len(prompt)

        # Extract parameters
        for param in [
            "max_tokens_to_sample",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
        ]:
            if param in kwargs:
                inputs[param] = kwargs[param]

        return inputs

    def _sanitize_args(self, kwargs: dict) -> Dict[str, Any]:
        """Create a sanitized version of arguments for debugging/tracing."""
        sanitized = {}

        # Safe parameters to include
        safe_params = {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stream",
            "stop_sequences",
            "max_tokens_to_sample",
        }

        for param in safe_params:
            if param in kwargs:
                sanitized[param] = kwargs[param]

        # Handle messages specially (sanitize content)
        if "messages" in kwargs:
            messages = kwargs["messages"]
            sanitized["messages"] = self._sanitize_messages(messages)

        # Handle prompt specially (truncate if too long)
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, str) and len(prompt) > 1000:
                sanitized["prompt"] = prompt[:1000] + "... [truncated]"
            else:
                sanitized["prompt"] = prompt

        # Include system message if present
        if "system" in kwargs:
            system = kwargs["system"]
            if isinstance(system, str) and len(system) > 500:
                sanitized["system"] = system[:500] + "... [truncated]"
            else:
                sanitized["system"] = system

        return sanitized

    def _sanitize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Sanitize messages for safe logging."""
        sanitized_messages = []

        for message in messages:
            if isinstance(message, dict):
                sanitized_message = {"role": message.get("role", "unknown")}

                content = message.get("content")
                if isinstance(content, str):
                    # Truncate long text content
                    if len(content) > 500:
                        sanitized_message["content"] = content[:500] + "... [truncated]"
                    else:
                        sanitized_message["content"] = content
                elif isinstance(content, list):
                    # Handle multi-modal content
                    sanitized_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                if len(text) > 200:
                                    sanitized_content.append(
                                        {
                                            "type": "text",
                                            "text": text[:200] + "... [truncated]",
                                        }
                                    )
                                else:
                                    sanitized_content.append(item)
                            elif item.get("type") == "image":
                                sanitized_content.append(
                                    {"type": "image", "source": "[image_data_omitted]"}
                                )
                            else:
                                sanitized_content.append(item)
                    sanitized_message["content"] = sanitized_content
                else:
                    sanitized_message["content"] = str(content)

                sanitized_messages.append(sanitized_message)
            else:
                # Handle non-dict messages
                sanitized_messages.append({"role": "unknown", "content": str(message)})

        return sanitized_messages
