"""OpenAI metrics extraction utilities.

This module provides utilities for extracting structured data from OpenAI
API requests and responses, supporting all OpenAI endpoint types.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..base import AlignXProviderCallData

logger = logging.getLogger(__name__)


class OpenAIMetricsExtractor:
    """Extracts structured metrics data from OpenAI API calls.

    This class handles parsing of OpenAI request arguments and responses
    to extract relevant information for observability and cost tracking.
    """

    # Mapping of OpenAI endpoints to operation names
    ENDPOINT_OPERATION_MAP = {
        "/v1/chat/completions": "chat.completion",
        "/v1/completions": "text.completion",
        "/v1/embeddings": "embedding",
        "/v1/images/generations": "image.generation",
        "/v1/images/edits": "image.edit",
        "/v1/images/variations": "image.variation",
        "/v1/audio/transcriptions": "audio.transcription",
        "/v1/audio/translations": "audio.translation",
        "/v1/audio/speech": "audio.speech",
        "/v1/files": "file.operation",
        "/v1/fine-tuning/jobs": "fine_tuning",
        "/v1/moderations": "moderation",
        "/v1/assistants": "assistant",
        "/v1/threads": "thread",
        "/v1/batches": "batch",
    }

    def extract_call_data(
        self, method: str, args: tuple, kwargs: dict
    ) -> AlignXProviderCallData:
        """Extract call data from OpenAI request arguments.

        Args:
            method: Request method name
            args: Positional arguments (typically: cast, options)
            kwargs: Keyword arguments

        Returns:
            AlignXProviderCallData: Extracted call data
        """
        try:
            # Extract request information from arguments
            request_info = self._parse_request_args(args, kwargs)

            # Determine operation type from URL
            operation = self._determine_operation(request_info.get("url", ""))

            # Extract model information
            model = self._extract_model(request_info)

            # Create call data
            call_data = AlignXProviderCallData(
                method=operation,
                model=model,
                inputs=request_info,
                provider_metadata={
                    "endpoint": request_info.get("url"),
                    "method": request_info.get("http_method", "POST"),
                },
            )

            return call_data

        except Exception as e:
            logger.warning(f"Error extracting OpenAI call data: {e}")
            return AlignXProviderCallData(method="unknown", model="unknown", inputs={})

    def _parse_request_args(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Parse OpenAI request arguments to extract request information.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dictionary containing parsed request information
        """
        request_info = {}

        try:
            # OpenAI SDK typically passes arguments in specific patterns
            # args[0] is usually the cast type, args[1] is options
            if len(args) >= 2:
                # Extract options (request details)
                options = args[1]
                if hasattr(options, "__dict__"):
                    options_dict = vars(options)
                elif isinstance(options, dict):
                    options_dict = options
                else:
                    options_dict = {}

                # Extract common fields
                request_info.update(
                    {
                        "url": options_dict.get("url", ""),
                        "http_method": options_dict.get("method", "POST"),
                        "json_data": options_dict.get("json_data", {}),
                        "params": options_dict.get("params", {}),
                        "headers": options_dict.get("headers", {}),
                    }
                )

                # Extract request body from json_data
                if "json_data" in options_dict and options_dict["json_data"]:
                    json_data = options_dict["json_data"]
                    if isinstance(json_data, dict):
                        request_info.update(json_data)

            # Also check kwargs for additional information
            request_info.update(kwargs)

        except Exception as e:
            logger.debug(f"Error parsing request args: {e}")

        return request_info

    def _determine_operation(self, url: str) -> str:
        """Determine operation type from OpenAI API URL.

        Args:
            url: API endpoint URL

        Returns:
            Operation name (e.g., "chat.completion", "embedding")
        """
        # Extract path from URL
        if url:
            # Handle full URLs and path-only URLs
            if url.startswith("http"):
                from urllib.parse import urlparse

                parsed = urlparse(url)
                path = parsed.path
            else:
                path = url

            # Match against known endpoints
            for endpoint, operation in self.ENDPOINT_OPERATION_MAP.items():
                if endpoint in path:
                    return operation

        return "unknown"

    def _extract_model(self, request_info: Dict[str, Any]) -> str:
        """Extract model name from request information.

        Args:
            request_info: Parsed request information

        Returns:
            Model name or "unknown"
        """
        # Try various ways to extract model name
        model_candidates = [
            request_info.get("model"),
            (
                request_info.get("json_data", {}).get("model")
                if isinstance(request_info.get("json_data"), dict)
                else None
            ),
            (
                request_info.get("params", {}).get("model")
                if isinstance(request_info.get("params"), dict)
                else None
            ),
        ]

        for candidate in model_candidates:
            if candidate and isinstance(candidate, str):
                return candidate

        return "unknown"

    def extract_request_metadata(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional metadata from request information.

        Args:
            request_info: Parsed request information

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        try:
            # Extract common request parameters
            if "temperature" in request_info:
                metadata["temperature"] = request_info["temperature"]

            if "max_tokens" in request_info:
                metadata["max_tokens"] = request_info["max_tokens"]

            if "top_p" in request_info:
                metadata["top_p"] = request_info["top_p"]

            if "frequency_penalty" in request_info:
                metadata["frequency_penalty"] = request_info["frequency_penalty"]

            if "presence_penalty" in request_info:
                metadata["presence_penalty"] = request_info["presence_penalty"]

            if "stream" in request_info:
                metadata["stream"] = request_info["stream"]

            # Extract message count for chat completions
            if "messages" in request_info and isinstance(
                request_info["messages"], list
            ):
                metadata["message_count"] = len(request_info["messages"])

                # Estimate input tokens from messages (rough approximation)
                total_chars = sum(
                    len(str(msg.get("content", "")))
                    for msg in request_info["messages"]
                    if isinstance(msg, dict)
                )
                # Rough estimate: 1 token ≈ 4 characters
                metadata["estimated_input_tokens"] = total_chars // 4

            # Extract input text for other endpoints
            if "prompt" in request_info:
                prompt = request_info["prompt"]
                if isinstance(prompt, str):
                    metadata["prompt_length"] = len(prompt)
                    metadata["estimated_input_tokens"] = len(prompt) // 4
                elif isinstance(prompt, list):
                    total_length = sum(len(str(p)) for p in prompt)
                    metadata["prompt_length"] = total_length
                    metadata["estimated_input_tokens"] = total_length // 4

            # Extract function calling information
            if "functions" in request_info:
                metadata["function_count"] = len(request_info["functions"])
                metadata["function_calling_enabled"] = True

            if "tools" in request_info:
                metadata["tool_count"] = len(request_info["tools"])
                metadata["tool_calling_enabled"] = True

        except Exception as e:
            logger.debug(f"Error extracting request metadata: {e}")

        return metadata

    def extract_base_url_info(self, request_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract base URL information to detect custom providers.

        Args:
            request_info: Parsed request information

        Returns:
            Dictionary with base URL information
        """
        base_url_info = {
            "provider": "openai",
            "is_azure": False,
            "is_custom": False,
        }

        try:
            url = request_info.get("url", "")

            if url:
                if "azure" in url.lower():
                    base_url_info["provider"] = "azure_openai"
                    base_url_info["is_azure"] = True
                elif "api.openai.com" not in url:
                    base_url_info["is_custom"] = True
                    # Try to extract provider name from URL
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    if parsed.hostname:
                        base_url_info["provider"] = parsed.hostname.replace(
                            "api.", ""
                        ).replace(".com", "")

        except Exception as e:
            logger.debug(f"Error extracting base URL info: {e}")

        return base_url_info

    def sanitize_request_data(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data for logging and tracing.

        Args:
            request_info: Raw request information

        Returns:
            Sanitized request data safe for logging
        """
        sanitized = {}

        # Fields to include (non-sensitive)
        safe_fields = {
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stream",
            "n",
            "stop",
            "echo",
            "logprobs",
            "response_format",
            "tool_choice",
            "parallel_tool_calls",
        }

        try:
            for field in safe_fields:
                if field in request_info:
                    sanitized[field] = request_info[field]

            # Include message count but not content
            if "messages" in request_info and isinstance(
                request_info["messages"], list
            ):
                sanitized["message_count"] = len(request_info["messages"])

                # Include role distribution
                role_counts = {}
                for msg in request_info["messages"]:
                    if isinstance(msg, dict) and "role" in msg:
                        role = msg["role"]
                        role_counts[role] = role_counts.get(role, 0) + 1
                sanitized["role_distribution"] = role_counts

            # Include prompt length but not content
            if "prompt" in request_info:
                prompt = request_info["prompt"]
                if isinstance(prompt, str):
                    sanitized["prompt_length"] = len(prompt)
                elif isinstance(prompt, list):
                    sanitized["prompt_count"] = len(prompt)
                    sanitized["total_prompt_length"] = sum(len(str(p)) for p in prompt)

            # Include function/tool information without sensitive details
            if "functions" in request_info:
                sanitized["function_count"] = len(request_info["functions"])

            if "tools" in request_info:
                sanitized["tool_count"] = len(request_info["tools"])

        except Exception as e:
            logger.debug(f"Error sanitizing request data: {e}")

        return sanitized
