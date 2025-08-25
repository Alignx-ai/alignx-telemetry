"""AlignX Anthropic Response Parser

Parses Anthropic API responses to extract usage data, metadata, and handles streaming responses.
"""

import logging
from typing import Any, Dict, List, Optional, Union

try:
    from anthropic.types import Message, Usage, TextBlock
    from anthropic._streaming import Stream, AsyncStream

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Message = None
    Usage = None
    TextBlock = None
    Stream = None
    AsyncStream = None

logger = logging.getLogger(__name__)


class AnthropicResponseParser:
    """Parses Anthropic API responses to extract usage and metadata."""

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract structured data from Anthropic API response."""

        if not response:
            return {}

        # Handle different response types
        if ANTHROPIC_AVAILABLE and isinstance(response, Message):
            return self._parse_message_response(response)
        elif hasattr(response, "content") and hasattr(response, "usage"):
            # Duck-typed Message response
            return self._parse_message_response(response)
        elif isinstance(response, dict):
            return self._parse_dict_response(response)
        else:
            # Fallback for unknown response types
            return self._parse_generic_response(response)

    def _parse_message_response(self, response) -> Dict[str, Any]:
        """Parse a Message response from the Messages API."""
        data = {}

        # Extract usage information
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            data["input_tokens"] = getattr(usage, "input_tokens", 0)
            data["output_tokens"] = getattr(usage, "output_tokens", 0)
            data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        # Extract content
        if hasattr(response, "content") and response.content:
            content_data = self._extract_content_data(response.content)
            data.update(content_data)

        # Extract metadata
        data["response_id"] = getattr(response, "id", None)
        data["model"] = getattr(response, "model", None)
        data["role"] = getattr(response, "role", None)
        data["stop_reason"] = getattr(response, "stop_reason", None)
        data["stop_sequence"] = getattr(response, "stop_sequence", None)

        return data

    def _parse_dict_response(self, response: dict) -> Dict[str, Any]:
        """Parse a dictionary response."""
        data = {}

        # Extract usage information
        if "usage" in response:
            usage = response["usage"]
            if isinstance(usage, dict):
                data["input_tokens"] = usage.get("input_tokens", 0)
                data["output_tokens"] = usage.get("output_tokens", 0)
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        # Extract content
        if "content" in response:
            content_data = self._extract_content_data(response["content"])
            data.update(content_data)

        # Extract metadata
        data["response_id"] = response.get("id")
        data["model"] = response.get("model")
        data["role"] = response.get("role")
        data["stop_reason"] = response.get("stop_reason")
        data["stop_sequence"] = response.get("stop_sequence")

        return data

    def _parse_generic_response(self, response: Any) -> Dict[str, Any]:
        """Parse a generic response object."""
        data = {}

        # Try to extract common attributes
        for attr in ["id", "model", "role", "stop_reason", "stop_sequence"]:
            if hasattr(response, attr):
                data[attr] = getattr(response, attr)

        # Try to extract usage
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                data["input_tokens"] = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                data["output_tokens"] = usage.output_tokens
            if "input_tokens" in data and "output_tokens" in data:
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        # Try to extract content
        if hasattr(response, "content"):
            content_data = self._extract_content_data(response.content)
            data.update(content_data)

        return data

    def _extract_content_data(self, content: Any) -> Dict[str, Any]:
        """Extract data from content field."""
        data = {}

        if not content:
            return data

        if isinstance(content, list):
            # Handle list of content blocks
            text_blocks = []
            content_types = set()

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    content_types.add(block_type)

                    if block_type == "text":
                        text = block.get("text", "")
                        text_blocks.append(text)
                elif ANTHROPIC_AVAILABLE and isinstance(block, TextBlock):
                    content_types.add("text")
                    text_blocks.append(block.text)
                elif hasattr(block, "text"):
                    content_types.add("text")
                    text_blocks.append(block.text)

            data["content_types"] = list(content_types)
            data["text_content"] = text_blocks
            data["output_length"] = sum(len(text) for text in text_blocks)

        elif isinstance(content, str):
            # Handle simple string content
            data["text_content"] = [content]
            data["output_length"] = len(content)
            data["content_types"] = ["text"]

        return data

    def combine_streaming_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        """Combine streaming response chunks into a single response for analysis."""

        if not chunks:
            return {}

        combined_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "text_content": [],
            "content_types": set(),
            "stop_reason": None,
            "stop_sequence": None,
            "model": None,
            "response_id": None,
        }

        accumulated_text = ""

        for chunk in chunks:
            chunk_data = self._parse_streaming_chunk(chunk)

            # Accumulate usage
            if "input_tokens" in chunk_data:
                combined_data["input_tokens"] = max(
                    combined_data["input_tokens"], chunk_data["input_tokens"]
                )
            if "output_tokens" in chunk_data:
                combined_data["output_tokens"] = max(
                    combined_data["output_tokens"], chunk_data["output_tokens"]
                )

            # Accumulate text content
            if "text_delta" in chunk_data:
                accumulated_text += chunk_data["text_delta"]
                combined_data["content_types"].add("text")

            # Update metadata (last chunk wins)
            for field in ["stop_reason", "stop_sequence", "model", "response_id"]:
                if chunk_data.get(field) is not None:
                    combined_data[field] = chunk_data[field]

        # Finalize combined data
        if accumulated_text:
            combined_data["text_content"] = [accumulated_text]
            combined_data["output_length"] = len(accumulated_text)

        combined_data["content_types"] = list(combined_data["content_types"])
        combined_data["total_tokens"] = (
            combined_data["input_tokens"] + combined_data["output_tokens"]
        )

        return combined_data

    def _parse_streaming_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Parse a single streaming response chunk."""
        data = {}

        # Handle different chunk formats
        if isinstance(chunk, dict):
            # Handle dictionary chunks
            if "delta" in chunk:
                delta = chunk["delta"]
                if isinstance(delta, dict) and "text" in delta:
                    data["text_delta"] = delta["text"]

            if "usage" in chunk:
                usage = chunk["usage"]
                if isinstance(usage, dict):
                    data["input_tokens"] = usage.get("input_tokens", 0)
                    data["output_tokens"] = usage.get("output_tokens", 0)

            # Extract metadata
            for field in ["stop_reason", "stop_sequence", "model", "id"]:
                if field in chunk:
                    if field == "id":
                        data["response_id"] = chunk[field]
                    else:
                        data[field] = chunk[field]

        elif hasattr(chunk, "delta") or hasattr(chunk, "usage"):
            # Handle object chunks with delta/usage attributes
            if hasattr(chunk, "delta"):
                delta = chunk.delta
                if hasattr(delta, "text"):
                    data["text_delta"] = delta.text

            if hasattr(chunk, "usage"):
                usage = chunk.usage
                if hasattr(usage, "input_tokens"):
                    data["input_tokens"] = usage.input_tokens
                if hasattr(usage, "output_tokens"):
                    data["output_tokens"] = usage.output_tokens

            # Extract metadata
            for field in ["stop_reason", "stop_sequence", "model", "id"]:
                if hasattr(chunk, field):
                    if field == "id":
                        data["response_id"] = getattr(chunk, field)
                    else:
                        data[field] = getattr(chunk, field)

        return data
