"""OpenAI response parsing utilities.

This module provides utilities for parsing OpenAI API responses to extract
token usage, costs, and other relevant metrics for observability.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..base import AlignXProviderCallData

logger = logging.getLogger(__name__)


class OpenAIResponseParser:
    """Parses OpenAI API responses to extract metrics and metadata.

    This class handles parsing of various OpenAI response types including
    chat completions, text completions, embeddings, and streaming responses.
    """

    def parse_response(self, call_data: AlignXProviderCallData, response: Any) -> None:
        """Parse OpenAI response and update call data.

        Args:
            call_data: Call data to update with response information
            response: OpenAI API response object
        """
        try:
            # Convert response to dictionary if needed
            response_dict = self._response_to_dict(response)

            if not response_dict:
                logger.debug("Could not convert response to dictionary")
                return

            # Extract token usage information
            usage_info = self._extract_usage_info(response_dict)

            # Extract response metadata
            metadata = self._extract_response_metadata(response_dict, call_data.method)

            # Update call data
            call_data.outputs = self._sanitize_response_outputs(response_dict)
            call_data.response_metadata = {**usage_info, **metadata}

        except Exception as e:
            logger.warning(f"Error parsing OpenAI response: {e}")

    def _response_to_dict(self, response: Any) -> Optional[Dict[str, Any]]:
        """Convert OpenAI response object to dictionary.

        Args:
            response: OpenAI response object

        Returns:
            Dictionary representation of response or None
        """
        try:
            # Handle different response types
            if isinstance(response, dict):
                return response

            # Handle OpenAI response objects with model_dump or dict methods
            if hasattr(response, "model_dump"):
                return response.model_dump()
            elif hasattr(response, "to_dict"):
                return response.to_dict()
            elif hasattr(response, "__dict__"):
                return vars(response)

            # Try JSON serialization as last resort
            if hasattr(response, "__str__"):
                try:
                    return json.loads(str(response))
                except json.JSONDecodeError:
                    pass

            return None

        except Exception as e:
            logger.debug(f"Error converting response to dict: {e}")
            return None

    def _extract_usage_info(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token usage information from response.

        Args:
            response_dict: Response dictionary

        Returns:
            Dictionary with usage information
        """
        usage_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        try:
            # Standard OpenAI usage field
            if "usage" in response_dict:
                usage = response_dict["usage"]

                # Handle different usage field names
                usage_info["input_tokens"] = usage.get("prompt_tokens", 0)
                usage_info["output_tokens"] = usage.get("completion_tokens", 0)
                usage_info["total_tokens"] = usage.get("total_tokens", 0)

                # Calculate total if not provided
                if usage_info["total_tokens"] == 0:
                    usage_info["total_tokens"] = (
                        usage_info["input_tokens"] + usage_info["output_tokens"]
                    )

            # Alternative usage extraction for different response formats
            elif "prompt_tokens" in response_dict:
                usage_info["input_tokens"] = response_dict.get("prompt_tokens", 0)
                usage_info["output_tokens"] = response_dict.get("completion_tokens", 0)
                usage_info["total_tokens"] = response_dict.get("total_tokens", 0)

            # For embeddings, only input tokens are relevant
            elif "data" in response_dict and isinstance(response_dict["data"], list):
                # Estimate tokens for embeddings (rough approximation)
                usage_info["input_tokens"] = response_dict.get("usage", {}).get(
                    "prompt_tokens", 0
                )
                usage_info["output_tokens"] = 0
                usage_info["total_tokens"] = usage_info["input_tokens"]

        except Exception as e:
            logger.debug(f"Error extracting usage info: {e}")

        return usage_info

    def _extract_response_metadata(
        self, response_dict: Dict[str, Any], operation: str
    ) -> Dict[str, Any]:
        """Extract additional metadata from response.

        Args:
            response_dict: Response dictionary
            operation: Operation type (e.g., "chat.completion")

        Returns:
            Dictionary with response metadata
        """
        metadata = {}

        try:
            # Common response fields
            if "id" in response_dict:
                metadata["response_id"] = response_dict["id"]

            if "created" in response_dict:
                metadata["created_timestamp"] = response_dict["created"]

            if "model" in response_dict:
                metadata["response_model"] = response_dict["model"]

            if "object" in response_dict:
                metadata["object_type"] = response_dict["object"]

            # Operation-specific metadata extraction
            if operation == "chat.completion":
                metadata.update(self._extract_chat_completion_metadata(response_dict))
            elif operation == "text.completion":
                metadata.update(self._extract_text_completion_metadata(response_dict))
            elif operation == "embedding":
                metadata.update(self._extract_embedding_metadata(response_dict))
            elif operation.startswith("image."):
                metadata.update(self._extract_image_metadata(response_dict))
            elif operation.startswith("audio."):
                metadata.update(self._extract_audio_metadata(response_dict))

        except Exception as e:
            logger.debug(f"Error extracting response metadata: {e}")

        return metadata

    def _extract_chat_completion_metadata(
        self, response_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata specific to chat completion responses.

        Args:
            response_dict: Response dictionary

        Returns:
            Chat completion specific metadata
        """
        metadata = {}

        try:
            if "choices" in response_dict and isinstance(
                response_dict["choices"], list
            ):
                choices = response_dict["choices"]
                metadata["choice_count"] = len(choices)

                if choices:
                    first_choice = choices[0]

                    # Finish reason
                    if "finish_reason" in first_choice:
                        metadata["finish_reason"] = first_choice["finish_reason"]

                    # Message information
                    if "message" in first_choice:
                        message = first_choice["message"]
                        if "role" in message:
                            metadata["response_role"] = message["role"]

                        # Function/tool calling information
                        if "function_call" in message:
                            metadata["function_call_used"] = True
                            metadata["function_name"] = message["function_call"].get(
                                "name"
                            )

                        if "tool_calls" in message and message["tool_calls"]:
                            metadata["tool_calls_used"] = True
                            metadata["tool_call_count"] = len(message["tool_calls"])

                            # Extract tool types
                            tool_types = set()
                            for tool_call in message["tool_calls"]:
                                if "type" in tool_call:
                                    tool_types.add(tool_call["type"])
                            metadata["tool_types"] = list(tool_types)

                        # Content length estimation
                        if "content" in message and message["content"]:
                            content = message["content"]
                            if isinstance(content, str):
                                metadata["response_content_length"] = len(content)
                                metadata["estimated_output_tokens"] = len(content) // 4

        except Exception as e:
            logger.debug(f"Error extracting chat completion metadata: {e}")

        return metadata

    def _extract_text_completion_metadata(
        self, response_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata specific to text completion responses.

        Args:
            response_dict: Response dictionary

        Returns:
            Text completion specific metadata
        """
        metadata = {}

        try:
            if "choices" in response_dict and isinstance(
                response_dict["choices"], list
            ):
                choices = response_dict["choices"]
                metadata["choice_count"] = len(choices)

                if choices:
                    first_choice = choices[0]

                    if "finish_reason" in first_choice:
                        metadata["finish_reason"] = first_choice["finish_reason"]

                    if "text" in first_choice:
                        text = first_choice["text"]
                        metadata["response_text_length"] = len(text)
                        metadata["estimated_output_tokens"] = len(text) // 4

                    if "logprobs" in first_choice:
                        metadata["logprobs_available"] = True

        except Exception as e:
            logger.debug(f"Error extracting text completion metadata: {e}")

        return metadata

    def _extract_embedding_metadata(
        self, response_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata specific to embedding responses.

        Args:
            response_dict: Response dictionary

        Returns:
            Embedding specific metadata
        """
        metadata = {}

        try:
            if "data" in response_dict and isinstance(response_dict["data"], list):
                data = response_dict["data"]
                metadata["embedding_count"] = len(data)

                if data and isinstance(data[0], dict) and "embedding" in data[0]:
                    # Extract embedding dimensions
                    embedding = data[0]["embedding"]
                    if isinstance(embedding, list):
                        metadata["embedding_dimensions"] = len(embedding)

        except Exception as e:
            logger.debug(f"Error extracting embedding metadata: {e}")

        return metadata

    def _extract_image_metadata(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata specific to image generation responses.

        Args:
            response_dict: Response dictionary

        Returns:
            Image generation specific metadata
        """
        metadata = {}

        try:
            if "data" in response_dict and isinstance(response_dict["data"], list):
                data = response_dict["data"]
                metadata["image_count"] = len(data)

                # Extract image information
                if data:
                    first_image = data[0]
                    if "url" in first_image:
                        metadata["has_image_urls"] = True
                    if "b64_json" in first_image:
                        metadata["has_base64_images"] = True

        except Exception as e:
            logger.debug(f"Error extracting image metadata: {e}")

        return metadata

    def _extract_audio_metadata(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata specific to audio responses.

        Args:
            response_dict: Response dictionary

        Returns:
            Audio specific metadata
        """
        metadata = {}

        try:
            # For transcription/translation responses
            if "text" in response_dict:
                text = response_dict["text"]
                metadata["transcript_length"] = len(text)
                metadata["estimated_transcript_tokens"] = len(text) // 4

            # For speech responses (binary data)
            if isinstance(response_dict, bytes):
                metadata["audio_data_size"] = len(response_dict)

        except Exception as e:
            logger.debug(f"Error extracting audio metadata: {e}")

        return metadata

    def _sanitize_response_outputs(
        self, response_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize response outputs for safe logging and storage.

        Args:
            response_dict: Raw response dictionary

        Returns:
            Sanitized response data
        """
        sanitized = {}

        try:
            # Include safe metadata fields
            safe_fields = {
                "id",
                "object",
                "created",
                "model",
                "finish_reason",
                "choice_count",
                "embedding_count",
                "image_count",
            }

            for field in safe_fields:
                if field in response_dict:
                    sanitized[field] = response_dict[field]

            # Include usage information (safe)
            if "usage" in response_dict:
                sanitized["usage"] = response_dict["usage"]

            # Include choices metadata without content
            if "choices" in response_dict and isinstance(
                response_dict["choices"], list
            ):
                choices_metadata = []
                for choice in response_dict["choices"]:
                    choice_meta = {}
                    if "index" in choice:
                        choice_meta["index"] = choice["index"]
                    if "finish_reason" in choice:
                        choice_meta["finish_reason"] = choice["finish_reason"]

                    # Include tool/function call information
                    if "message" in choice:
                        message = choice["message"]
                        if "function_call" in message:
                            choice_meta["has_function_call"] = True
                        if "tool_calls" in message:
                            choice_meta["tool_call_count"] = len(message["tool_calls"])

                    choices_metadata.append(choice_meta)

                sanitized["choices_metadata"] = choices_metadata

            # Include data array metadata without actual content
            if "data" in response_dict and isinstance(response_dict["data"], list):
                sanitized["data_count"] = len(response_dict["data"])

        except Exception as e:
            logger.debug(f"Error sanitizing response outputs: {e}")

        return sanitized

    def combine_streaming_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        """Combine streaming response chunks into a final response.

        Args:
            chunks: List of streaming response chunks

        Returns:
            Combined response dictionary
        """
        combined_response = {
            "choices": [{"message": {"content": "", "role": "assistant"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "unknown",
        }

        try:
            content_parts = []
            completion_tokens = 0

            for chunk in chunks:
                chunk_dict = self._response_to_dict(chunk)
                if not chunk_dict:
                    continue

                # Extract model information from first chunk
                if combined_response["model"] == "unknown" and "model" in chunk_dict:
                    combined_response["model"] = chunk_dict["model"]

                # Process choices for content
                if "choices" in chunk_dict and isinstance(chunk_dict["choices"], list):
                    for choice in chunk_dict["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                content_parts.append(content)
                                # Rough token estimation
                                completion_tokens += len(content) // 4

                        # Capture finish reason from final chunk
                        if "finish_reason" in choice and choice["finish_reason"]:
                            combined_response["finish_reason"] = choice["finish_reason"]

                # Extract usage from final chunk
                if "usage" in chunk_dict:
                    combined_response["usage"] = chunk_dict["usage"]

            # Combine content
            combined_content = "".join(content_parts)
            combined_response["choices"][0]["message"]["content"] = combined_content

            # Update token estimates if no usage provided
            if combined_response["usage"]["completion_tokens"] == 0:
                combined_response["usage"]["completion_tokens"] = completion_tokens
                combined_response["usage"]["total_tokens"] = completion_tokens

        except Exception as e:
            logger.warning(f"Error combining streaming chunks: {e}")

        return combined_response
