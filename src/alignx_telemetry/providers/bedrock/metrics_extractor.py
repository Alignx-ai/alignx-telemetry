"""AlignX Bedrock Metrics Extractor

Extracts structured data and metadata from AWS Bedrock API requests for tracing and metrics.
Supports multiple foundation models available through Bedrock.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..base import AlignXProviderCallData

logger = logging.getLogger(__name__)


class BedrockMetricsExtractor:
    """Extracts metrics and metadata from Bedrock API calls."""

    def extract_call_data(
        self, args: tuple, kwargs: dict, operation: str = "unknown"
    ) -> AlignXProviderCallData:
        """Extract structured call data from Bedrock API arguments."""

        # Determine the model being used
        model = self._extract_model(kwargs)

        # Extract operation-specific inputs
        inputs = self._extract_inputs(kwargs, operation)

        # Extract sanitized arguments for debugging
        sanitized_args = self._sanitize_args(kwargs)

        return AlignXProviderCallData(
            provider="bedrock",
            model=model,
            operation=operation,
            inputs=inputs,
            args=sanitized_args,
        )

    def _extract_model(self, kwargs: dict) -> str:
        """Extract the model ID from API call arguments."""
        model_id = kwargs.get("modelId") or kwargs.get("ModelId", "unknown")
        return self._normalize_model_id(model_id)

    def _normalize_model_id(self, model_id: str) -> str:
        """Normalize model ID for consistency."""
        if not model_id:
            return "unknown"

        # Common model ID mappings for consistency
        model_mapping = {
            # Anthropic Claude models
            "anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3.5-sonnet",
            "anthropic.claude-3-5-sonnet-20240620-v1:0": "claude-3.5-sonnet",
            "anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3.5-haiku",
            "anthropic.claude-3-opus-20240229-v1:0": "claude-3-opus",
            "anthropic.claude-3-sonnet-20240229-v1:0": "claude-3-sonnet",
            "anthropic.claude-3-haiku-20240307-v1:0": "claude-3-haiku",
            "anthropic.claude-v2:1": "claude-2.1",
            "anthropic.claude-v2": "claude-2.0",
            "anthropic.claude-instant-v1": "claude-instant-1.2",
            # Amazon Titan models
            "amazon.titan-text-lite-v1": "titan-text-lite",
            "amazon.titan-text-express-v1": "titan-text-express",
            "amazon.titan-text-premier-v1:0": "titan-text-premier",
            "amazon.titan-embed-text-v1": "titan-embed-text",
            "amazon.titan-embed-text-v2:0": "titan-embed-text-v2",
            "amazon.titan-embed-image-v1": "titan-embed-image",
            "amazon.titan-image-generator-v1": "titan-image-generator",
            # Cohere Command models
            "cohere.command-text-v14": "command-text",
            "cohere.command-light-text-v14": "command-light-text",
            "cohere.command-r-v1:0": "command-r",
            "cohere.command-r-plus-v1:0": "command-r-plus",
            "cohere.embed-english-v3": "embed-english",
            "cohere.embed-multilingual-v3": "embed-multilingual",
            # Meta Llama models
            "meta.llama2-13b-chat-v1": "llama2-13b-chat",
            "meta.llama2-70b-chat-v1": "llama2-70b-chat",
            "meta.llama3-8b-instruct-v1:0": "llama3-8b-instruct",
            "meta.llama3-70b-instruct-v1:0": "llama3-70b-instruct",
            "meta.llama3-1-8b-instruct-v1:0": "llama3.1-8b-instruct",
            "meta.llama3-1-70b-instruct-v1:0": "llama3.1-70b-instruct",
            "meta.llama3-1-405b-instruct-v1:0": "llama3.1-405b-instruct",
            # Mistral AI models
            "mistral.mistral-7b-instruct-v0:2": "mistral-7b-instruct",
            "mistral.mixtral-8x7b-instruct-v0:1": "mixtral-8x7b-instruct",
            "mistral.mistral-large-2402-v1:0": "mistral-large",
            "mistral.mistral-small-2402-v1:0": "mistral-small",
        }

        return model_mapping.get(model_id, model_id)

    def _extract_inputs(self, kwargs: dict, operation: str) -> Dict[str, Any]:
        """Extract input data from API call arguments."""

        if operation in ["invoke_model", "invoke_model_with_response_stream"]:
            return self._extract_invoke_model_inputs(kwargs)
        elif operation in ["converse", "converse_stream"]:
            return self._extract_converse_inputs(kwargs)
        else:
            return {}

    def _extract_invoke_model_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from invoke_model API call."""
        inputs = {}

        # Extract body (request payload)
        body = kwargs.get("body")
        if body:
            if isinstance(body, str):
                try:
                    body_data = json.loads(body)
                except json.JSONDecodeError:
                    body_data = {}
            elif isinstance(body, (bytes, bytearray)):
                try:
                    body_data = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body_data = {}
            else:
                body_data = body if isinstance(body, dict) else {}

            # Extract common parameters across different model formats
            self._extract_model_specific_inputs(body_data, inputs)

        # Extract other parameters
        for param in ["accept", "contentType"]:
            if param in kwargs:
                inputs[param] = kwargs[param]

        return inputs

    def _extract_converse_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from converse API call."""
        inputs = {}

        # Extract messages
        messages = kwargs.get("messages", [])
        if messages:
            inputs["message_count"] = len(messages)

            # Calculate total input length and content types
            total_length = 0
            content_types = set()

            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if "text" in item:
                                    text = item["text"]
                                    total_length += len(text)
                                    content_types.add("text")
                                elif "image" in item:
                                    content_types.add("image")
                                elif "document" in item:
                                    content_types.add("document")
                                elif "toolUse" in item:
                                    content_types.add("tool_use")
                                elif "toolResult" in item:
                                    content_types.add("tool_result")

            inputs["total_input_length"] = total_length
            inputs["content_types"] = list(content_types)

        # Extract system message
        system = kwargs.get("system")
        if system:
            if isinstance(system, list):
                inputs["system_message_count"] = len(system)
            else:
                inputs["system_message"] = True

        # Extract inference configuration
        inference_config = kwargs.get("inferenceConfig", {})
        for param in ["maxTokens", "temperature", "topP", "stopSequences"]:
            if param in inference_config:
                inputs[param] = inference_config[param]

        # Extract tool configuration
        tool_config = kwargs.get("toolConfig", {})
        if "tools" in tool_config:
            inputs["tool_count"] = len(tool_config["tools"])

        return inputs

    def _extract_model_specific_inputs(self, body_data: dict, inputs: dict):
        """Extract inputs specific to different model formats."""

        # Anthropic Claude format
        if "messages" in body_data:
            messages = body_data["messages"]
            inputs["message_count"] = len(messages)

            total_length = 0
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        total_length += len(content)
            inputs["total_input_length"] = total_length

            # Extract Claude-specific parameters
            for param in [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stop_sequences",
            ]:
                if param in body_data:
                    inputs[param] = body_data[param]

        # Amazon Titan format
        elif "inputText" in body_data:
            input_text = body_data["inputText"]
            inputs["input_length"] = len(input_text)

            # Extract Titan-specific parameters
            text_config = body_data.get("textGenerationConfig", {})
            for param in ["maxTokenCount", "temperature", "topP", "stopSequences"]:
                if param in text_config:
                    inputs[param] = text_config[param]

        # Cohere Command format
        elif "prompt" in body_data:
            prompt = body_data["prompt"]
            inputs["prompt_length"] = len(prompt)

            # Extract Cohere-specific parameters
            for param in ["max_tokens", "temperature", "p", "k", "stop_sequences"]:
                if param in body_data:
                    inputs[param] = body_data[param]

        # Meta Llama format
        elif "prompt" in body_data or "inputs" in body_data:
            prompt = body_data.get("prompt") or body_data.get("inputs", "")
            inputs["prompt_length"] = len(prompt)

            # Extract Llama-specific parameters
            parameters = body_data.get("parameters", {})
            for param in ["max_gen_len", "temperature", "top_p", "top_k"]:
                if param in parameters:
                    inputs[param] = parameters[param]

    def _sanitize_args(self, kwargs: dict) -> Dict[str, Any]:
        """Create a sanitized version of arguments for debugging/tracing."""
        sanitized = {}

        # Safe parameters to include
        safe_params = {"modelId", "accept", "contentType"}

        for param in safe_params:
            if param in kwargs:
                sanitized[param] = kwargs[param]

        # Handle body specially (sanitize content)
        body = kwargs.get("body")
        if body:
            sanitized["body"] = self._sanitize_body(body)

        # Handle converse parameters
        messages = kwargs.get("messages")
        if messages:
            sanitized["messages"] = self._sanitize_messages(messages)

        system = kwargs.get("system")
        if system:
            if isinstance(system, str) and len(system) > 500:
                sanitized["system"] = system[:500] + "... [truncated]"
            else:
                sanitized["system"] = system

        # Include inference config safely
        inference_config = kwargs.get("inferenceConfig")
        if inference_config:
            sanitized["inferenceConfig"] = {
                k: v
                for k, v in inference_config.items()
                if k in ["maxTokens", "temperature", "topP", "stopSequences"]
            }

        return sanitized

    def _sanitize_body(self, body: Any) -> Any:
        """Sanitize body content for safe logging."""
        if isinstance(body, str):
            try:
                body_data = json.loads(body)
                return json.dumps(self._sanitize_body_data(body_data))
            except json.JSONDecodeError:
                return "[invalid_json]"
        elif isinstance(body, (bytes, bytearray)):
            try:
                body_data = json.loads(body.decode("utf-8"))
                return json.dumps(self._sanitize_body_data(body_data))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return "[invalid_json_bytes]"
        elif isinstance(body, dict):
            return self._sanitize_body_data(body)
        else:
            return str(body)

    def _sanitize_body_data(self, body_data: dict) -> dict:
        """Sanitize body data dictionary."""
        sanitized = {}

        # Safe fields to include
        safe_fields = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "maxTokenCount",
            "topP",
            "stopSequences",
            "max_gen_len",
            "parameters",
        }

        for field in safe_fields:
            if field in body_data:
                sanitized[field] = body_data[field]

        # Handle content fields specially
        if "messages" in body_data:
            sanitized["messages"] = self._sanitize_messages(body_data["messages"])
        elif "inputText" in body_data:
            text = body_data["inputText"]
            if len(text) > 500:
                sanitized["inputText"] = text[:500] + "... [truncated]"
            else:
                sanitized["inputText"] = text
        elif "prompt" in body_data:
            prompt = body_data["prompt"]
            if len(prompt) > 500:
                sanitized["prompt"] = prompt[:500] + "... [truncated]"
            else:
                sanitized["prompt"] = prompt

        return sanitized

    def _sanitize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Sanitize messages for safe logging."""
        sanitized_messages = []

        for message in messages:
            if isinstance(message, dict):
                sanitized_message = {"role": message.get("role", "unknown")}

                content = message.get("content")
                if isinstance(content, str):
                    if len(content) > 200:
                        sanitized_message["content"] = content[:200] + "... [truncated]"
                    else:
                        sanitized_message["content"] = content
                elif isinstance(content, list):
                    sanitized_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text = item["text"]
                                if len(text) > 200:
                                    sanitized_content.append(
                                        {"text": text[:200] + "... [truncated]"}
                                    )
                                else:
                                    sanitized_content.append(item)
                            elif "image" in item:
                                sanitized_content.append(
                                    {"type": "image", "data": "[image_data_omitted]"}
                                )
                            elif "document" in item:
                                sanitized_content.append(
                                    {
                                        "type": "document",
                                        "data": "[document_data_omitted]",
                                    }
                                )
                            else:
                                sanitized_content.append(item)
                    sanitized_message["content"] = sanitized_content

                sanitized_messages.append(sanitized_message)

        return sanitized_messages
