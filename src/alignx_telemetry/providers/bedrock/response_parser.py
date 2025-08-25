"""AlignX Bedrock Response Parser

Parses AWS Bedrock API responses to extract usage data, metadata, and handles streaming responses.
Supports multiple foundation models available through Bedrock.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BedrockResponseParser:
    """Parses Bedrock API responses to extract usage and metadata."""

    def extract_response_data(
        self, response: Any, operation: str = "unknown"
    ) -> Dict[str, Any]:
        """Extract structured data from Bedrock API response."""

        if not response:
            return {}

        # Handle different response formats based on operation
        if operation in ["invoke_model", "invoke_model_with_response_stream"]:
            return self._parse_invoke_model_response(response)
        elif operation in ["converse", "converse_stream"]:
            return self._parse_converse_response(response)
        else:
            return self._parse_generic_response(response)

    def _parse_invoke_model_response(self, response: Any) -> Dict[str, Any]:
        """Parse invoke_model response."""
        data = {}

        if isinstance(response, dict):
            # Extract response body
            body = response.get("body")
            if body:
                body_data = self._extract_body_data(body)
                model_data = self._parse_model_specific_response(body_data)
                data.update(model_data)

            # Extract response metadata
            response_metadata = response.get("ResponseMetadata", {})
            if response_metadata:
                data["response_metadata"] = {
                    "request_id": response_metadata.get("RequestId"),
                    "http_status_code": response_metadata.get("HTTPStatusCode"),
                }

        return data

    def _parse_converse_response(self, response: Any) -> Dict[str, Any]:
        """Parse converse response."""
        data = {}

        if isinstance(response, dict):
            # Extract usage metadata
            usage = response.get("usage", {})
            if usage:
                data["input_tokens"] = usage.get("inputTokens", 0)
                data["output_tokens"] = usage.get("outputTokens", 0)
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

            # Extract output message
            output = response.get("output", {})
            if output and "message" in output:
                message_data = self._extract_converse_message_data(output["message"])
                data.update(message_data)

            # Extract stop reason
            data["stop_reason"] = response.get("stopReason")

            # Extract metrics
            metrics = response.get("metrics", {})
            if metrics:
                data["latency_ms"] = metrics.get("latencyMs")

            # Extract response metadata
            response_metadata = response.get("ResponseMetadata", {})
            if response_metadata:
                data["response_metadata"] = {
                    "request_id": response_metadata.get("RequestId"),
                    "http_status_code": response_metadata.get("HTTPStatusCode"),
                }

        return data

    def _parse_generic_response(self, response: Any) -> Dict[str, Any]:
        """Parse generic response format."""
        data = {}

        if isinstance(response, dict):
            # Try to extract common fields
            for field in [
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "stop_reason",
            ]:
                if field in response:
                    data[field] = response[field]

            # Extract body if present
            if "body" in response:
                body_data = self._extract_body_data(response["body"])
                model_data = self._parse_model_specific_response(body_data)
                data.update(model_data)

        return data

    def _extract_body_data(self, body: Any) -> Dict[str, Any]:
        """Extract data from response body."""
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        elif isinstance(body, (bytes, bytearray)):
            try:
                return json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
        elif hasattr(body, "read"):
            # Handle streaming body
            try:
                content = body.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                return json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                return {}
        elif isinstance(body, dict):
            return body
        else:
            return {}

    def _parse_model_specific_response(
        self, body_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse response data specific to different model formats."""
        data = {}

        # Anthropic Claude format
        if "content" in body_data and isinstance(body_data["content"], list):
            content_data = self._extract_anthropic_content_data(body_data["content"])
            data.update(content_data)

            # Extract usage
            usage = body_data.get("usage", {})
            if usage:
                data["input_tokens"] = usage.get("input_tokens", 0)
                data["output_tokens"] = usage.get("output_tokens", 0)
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

            # Extract metadata
            data["stop_reason"] = body_data.get("stop_reason")
            data["stop_sequence"] = body_data.get("stop_sequence")
            data["model"] = body_data.get("model")

        # Amazon Titan format
        elif "outputText" in body_data:
            output_text = body_data["outputText"]
            data["text_content"] = [output_text]
            data["output_length"] = len(output_text)
            data["content_types"] = ["text"]

            # Extract Titan usage if available
            if "inputTextTokenCount" in body_data:
                data["input_tokens"] = body_data["inputTextTokenCount"]
            if "totalTokenCount" in body_data:
                data["total_tokens"] = body_data["totalTokenCount"]
                if "input_tokens" in data:
                    data["output_tokens"] = data["total_tokens"] - data["input_tokens"]

            # Extract completion reason
            data["completion_reason"] = body_data.get("completionReason")

        # Cohere Command format
        elif "generations" in body_data:
            generations = body_data["generations"]
            if generations and len(generations) > 0:
                first_gen = generations[0]
                text = first_gen.get("text", "")
                data["text_content"] = [text]
                data["output_length"] = len(text)
                data["content_types"] = ["text"]
                data["finish_reason"] = first_gen.get("finish_reason")

            # Extract Cohere usage
            if "prompt_tokens" in body_data:
                data["input_tokens"] = body_data["prompt_tokens"]
            if "generation_tokens" in body_data:
                data["output_tokens"] = body_data["generation_tokens"]
            if "input_tokens" in data and "output_tokens" in data:
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        # Meta Llama format
        elif "generation" in body_data:
            generation = body_data["generation"]
            data["text_content"] = [generation]
            data["output_length"] = len(generation)
            data["content_types"] = ["text"]

            # Extract Llama usage
            if "prompt_token_count" in body_data:
                data["input_tokens"] = body_data["prompt_token_count"]
            if "generation_token_count" in body_data:
                data["output_tokens"] = body_data["generation_token_count"]
            if "input_tokens" in data and "output_tokens" in data:
                data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

            data["stop_reason"] = body_data.get("stop_reason")

        # Mistral format
        elif "outputs" in body_data:
            outputs = body_data["outputs"]
            if outputs and len(outputs) > 0:
                first_output = outputs[0]
                text = first_output.get("text", "")
                data["text_content"] = [text]
                data["output_length"] = len(text)
                data["content_types"] = ["text"]
                data["stop_reason"] = first_output.get("stop_reason")

        return data

    def _extract_anthropic_content_data(
        self, content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract data from Anthropic content format."""
        data = {}

        text_blocks = []
        content_types = set()

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "unknown")
                content_types.add(block_type)

                if block_type == "text":
                    text = block.get("text", "")
                    text_blocks.append(text)

        data["content_types"] = list(content_types)
        data["text_content"] = text_blocks
        data["output_length"] = sum(len(text) for text in text_blocks)

        return data

    def _extract_converse_message_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from converse message format."""
        data = {}

        # Extract role
        data["role"] = message.get("role", "assistant")

        # Extract content
        content = message.get("content", [])
        if content:
            text_blocks = []
            content_types = set()

            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        text_blocks.append(item["text"])
                        content_types.add("text")
                    elif "toolUse" in item:
                        content_types.add("tool_use")
                        # Extract tool use details
                        tool_use = item["toolUse"]
                        data["tool_name"] = tool_use.get("name")
                        data["tool_use_id"] = tool_use.get("toolUseId")

            data["text_content"] = text_blocks
            data["output_length"] = sum(len(text) for text in text_blocks)
            data["content_types"] = list(content_types)

        return data

    def combine_streaming_events(self, events: List[Any]) -> Dict[str, Any]:
        """Combine streaming response events into a single response for analysis."""

        if not events:
            return {}

        combined_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "text_content": [],
            "content_types": set(),
            "stop_reason": None,
            "finish_reason": None,
        }

        accumulated_text = ""

        for event in events:
            event_data = self._parse_streaming_event(event)

            # Accumulate usage (take max values as they're cumulative)
            for token_field in ["input_tokens", "output_tokens", "total_tokens"]:
                if token_field in event_data:
                    combined_data[token_field] = max(
                        combined_data[token_field], event_data[token_field]
                    )

            # Accumulate text content
            if "text_delta" in event_data:
                accumulated_text += event_data["text_delta"]
                combined_data["content_types"].add("text")

            # Collect content types
            if "content_types" in event_data:
                combined_data["content_types"].update(event_data["content_types"])

            # Update metadata (last event wins)
            for field in ["stop_reason", "finish_reason"]:
                if event_data.get(field) is not None:
                    combined_data[field] = event_data[field]

        # Finalize combined data
        if accumulated_text:
            combined_data["text_content"] = [accumulated_text]
            combined_data["output_length"] = len(accumulated_text)

        combined_data["content_types"] = list(combined_data["content_types"])

        return combined_data

    def _parse_streaming_event(self, event: Any) -> Dict[str, Any]:
        """Parse a single streaming event."""
        data = {}

        if isinstance(event, dict):
            # Handle different streaming event types

            # Content block delta (Anthropic format)
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    data["text_delta"] = delta["text"]

            # Message delta (Anthropic format)
            elif "messageDelta" in event:
                delta = event["messageDelta"].get("delta", {})
                if "stop_reason" in delta:
                    data["stop_reason"] = delta["stop_reason"]

            # Usage (various formats)
            elif "messageStop" in event:
                usage = event["messageStop"].get("usage", {})
                if usage:
                    data["input_tokens"] = usage.get("inputTokens", 0)
                    data["output_tokens"] = usage.get("outputTokens", 0)
                    data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

            # Converse stream format
            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    data["text_delta"] = delta["text"]

            # Generic streaming formats
            elif "chunk" in event:
                chunk = event["chunk"]
                if isinstance(chunk, dict):
                    # Try to extract text from various chunk formats
                    for text_field in ["text", "token", "generated_text"]:
                        if text_field in chunk:
                            data["text_delta"] = chunk[text_field]
                            break

        return data
