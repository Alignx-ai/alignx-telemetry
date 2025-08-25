"""AlignX Google GenAI Response Parser

Parses Google GenAI API responses to extract usage data, metadata, and handles streaming responses.
"""

import logging
from typing import Any, Dict, List, Optional, Union

try:
    from google.generativeai.types import GenerateContentResponse

    GOOGLE_GENERATIVEAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENERATIVEAI_AVAILABLE = False
    GenerateContentResponse = None

try:
    from google.genai.types import GenerateContentResponse as NewGenerateContentResponse

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    NewGenerateContentResponse = None

logger = logging.getLogger(__name__)


class GoogleGenAIResponseParser:
    """Parses Google GenAI API responses to extract usage and metadata."""

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract structured data from Google GenAI API response."""

        if not response:
            return {}

        # Handle different response types
        if self._is_generate_content_response(response):
            return self._parse_generate_content_response(response)
        elif self._is_count_tokens_response(response):
            return self._parse_count_tokens_response(response)
        elif isinstance(response, dict):
            return self._parse_dict_response(response)
        else:
            # Fallback for unknown response types
            return self._parse_generic_response(response)

    def _is_generate_content_response(self, response: Any) -> bool:
        """Check if response is a GenerateContentResponse."""
        if GOOGLE_GENERATIVEAI_AVAILABLE and isinstance(
            response, GenerateContentResponse
        ):
            return True
        if GOOGLE_GENAI_AVAILABLE and isinstance(response, NewGenerateContentResponse):
            return True
        if hasattr(response, "candidates") and hasattr(response, "usage_metadata"):
            return True
        return False

    def _is_count_tokens_response(self, response: Any) -> bool:
        """Check if response is a count tokens response."""
        if hasattr(response, "total_tokens") and not hasattr(response, "candidates"):
            return True
        if (
            isinstance(response, dict)
            and "total_tokens" in response
            and "candidates" not in response
        ):
            return True
        return False

    def _parse_generate_content_response(self, response) -> Dict[str, Any]:
        """Parse a GenerateContentResponse."""
        data = {}

        # Extract usage metadata
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            data["input_tokens"] = getattr(usage, "prompt_token_count", 0)
            data["output_tokens"] = getattr(usage, "candidates_token_count", 0)
            data["total_tokens"] = getattr(usage, "total_token_count", 0)

            # Handle cached content tokens if available
            if hasattr(usage, "cached_content_token_count"):
                data["cached_tokens"] = getattr(usage, "cached_content_token_count", 0)

        # Extract candidates and content
        if hasattr(response, "candidates") and response.candidates:
            candidates_data = self._extract_candidates_data(response.candidates)
            data.update(candidates_data)

        # Extract prompt feedback if available
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            prompt_feedback = response.prompt_feedback
            data["prompt_blocked"] = (
                getattr(prompt_feedback, "block_reason", None) is not None
            )
            if hasattr(prompt_feedback, "safety_ratings"):
                data["prompt_safety_ratings"] = len(prompt_feedback.safety_ratings)

        return data

    def _parse_count_tokens_response(self, response) -> Dict[str, Any]:
        """Parse a count tokens response."""
        data = {}

        if hasattr(response, "total_tokens"):
            data["total_tokens"] = response.total_tokens
        elif isinstance(response, dict) and "total_tokens" in response:
            data["total_tokens"] = response["total_tokens"]

        return data

    def _parse_dict_response(self, response: dict) -> Dict[str, Any]:
        """Parse a dictionary response."""
        data = {}

        # Extract usage metadata
        if "usageMetadata" in response:
            usage = response["usageMetadata"]
            data["input_tokens"] = usage.get("promptTokenCount", 0)
            data["output_tokens"] = usage.get("candidatesTokenCount", 0)
            data["total_tokens"] = usage.get("totalTokenCount", 0)

            if "cachedContentTokenCount" in usage:
                data["cached_tokens"] = usage["cachedContentTokenCount"]

        # Extract candidates
        if "candidates" in response:
            candidates_data = self._extract_candidates_data(response["candidates"])
            data.update(candidates_data)

        # Extract prompt feedback
        if "promptFeedback" in response:
            prompt_feedback = response["promptFeedback"]
            data["prompt_blocked"] = prompt_feedback.get("blockReason") is not None
            if "safetyRatings" in prompt_feedback:
                data["prompt_safety_ratings"] = len(prompt_feedback["safetyRatings"])

        # Handle count tokens response
        if "totalTokens" in response:
            data["total_tokens"] = response["totalTokens"]

        return data

    def _parse_generic_response(self, response: Any) -> Dict[str, Any]:
        """Parse a generic response object."""
        data = {}

        # Try to extract common attributes
        for attr in [
            "total_tokens",
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
        ]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if attr == "total_tokens" or attr == "total_token_count":
                    data["total_tokens"] = value
                elif attr == "prompt_token_count":
                    data["input_tokens"] = value
                elif attr == "candidates_token_count":
                    data["output_tokens"] = value

        # Calculate total if not present
        if (
            "total_tokens" not in data
            and "input_tokens" in data
            and "output_tokens" in data
        ):
            data["total_tokens"] = data["input_tokens"] + data["output_tokens"]

        # Try to extract candidates
        if hasattr(response, "candidates"):
            candidates_data = self._extract_candidates_data(response.candidates)
            data.update(candidates_data)

        return data

    def _extract_candidates_data(self, candidates: Any) -> Dict[str, Any]:
        """Extract data from response candidates."""
        data = {}

        if not candidates:
            return data

        # Ensure candidates is a list
        if not isinstance(candidates, list):
            candidates = [candidates]

        data["candidate_count"] = len(candidates)

        # Extract data from first candidate (most common case)
        if candidates:
            first_candidate = candidates[0]

            # Extract finish reason
            if isinstance(first_candidate, dict):
                data["finish_reason"] = first_candidate.get("finishReason")
                data["blocked"] = first_candidate.get("finishReason") == "SAFETY"

                # Extract content
                if "content" in first_candidate:
                    content_data = self._extract_content_data(
                        first_candidate["content"]
                    )
                    data.update(content_data)

                # Extract safety ratings
                if "safetyRatings" in first_candidate:
                    data["safety_ratings"] = len(first_candidate["safetyRatings"])
                    data["blocked"] = any(
                        rating.get("blocked", False)
                        for rating in first_candidate["safetyRatings"]
                    )

            elif hasattr(first_candidate, "finish_reason"):
                data["finish_reason"] = getattr(first_candidate, "finish_reason", None)
                data["blocked"] = data["finish_reason"] == "SAFETY"

                # Extract content
                if hasattr(first_candidate, "content"):
                    content_data = self._extract_content_data(first_candidate.content)
                    data.update(content_data)

                # Extract safety ratings
                if hasattr(first_candidate, "safety_ratings"):
                    safety_ratings = first_candidate.safety_ratings
                    data["safety_ratings"] = len(safety_ratings)
                    data["blocked"] = any(
                        getattr(rating, "blocked", False) for rating in safety_ratings
                    )

        return data

    def _extract_content_data(self, content: Any) -> Dict[str, Any]:
        """Extract data from content field."""
        data = {}

        if not content:
            return data

        # Handle different content formats
        if isinstance(content, dict):
            parts = content.get("parts", [])
        elif hasattr(content, "parts"):
            parts = content.parts
        else:
            parts = []

        if parts:
            text_parts = []
            content_types = set()

            for part in parts:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(part["text"])
                        content_types.add("text")
                    elif "functionCall" in part:
                        content_types.add("function_call")
                    elif "functionResponse" in part:
                        content_types.add("function_response")
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
                    content_types.add("text")
                elif hasattr(part, "function_call"):
                    content_types.add("function_call")
                elif hasattr(part, "function_response"):
                    content_types.add("function_response")

            data["text_content"] = text_parts
            data["output_length"] = sum(len(text) for text in text_parts)
            data["content_types"] = list(content_types)

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
            "finish_reason": None,
            "safety_ratings": 0,
            "blocked": False,
        }

        accumulated_text = ""

        for chunk in chunks:
            chunk_data = self.extract_response_data(chunk)

            # Accumulate usage (take max values as they're cumulative)
            for token_field in ["input_tokens", "output_tokens", "total_tokens"]:
                if token_field in chunk_data:
                    combined_data[token_field] = max(
                        combined_data[token_field], chunk_data[token_field]
                    )

            # Accumulate text content
            if "text_content" in chunk_data:
                for text in chunk_data["text_content"]:
                    accumulated_text += text

            # Collect content types
            if "content_types" in chunk_data:
                combined_data["content_types"].update(chunk_data["content_types"])

            # Update metadata (last chunk wins)
            for field in ["finish_reason", "safety_ratings", "blocked"]:
                if chunk_data.get(field) is not None:
                    combined_data[field] = chunk_data[field]

        # Finalize combined data
        if accumulated_text:
            combined_data["text_content"] = [accumulated_text]
            combined_data["output_length"] = len(accumulated_text)

        combined_data["content_types"] = list(combined_data["content_types"])

        return combined_data
