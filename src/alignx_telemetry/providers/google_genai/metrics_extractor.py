"""AlignX Google GenAI Metrics Extractor

Extracts structured data and metadata from Google GenAI API requests for tracing and metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Union

try:
    import google.generativeai as genai

    GOOGLE_GENERATIVEAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENERATIVEAI_AVAILABLE = False
    genai = None

try:
    import google.genai as google_genai

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    google_genai = None

from ..base import AlignXProviderCallData

logger = logging.getLogger(__name__)


class GoogleGenAIMetricsExtractor:
    """Extracts metrics and metadata from Google GenAI API calls."""

    def extract_call_data(
        self,
        args: tuple,
        kwargs: dict,
        operation: str = "unknown",
        instance=None,
        package: str = "unknown",
    ) -> AlignXProviderCallData:
        """Extract structured call data from Google GenAI API arguments."""

        # Determine the model being used
        model = self._extract_model(kwargs, instance)

        # Extract operation-specific inputs
        inputs = self._extract_inputs(kwargs, operation)

        # Extract sanitized arguments for debugging
        sanitized_args = self._sanitize_args(kwargs)
        sanitized_args["package"] = package

        return AlignXProviderCallData(
            provider="google_genai",
            model=model,
            operation=operation,
            inputs=inputs,
            args=sanitized_args,
        )

    def _extract_model(self, kwargs: dict, instance=None) -> str:
        """Extract the model name from API call arguments or instance."""

        # Check explicit model parameter
        if "model" in kwargs:
            return self._normalize_model_name(kwargs["model"])

        # Extract from instance if available
        if instance:
            # Check various model attributes
            for attr in ["_model_id", "_model_name", "model_name"]:
                if hasattr(instance, attr):
                    model = getattr(instance, attr)
                    if model:
                        return self._normalize_model_name(str(model))

            # Check nested model object
            if hasattr(instance, "model") and hasattr(instance.model, "model_name"):
                return self._normalize_model_name(instance.model.model_name)

        return "unknown"

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for consistency."""
        if not model:
            return "unknown"

        # Remove common prefixes
        model = model.replace("models/", "")
        model = model.replace("publishers/google/models/", "")

        # Common model mappings
        model_mapping = {
            "gemini-pro": "gemini-1.0-pro",
            "gemini-pro-vision": "gemini-1.0-pro-vision",
            "gemini-1.5-pro-latest": "gemini-1.5-pro",
            "gemini-1.5-flash-latest": "gemini-1.5-flash",
            "gemini-2.0-flash-exp": "gemini-2.0-flash",
        }

        return model_mapping.get(model, model)

    def _extract_inputs(self, kwargs: dict, operation: str) -> Dict[str, Any]:
        """Extract input data from API call arguments."""

        if operation in ["generate_content", "generate_content_async"]:
            return self._extract_generate_content_inputs(kwargs)
        elif operation in ["count_tokens", "count_tokens_async"]:
            return self._extract_count_tokens_inputs(kwargs)
        else:
            return {}

    def _extract_generate_content_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from generate_content API call."""
        inputs = {}

        # Extract contents
        contents = kwargs.get("contents") or kwargs.get("prompt", [])
        if contents:
            if isinstance(contents, str):
                # Single string prompt
                inputs["prompt_length"] = len(contents)
                inputs["content_count"] = 1
                inputs["content_types"] = ["text"]
            elif isinstance(contents, list):
                # List of content parts
                inputs["content_count"] = len(contents)
                total_length = 0
                content_types = set()

                for content in contents:
                    if isinstance(content, str):
                        total_length += len(content)
                        content_types.add("text")
                    elif isinstance(content, dict):
                        if "text" in content:
                            total_length += len(content["text"])
                            content_types.add("text")
                        elif "image" in content or "inline_data" in content:
                            content_types.add("image")
                        elif "audio" in content:
                            content_types.add("audio")
                        elif "video" in content:
                            content_types.add("video")

                inputs["total_input_length"] = total_length
                inputs["content_types"] = list(content_types)

        # Extract generation config
        generation_config = kwargs.get("generation_config") or kwargs.get("config")
        if generation_config:
            if isinstance(generation_config, dict):
                config_data = generation_config
            else:
                # Handle GenerationConfig object
                config_data = getattr(generation_config, "__dict__", {})

            for param in [
                "max_output_tokens",
                "temperature",
                "top_p",
                "top_k",
                "candidate_count",
            ]:
                if param in config_data:
                    inputs[param] = config_data[param]

        # Extract safety settings
        safety_settings = kwargs.get("safety_settings")
        if safety_settings:
            inputs["safety_settings_count"] = (
                len(safety_settings) if isinstance(safety_settings, list) else 1
            )

        # Extract streaming flag
        inputs["stream"] = kwargs.get("stream", False)

        return inputs

    def _extract_count_tokens_inputs(self, kwargs: dict) -> Dict[str, Any]:
        """Extract inputs from count_tokens API call."""
        inputs = {}

        # Extract contents
        contents = kwargs.get("contents") or kwargs.get("prompt", [])
        if contents:
            if isinstance(contents, str):
                inputs["prompt_length"] = len(contents)
            elif isinstance(contents, list):
                total_length = sum(
                    (
                        len(content)
                        if isinstance(content, str)
                        else (
                            len(content.get("text", ""))
                            if isinstance(content, dict) and "text" in content
                            else 0
                        )
                    )
                    for content in contents
                )
                inputs["total_input_length"] = total_length

        return inputs

    def _sanitize_args(self, kwargs: dict) -> Dict[str, Any]:
        """Create a sanitized version of arguments for debugging/tracing."""
        sanitized = {}

        # Safe parameters to include
        safe_params = {
            "model",
            "stream",
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "stop_sequences",
        }

        for param in safe_params:
            if param in kwargs:
                sanitized[param] = kwargs[param]

        # Handle generation_config specially
        if "generation_config" in kwargs:
            config = kwargs["generation_config"]
            if isinstance(config, dict):
                sanitized["generation_config"] = {
                    k: v for k, v in config.items() if k in safe_params
                }
            else:
                # Handle GenerationConfig object
                sanitized["generation_config"] = "GenerationConfig object"

        # Handle contents specially (sanitize and truncate)
        contents = kwargs.get("contents") or kwargs.get("prompt")
        if contents:
            sanitized["contents"] = self._sanitize_contents(contents)

        # Handle safety settings
        if "safety_settings" in kwargs:
            safety_settings = kwargs["safety_settings"]
            if isinstance(safety_settings, list):
                sanitized["safety_settings"] = (
                    f"[{len(safety_settings)} safety settings]"
                )
            else:
                sanitized["safety_settings"] = "safety settings configured"

        return sanitized

    def _sanitize_contents(self, contents: Any) -> Any:
        """Sanitize contents for safe logging."""
        if isinstance(contents, str):
            # Truncate long strings
            if len(contents) > 500:
                return contents[:500] + "... [truncated]"
            return contents

        elif isinstance(contents, list):
            sanitized_contents = []

            for content in contents:
                if isinstance(content, str):
                    if len(content) > 200:
                        sanitized_contents.append(content[:200] + "... [truncated]")
                    else:
                        sanitized_contents.append(content)
                elif isinstance(content, dict):
                    sanitized_content = {}

                    if "text" in content:
                        text = content["text"]
                        if len(text) > 200:
                            sanitized_content["text"] = text[:200] + "... [truncated]"
                        else:
                            sanitized_content["text"] = text

                    if "image" in content or "inline_data" in content:
                        sanitized_content["content_type"] = "image"
                        sanitized_content["data"] = "[image_data_omitted]"
                    elif "audio" in content:
                        sanitized_content["content_type"] = "audio"
                        sanitized_content["data"] = "[audio_data_omitted]"
                    elif "video" in content:
                        sanitized_content["content_type"] = "video"
                        sanitized_content["data"] = "[video_data_omitted]"
                    else:
                        # Include other safe fields
                        for key in ["role", "parts"]:
                            if key in content:
                                sanitized_content[key] = content[key]

                    sanitized_contents.append(sanitized_content)
                else:
                    sanitized_contents.append(str(content))

            return sanitized_contents

        else:
            return str(contents)
