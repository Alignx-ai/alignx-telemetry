"""OpenAI cost calculation utilities.

This module provides accurate cost calculation for OpenAI API usage
based on current pricing models and token consumption.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class OpenAICostCalculator:
    """Calculates costs for OpenAI API usage based on current pricing.

    This class maintains up-to-date pricing information for OpenAI models
    and provides accurate cost calculations for various operation types.
    """

    # OpenAI pricing as of 2024 (USD per token)
    # Updated regularly to reflect current pricing
    PRICING_DATA = {
        # GPT-4 Models
        "gpt-4": {
            "input": 0.00003,  # $30 per 1M tokens
            "output": 0.00006,  # $60 per 1M tokens
        },
        "gpt-4-32k": {
            "input": 0.00006,  # $60 per 1M tokens
            "output": 0.00012,  # $120 per 1M tokens
        },
        "gpt-4-turbo": {
            "input": 0.00001,  # $10 per 1M tokens
            "output": 0.00003,  # $30 per 1M tokens
        },
        "gpt-4-turbo-preview": {
            "input": 0.00001,  # $10 per 1M tokens
            "output": 0.00003,  # $30 per 1M tokens
        },
        "gpt-4-1106-preview": {
            "input": 0.00001,  # $10 per 1M tokens
            "output": 0.00003,  # $30 per 1M tokens
        },
        "gpt-4-0125-preview": {
            "input": 0.00001,  # $10 per 1M tokens
            "output": 0.00003,  # $30 per 1M tokens
        },
        "gpt-4-vision-preview": {
            "input": 0.00001,  # $10 per 1M tokens
            "output": 0.00003,  # $30 per 1M tokens
        },
        "gpt-4o": {
            "input": 0.000005,  # $5 per 1M tokens
            "output": 0.000015,  # $15 per 1M tokens
        },
        "gpt-4o-mini": {
            "input": 0.00000015,  # $0.15 per 1M tokens
            "output": 0.0000006,  # $0.6 per 1M tokens
        },
        # GPT-3.5 Models
        "gpt-3.5-turbo": {
            "input": 0.0000015,  # $1.50 per 1M tokens
            "output": 0.000002,  # $2.00 per 1M tokens
        },
        "gpt-3.5-turbo-16k": {
            "input": 0.000003,  # $3.00 per 1M tokens
            "output": 0.000004,  # $4.00 per 1M tokens
        },
        "gpt-3.5-turbo-instruct": {
            "input": 0.0000015,  # $1.50 per 1M tokens
            "output": 0.000002,  # $2.00 per 1M tokens
        },
        "gpt-3.5-turbo-0125": {
            "input": 0.0000005,  # $0.50 per 1M tokens
            "output": 0.0000015,  # $1.50 per 1M tokens
        },
        "gpt-3.5-turbo-1106": {
            "input": 0.000001,  # $1.00 per 1M tokens
            "output": 0.000002,  # $2.00 per 1M tokens
        },
        # Legacy GPT-3 Models
        "text-davinci-003": {
            "input": 0.00002,  # $20 per 1M tokens
            "output": 0.00002,  # $20 per 1M tokens
        },
        "text-davinci-002": {
            "input": 0.00002,  # $20 per 1M tokens
            "output": 0.00002,  # $20 per 1M tokens
        },
        "code-davinci-002": {
            "input": 0.00002,  # $20 per 1M tokens
            "output": 0.00002,  # $20 per 1M tokens
        },
        # Embedding Models
        "text-embedding-ada-002": {
            "input": 0.0000001,  # $0.10 per 1M tokens
            "output": 0.0,  # No output tokens for embeddings
        },
        "text-embedding-3-small": {
            "input": 0.00000002,  # $0.02 per 1M tokens
            "output": 0.0,
        },
        "text-embedding-3-large": {
            "input": 0.00000013,  # $0.13 per 1M tokens
            "output": 0.0,
        },
        # Image Models (per image, converted to per-token equivalent)
        "dall-e-2": {
            "input": 0.02,  # $0.02 per image (1024x1024)
            "output": 0.0,
        },
        "dall-e-3": {
            "input": 0.04,  # $0.04 per image (1024x1024 standard)
            "output": 0.0,
        },
        # Audio Models
        "whisper-1": {
            "input": 0.006,  # $0.006 per minute
            "output": 0.0,
        },
        "tts-1": {
            "input": 0.000015,  # $15 per 1M characters
            "output": 0.0,
        },
        "tts-1-hd": {
            "input": 0.00003,  # $30 per 1M characters
            "output": 0.0,
        },
    }

    # Model aliases and variations
    MODEL_ALIASES = {
        "gpt-4-0314": "gpt-4",
        "gpt-4-0613": "gpt-4",
        "gpt-4-32k-0314": "gpt-4-32k",
        "gpt-4-32k-0613": "gpt-4-32k",
        "gpt-3.5-turbo-0301": "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613": "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k-0613": "gpt-3.5-turbo-16k",
    }

    def __init__(self):
        """Initialize the cost calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        operation_type: str = "completion",
    ) -> float:
        """Calculate cost for OpenAI API usage.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation ("completion", "embedding", "image", etc.)

        Returns:
            Cost in USD
        """
        try:
            # Normalize model name
            normalized_model = self._normalize_model_name(model)

            # Get pricing for the model
            pricing = self._get_model_pricing(normalized_model)

            if not pricing:
                self.logger.debug(f"No pricing available for model: {model}")
                return 0.0

            # Calculate cost based on operation type
            if operation_type == "image":
                return self._calculate_image_cost(normalized_model, input_tokens)
            elif operation_type == "audio":
                return self._calculate_audio_cost(normalized_model, input_tokens)
            else:
                # Standard token-based pricing
                input_cost = input_tokens * pricing["input"]
                output_cost = output_tokens * pricing["output"]
                total_cost = input_cost + output_cost

                self.logger.debug(
                    f"Cost calculation - Model: {model}, "
                    f"Input tokens: {input_tokens} (${input_cost:.6f}), "
                    f"Output tokens: {output_tokens} (${output_cost:.6f}), "
                    f"Total: ${total_cost:.6f}"
                )

                return total_cost

        except Exception as e:
            self.logger.error(f"Error calculating cost: {e}")
            return 0.0

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for pricing lookup.

        Args:
            model: Raw model name

        Returns:
            Normalized model name
        """
        if not model:
            return "unknown"

        model = model.lower().strip()

        # Check for direct alias match
        if model in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model]

        # Check for partial matches
        for alias, canonical in self.MODEL_ALIASES.items():
            if model.startswith(alias):
                return canonical

        # For fine-tuned models, extract base model
        if model.startswith("ft:"):
            # Fine-tuned model format: ft:gpt-3.5-turbo:org:name:id
            parts = model.split(":")
            if len(parts) >= 2:
                base_model = parts[1]
                return self._normalize_model_name(base_model)

        return model

    def _get_model_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Get pricing data for a model.

        Args:
            model: Normalized model name

        Returns:
            Pricing dictionary or None if not found
        """
        # Direct lookup
        if model in self.PRICING_DATA:
            return self.PRICING_DATA[model]

        # Fuzzy matching for similar models
        for pricing_model in self.PRICING_DATA:
            if model.startswith(pricing_model):
                return self.PRICING_DATA[pricing_model]

        # Default fallback for unknown GPT models
        if "gpt-4" in model:
            return self.PRICING_DATA.get("gpt-4")
        elif "gpt-3.5" in model:
            return self.PRICING_DATA.get("gpt-3.5-turbo")
        elif "davinci" in model:
            return self.PRICING_DATA.get("text-davinci-003")
        elif "embedding" in model:
            return self.PRICING_DATA.get("text-embedding-ada-002")

        return None

    def _calculate_image_cost(self, model: str, quantity: int) -> float:
        """Calculate cost for image generation.

        Args:
            model: Image model name
            quantity: Number of images

        Returns:
            Cost in USD
        """
        pricing = self._get_model_pricing(model)
        if not pricing:
            return 0.0

        # Image models have per-image pricing
        cost_per_image = pricing["input"]
        return quantity * cost_per_image

    def _calculate_audio_cost(self, model: str, duration_or_chars: int) -> float:
        """Calculate cost for audio operations.

        Args:
            model: Audio model name
            duration_or_chars: Duration in seconds (Whisper) or character count (TTS)

        Returns:
            Cost in USD
        """
        pricing = self._get_model_pricing(model)
        if not pricing:
            return 0.0

        if "whisper" in model:
            # Whisper pricing is per minute
            minutes = duration_or_chars / 60.0
            return minutes * pricing["input"]
        elif "tts" in model:
            # TTS pricing is per character
            return duration_or_chars * pricing["input"]

        return 0.0

    def get_model_pricing_info(self, model: str) -> Dict[str, any]:
        """Get detailed pricing information for a model.

        Args:
            model: Model name

        Returns:
            Dictionary with pricing details
        """
        normalized_model = self._normalize_model_name(model)
        pricing = self._get_model_pricing(normalized_model)

        if not pricing:
            return {
                "model": model,
                "normalized_model": normalized_model,
                "pricing_available": False,
                "input_price_per_token": 0.0,
                "output_price_per_token": 0.0,
            }

        return {
            "model": model,
            "normalized_model": normalized_model,
            "pricing_available": True,
            "input_price_per_token": pricing["input"],
            "output_price_per_token": pricing["output"],
            "input_price_per_1k_tokens": pricing["input"] * 1000,
            "output_price_per_1k_tokens": pricing["output"] * 1000,
            "input_price_per_1m_tokens": pricing["input"] * 1000000,
            "output_price_per_1m_tokens": pricing["output"] * 1000000,
        }

    def estimate_cost_from_text(
        self, model: str, input_text: str, output_text: str = ""
    ) -> Tuple[float, Dict[str, int]]:
        """Estimate cost from text content.

        Args:
            model: Model name
            input_text: Input text content
            output_text: Output text content

        Returns:
            Tuple of (estimated_cost, token_counts)
        """
        # Rough token estimation (1 token ≈ 4 characters for English)
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4

        cost = self.calculate_cost(model, input_tokens, output_tokens)

        token_counts = {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": input_tokens + output_tokens,
        }

        return cost, token_counts

    def update_pricing(
        self, model: str, input_price: float, output_price: float
    ) -> None:
        """Update pricing for a model.

        Args:
            model: Model name
            input_price: Price per input token
            output_price: Price per output token
        """
        self.PRICING_DATA[model.lower()] = {
            "input": input_price,
            "output": output_price,
        }
        self.logger.info(
            f"Updated pricing for {model}: input=${input_price}, output=${output_price}"
        )

    def get_all_models(self) -> list:
        """Get list of all models with pricing data.

        Returns:
            List of model names with available pricing
        """
        return list(self.PRICING_DATA.keys())

    def validate_pricing_data(self) -> Dict[str, any]:
        """Validate and return pricing data statistics.

        Returns:
            Dictionary with validation results
        """
        total_models = len(self.PRICING_DATA)
        models_with_output_pricing = sum(
            1 for pricing in self.PRICING_DATA.values() if pricing["output"] > 0
        )

        return {
            "total_models": total_models,
            "models_with_output_pricing": models_with_output_pricing,
            "embedding_models": len([m for m in self.PRICING_DATA if "embedding" in m]),
            "gpt4_models": len([m for m in self.PRICING_DATA if "gpt-4" in m]),
            "gpt35_models": len([m for m in self.PRICING_DATA if "gpt-3.5" in m]),
            "last_updated": datetime.now().isoformat(),
        }
