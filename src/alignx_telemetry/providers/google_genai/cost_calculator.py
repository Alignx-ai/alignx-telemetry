"""AlignX Google GenAI Cost Calculator

Calculates usage costs for Google's Generative AI models (Gemini) based on current pricing.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GoogleGenAICostCalculator:
    """Calculates costs for Google GenAI model usage."""

    # Pricing data as of December 2024 (per 1,000 tokens)
    # Source: https://ai.google.dev/pricing
    PRICING_DATA = {
        # Gemini 2.0 Flash Experimental (Free during preview)
        "gemini-2.0-flash": {
            "input": 0.0,  # Free during preview
            "output": 0.0,  # Free during preview
        },
        "gemini-2.0-flash-exp": {
            "input": 0.0,
            "output": 0.0,
        },
        # Gemini 1.5 Flash
        "gemini-1.5-flash": {
            "input": 0.000075,  # $0.075 per 1M tokens
            "output": 0.0003,  # $0.30 per 1M tokens
        },
        "gemini-1.5-flash-latest": {
            "input": 0.000075,
            "output": 0.0003,
        },
        "gemini-1.5-flash-002": {
            "input": 0.000075,
            "output": 0.0003,
        },
        "gemini-1.5-flash-8b": {
            "input": 0.0000375,  # $0.0375 per 1M tokens (50% discount)
            "output": 0.00015,  # $0.15 per 1M tokens (50% discount)
        },
        # Gemini 1.5 Pro
        "gemini-1.5-pro": {
            "input": 0.00125,  # $1.25 per 1M tokens
            "output": 0.005,  # $5.00 per 1M tokens
        },
        "gemini-1.5-pro-latest": {
            "input": 0.00125,
            "output": 0.005,
        },
        "gemini-1.5-pro-002": {
            "input": 0.00125,
            "output": 0.005,
        },
        # Gemini 1.0 Pro (Legacy)
        "gemini-1.0-pro": {
            "input": 0.0005,  # $0.50 per 1M tokens
            "output": 0.0015,  # $1.50 per 1M tokens
        },
        "gemini-pro": {
            "input": 0.0005,
            "output": 0.0015,
        },
        # Gemini 1.0 Pro Vision (Legacy)
        "gemini-1.0-pro-vision": {
            "input": 0.00025,  # $0.25 per 1M tokens
            "output": 0.0005,  # $0.50 per 1M tokens
        },
        "gemini-pro-vision": {
            "input": 0.00025,
            "output": 0.0005,
        },
        # Gemini Nano (Edge/Mobile - typically free for developers)
        "gemini-nano": {
            "input": 0.0,
            "output": 0.0,
        },
    }

    # Model aliases for normalization
    MODEL_ALIASES = {
        # Latest aliases
        "gemini-1.5-flash-latest": "gemini-1.5-flash",
        "gemini-1.5-pro-latest": "gemini-1.5-pro",
        "gemini-2.0-flash-exp": "gemini-2.0-flash",
        # Legacy aliases
        "gemini-pro": "gemini-1.0-pro",
        "gemini-pro-vision": "gemini-1.0-pro-vision",
        # Version-specific mappings
        "models/gemini-1.5-flash": "gemini-1.5-flash",
        "models/gemini-1.5-pro": "gemini-1.5-pro",
        "models/gemini-1.0-pro": "gemini-1.0-pro",
        "models/gemini-pro": "gemini-1.0-pro",
        "publishers/google/models/gemini-1.5-flash": "gemini-1.5-flash",
        "publishers/google/models/gemini-1.5-pro": "gemini-1.5-pro",
    }

    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """
        Calculate the cost of a Google GenAI API call.

        Args:
            model: The model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached content tokens (if applicable)

        Returns:
            Total cost in USD
        """
        try:
            # Normalize model name
            normalized_model = self._normalize_model_name(model)

            # Get pricing info
            pricing = self._get_pricing_info(normalized_model)
            if not pricing:
                logger.warning(f"No pricing information available for model: {model}")
                return 0.0

            # Calculate costs
            # Note: Cached tokens are typically charged at reduced rates or free
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]

            # For cached tokens, use 50% discount on input pricing (Google's typical approach)
            cached_cost = (
                (cached_tokens / 1000) * pricing["input"] * 0.5
                if cached_tokens > 0
                else 0.0
            )

            total_cost = input_cost + output_cost + cached_cost

            logger.debug(
                f"Cost calculation for {normalized_model}: "
                f"input_tokens={input_tokens} (${input_cost:.6f}), "
                f"output_tokens={output_tokens} (${output_cost:.6f}), "
                f"cached_tokens={cached_tokens} (${cached_cost:.6f}), "
                f"total=${total_cost:.6f}"
            )

            return total_cost

        except Exception as e:
            logger.error(f"Error calculating cost for model {model}: {e}")
            return 0.0

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for consistent pricing lookup."""
        if not model:
            return "unknown"

        # Convert to lowercase for consistent matching
        model_lower = model.lower().strip()

        # Remove common prefixes
        model_lower = model_lower.replace("models/", "")
        model_lower = model_lower.replace("publishers/google/models/", "")

        # Check direct aliases first
        if model_lower in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model_lower]

        # Check if it's already a known model
        if model_lower in self.PRICING_DATA:
            return model_lower

        # Try to match partial model names
        for known_model in self.PRICING_DATA.keys():
            if known_model in model_lower or model_lower in known_model:
                return known_model

        # Return original if no match found
        return model_lower

    def _get_pricing_info(self, model: str) -> Optional[Dict[str, float]]:
        """Get pricing information for a specific model."""
        return self.PRICING_DATA.get(model)

    def get_model_pricing_info(self, model: str) -> Optional[Dict[str, any]]:
        """
        Get detailed pricing information for a model.

        Args:
            model: The model name

        Returns:
            Dictionary with pricing details or None if not found
        """
        normalized_model = self._normalize_model_name(model)
        pricing = self._get_pricing_info(normalized_model)

        if not pricing:
            return None

        return {
            "model": normalized_model,
            "original_model": model,
            "input_cost_per_1k": pricing["input"],
            "output_cost_per_1k": pricing["output"],
            "input_cost_per_1m": pricing["input"] * 1000,
            "output_cost_per_1m": pricing["output"] * 1000,
            "cached_cost_per_1k": pricing["input"] * 0.5,  # 50% discount for cached
            "currency": "USD",
            "free_tier": pricing["input"] == 0.0 and pricing["output"] == 0.0,
        }

    def estimate_cost(
        self,
        model: str,
        input_text: str = "",
        estimated_output_tokens: int = 0,
        cached_content_length: int = 0,
    ) -> Dict[str, float]:
        """
        Estimate cost before making an API call.

        Args:
            model: The model name
            input_text: Input text to estimate tokens for
            estimated_output_tokens: Estimated output tokens
            cached_content_length: Length of cached content (characters)

        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimation (actual tokenization would be more accurate)
        # Google typically uses ~4 characters per token for English text
        estimated_input_tokens = len(input_text) / 4
        estimated_cached_tokens = cached_content_length / 4

        pricing = self._get_pricing_info(self._normalize_model_name(model))
        if not pricing:
            return {"error": "Model pricing not available"}

        input_cost = (estimated_input_tokens / 1000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1000) * pricing["output"]
        cached_cost = (estimated_cached_tokens / 1000) * pricing["input"] * 0.5
        total_cost = input_cost + output_cost + cached_cost

        return {
            "estimated_input_tokens": int(estimated_input_tokens),
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cached_tokens": int(estimated_cached_tokens),
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_cached_cost": cached_cost,
            "estimated_total_cost": total_cost,
            "currency": "USD",
            "model": self._normalize_model_name(model),
            "free_tier": pricing["input"] == 0.0 and pricing["output"] == 0.0,
        }

    def get_context_caching_savings(
        self, model: str, total_tokens: int, cached_percentage: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate potential savings from context caching.

        Args:
            model: The model name
            total_tokens: Total tokens in the context
            cached_percentage: Percentage of tokens that would be cached (0.0-1.0)

        Returns:
            Dictionary with savings information
        """
        cached_tokens = int(total_tokens * cached_percentage)
        uncached_tokens = total_tokens - cached_tokens

        pricing = self._get_pricing_info(self._normalize_model_name(model))
        if not pricing:
            return {"error": "Model pricing not available"}

        # Cost without caching (all tokens at full price)
        full_cost = (total_tokens / 1000) * pricing["input"]

        # Cost with caching (cached tokens at 50% discount)
        cached_cost = (uncached_tokens / 1000) * pricing["input"] + (
            cached_tokens / 1000
        ) * pricing["input"] * 0.5

        savings = full_cost - cached_cost
        savings_percentage = (savings / full_cost * 100) if full_cost > 0 else 0

        return {
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "uncached_tokens": uncached_tokens,
            "full_cost": full_cost,
            "cached_cost": cached_cost,
            "savings": savings,
            "savings_percentage": savings_percentage,
            "currency": "USD",
            "model": self._normalize_model_name(model),
        }
