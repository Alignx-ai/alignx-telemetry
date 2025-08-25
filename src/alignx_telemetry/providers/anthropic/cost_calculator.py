"""AlignX Anthropic Cost Calculator

Calculates usage costs for Anthropic Claude models based on current pricing.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AnthropicCostCalculator:
    """Calculates costs for Anthropic Claude model usage."""

    # Pricing data as of December 2024 (per 1,000 tokens)
    # Source: https://docs.anthropic.com/en/docs/models-overview#model-comparison
    PRICING_DATA = {
        # Claude 3.5 Sonnet (Latest)
        "claude-3.5-sonnet": {
            "input": 0.003,  # $3.00 per 1M tokens
            "output": 0.015,  # $15.00 per 1M tokens
        },
        "claude-3-5-sonnet-20241022": {
            "input": 0.003,
            "output": 0.015,
        },
        "claude-3-5-sonnet-20240620": {
            "input": 0.003,
            "output": 0.015,
        },
        # Claude 3.5 Haiku
        "claude-3.5-haiku": {
            "input": 0.001,  # $1.00 per 1M tokens
            "output": 0.005,  # $5.00 per 1M tokens
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.001,
            "output": 0.005,
        },
        # Claude 3 Opus
        "claude-3-opus": {
            "input": 0.015,  # $15.00 per 1M tokens
            "output": 0.075,  # $75.00 per 1M tokens
        },
        "claude-3-opus-20240229": {
            "input": 0.015,
            "output": 0.075,
        },
        # Claude 3 Sonnet
        "claude-3-sonnet": {
            "input": 0.003,  # $3.00 per 1M tokens
            "output": 0.015,  # $15.00 per 1M tokens
        },
        "claude-3-sonnet-20240229": {
            "input": 0.003,
            "output": 0.015,
        },
        # Claude 3 Haiku
        "claude-3-haiku": {
            "input": 0.00025,  # $0.25 per 1M tokens
            "output": 0.00125,  # $1.25 per 1M tokens
        },
        "claude-3-haiku-20240307": {
            "input": 0.00025,
            "output": 0.00125,
        },
        # Claude 2.1
        "claude-2.1": {
            "input": 0.008,  # $8.00 per 1M tokens
            "output": 0.024,  # $24.00 per 1M tokens
        },
        # Claude 2.0
        "claude-2.0": {
            "input": 0.008,
            "output": 0.024,
        },
        # Claude Instant 1.2
        "claude-instant-1.2": {
            "input": 0.0008,  # $0.80 per 1M tokens
            "output": 0.0024,  # $2.40 per 1M tokens
        },
    }

    # Model aliases for normalization
    MODEL_ALIASES = {
        # Latest model aliases
        "claude-3-5-sonnet-latest": "claude-3.5-sonnet",
        "claude-3-5-haiku-latest": "claude-3.5-haiku",
        "claude-3-opus-latest": "claude-3-opus",
        "claude-3-sonnet-latest": "claude-3-sonnet",
        "claude-3-haiku-latest": "claude-3-haiku",
        # Version-specific aliases
        "claude-3.5-sonnet-20241022": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3.5-haiku-20241022": "claude-3-5-haiku-20241022",
        # Legacy aliases
        "claude-v1": "claude-instant-1.2",
        "claude-v1.3": "claude-instant-1.2",
        "claude-instant": "claude-instant-1.2",
        "claude-instant-v1": "claude-instant-1.2",
        "claude-instant-1": "claude-instant-1.2",
        "claude-2": "claude-2.0",
    }

    def calculate_cost(
        self, model: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> float:
        """
        Calculate the cost of an Anthropic API call.

        Args:
            model: The model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

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
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]
            total_cost = input_cost + output_cost

            logger.debug(
                f"Cost calculation for {normalized_model}: "
                f"input_tokens={input_tokens} (${input_cost:.6f}), "
                f"output_tokens={output_tokens} (${output_cost:.6f}), "
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
            "currency": "USD",
        }

    def estimate_cost(
        self, model: str, input_text: str = "", estimated_output_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Estimate cost before making an API call.

        Args:
            model: The model name
            input_text: Input text to estimate tokens for
            estimated_output_tokens: Estimated output tokens

        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimation (actual tokenization would be more accurate)
        estimated_input_tokens = len(input_text.split()) * 1.3  # Rough approximation

        pricing = self._get_pricing_info(self._normalize_model_name(model))
        if not pricing:
            return {"error": "Model pricing not available"}

        input_cost = (estimated_input_tokens / 1000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "estimated_input_tokens": int(estimated_input_tokens),
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": total_cost,
            "currency": "USD",
            "model": self._normalize_model_name(model),
        }
