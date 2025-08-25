"""AlignX Bedrock Cost Calculator

Calculates usage costs for AWS Bedrock foundation models based on current pricing.
Supports multiple model providers available through Bedrock.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BedrockCostCalculator:
    """Calculates costs for AWS Bedrock foundation models."""

    # Pricing data as of December 2024 (per 1,000 tokens)
    # Source: https://aws.amazon.com/bedrock/pricing/
    PRICING_DATA = {
        # Anthropic Claude models on Bedrock
        "claude-3.5-sonnet": {
            "input": 0.003,  # $3.00 per 1M tokens
            "output": 0.015,  # $15.00 per 1M tokens
        },
        "claude-3.5-haiku": {
            "input": 0.001,  # $1.00 per 1M tokens
            "output": 0.005,  # $5.00 per 1M tokens
        },
        "claude-3-opus": {
            "input": 0.015,  # $15.00 per 1M tokens
            "output": 0.075,  # $75.00 per 1M tokens
        },
        "claude-3-sonnet": {
            "input": 0.003,  # $3.00 per 1M tokens
            "output": 0.015,  # $15.00 per 1M tokens
        },
        "claude-3-haiku": {
            "input": 0.00025,  # $0.25 per 1M tokens
            "output": 0.00125,  # $1.25 per 1M tokens
        },
        "claude-2.1": {
            "input": 0.008,  # $8.00 per 1M tokens
            "output": 0.024,  # $24.00 per 1M tokens
        },
        "claude-2.0": {
            "input": 0.008,  # $8.00 per 1M tokens
            "output": 0.024,  # $24.00 per 1M tokens
        },
        "claude-instant-1.2": {
            "input": 0.0008,  # $0.80 per 1M tokens
            "output": 0.0024,  # $2.40 per 1M tokens
        },
        # Amazon Titan models
        "titan-text-lite": {
            "input": 0.0003,  # $0.30 per 1M tokens
            "output": 0.0004,  # $0.40 per 1M tokens
        },
        "titan-text-express": {
            "input": 0.0013,  # $1.30 per 1M tokens
            "output": 0.0017,  # $1.70 per 1M tokens
        },
        "titan-text-premier": {
            "input": 0.0005,  # $0.50 per 1M tokens
            "output": 0.0015,  # $1.50 per 1M tokens
        },
        "titan-embed-text": {
            "input": 0.0001,  # $0.10 per 1M tokens (embedding)
            "output": 0.0,  # No output cost for embeddings
        },
        "titan-embed-text-v2": {
            "input": 0.00002,  # $0.02 per 1M tokens (embedding)
            "output": 0.0,  # No output cost for embeddings
        },
        "titan-embed-image": {
            "input": 0.0001,  # $0.10 per 1M tokens (embedding)
            "output": 0.0,  # No output cost for embeddings
        },
        "titan-image-generator": {
            "input": 0.008,  # $8.00 per image (converted to token equivalent)
            "output": 0.0,  # No output cost for image generation
        },
        # Cohere Command models
        "command-text": {
            "input": 0.0015,  # $1.50 per 1M tokens
            "output": 0.002,  # $2.00 per 1M tokens
        },
        "command-light-text": {
            "input": 0.0003,  # $0.30 per 1M tokens
            "output": 0.0006,  # $0.60 per 1M tokens
        },
        "command-r": {
            "input": 0.0005,  # $0.50 per 1M tokens
            "output": 0.0015,  # $1.50 per 1M tokens
        },
        "command-r-plus": {
            "input": 0.003,  # $3.00 per 1M tokens
            "output": 0.015,  # $15.00 per 1M tokens
        },
        "embed-english": {
            "input": 0.0001,  # $0.10 per 1M tokens (embedding)
            "output": 0.0,  # No output cost for embeddings
        },
        "embed-multilingual": {
            "input": 0.0001,  # $0.10 per 1M tokens (embedding)
            "output": 0.0,  # No output cost for embeddings
        },
        # Meta Llama models
        "llama2-13b-chat": {
            "input": 0.00075,  # $0.75 per 1M tokens
            "output": 0.001,  # $1.00 per 1M tokens
        },
        "llama2-70b-chat": {
            "input": 0.00195,  # $1.95 per 1M tokens
            "output": 0.00256,  # $2.56 per 1M tokens
        },
        "llama3-8b-instruct": {
            "input": 0.0003,  # $0.30 per 1M tokens
            "output": 0.0006,  # $0.60 per 1M tokens
        },
        "llama3-70b-instruct": {
            "input": 0.00265,  # $2.65 per 1M tokens
            "output": 0.0035,  # $3.50 per 1M tokens
        },
        "llama3.1-8b-instruct": {
            "input": 0.0003,  # $0.30 per 1M tokens
            "output": 0.0006,  # $0.60 per 1M tokens
        },
        "llama3.1-70b-instruct": {
            "input": 0.00265,  # $2.65 per 1M tokens
            "output": 0.0035,  # $3.50 per 1M tokens
        },
        "llama3.1-405b-instruct": {
            "input": 0.00532,  # $5.32 per 1M tokens
            "output": 0.016,  # $16.00 per 1M tokens
        },
        # Mistral AI models
        "mistral-7b-instruct": {
            "input": 0.00015,  # $0.15 per 1M tokens
            "output": 0.0002,  # $0.20 per 1M tokens
        },
        "mixtral-8x7b-instruct": {
            "input": 0.00045,  # $0.45 per 1M tokens
            "output": 0.0007,  # $0.70 per 1M tokens
        },
        "mistral-large": {
            "input": 0.008,  # $8.00 per 1M tokens
            "output": 0.024,  # $24.00 per 1M tokens
        },
        "mistral-small": {
            "input": 0.002,  # $2.00 per 1M tokens
            "output": 0.006,  # $6.00 per 1M tokens
        },
    }

    def calculate_cost(
        self, model: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> float:
        """
        Calculate the cost of a Bedrock API call.

        Args:
            model: The model ID used
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

        # Extract the simple model name from Bedrock model IDs
        if "anthropic.claude-3-5-sonnet" in model:
            return "claude-3.5-sonnet"
        elif "anthropic.claude-3-5-haiku" in model:
            return "claude-3.5-haiku"
        elif "anthropic.claude-3-opus" in model:
            return "claude-3-opus"
        elif "anthropic.claude-3-sonnet" in model:
            return "claude-3-sonnet"
        elif "anthropic.claude-3-haiku" in model:
            return "claude-3-haiku"
        elif "anthropic.claude-v2:1" in model:
            return "claude-2.1"
        elif "anthropic.claude-v2" in model:
            return "claude-2.0"
        elif "anthropic.claude-instant" in model:
            return "claude-instant-1.2"
        elif "amazon.titan-text-lite" in model:
            return "titan-text-lite"
        elif "amazon.titan-text-express" in model:
            return "titan-text-express"
        elif "amazon.titan-text-premier" in model:
            return "titan-text-premier"
        elif "amazon.titan-embed-text-v2" in model:
            return "titan-embed-text-v2"
        elif "amazon.titan-embed-text" in model:
            return "titan-embed-text"
        elif "amazon.titan-embed-image" in model:
            return "titan-embed-image"
        elif "amazon.titan-image-generator" in model:
            return "titan-image-generator"
        elif "cohere.command-r-plus" in model:
            return "command-r-plus"
        elif "cohere.command-r" in model:
            return "command-r"
        elif "cohere.command-text" in model:
            return "command-text"
        elif "cohere.command-light-text" in model:
            return "command-light-text"
        elif "cohere.embed-english" in model:
            return "embed-english"
        elif "cohere.embed-multilingual" in model:
            return "embed-multilingual"
        elif "meta.llama3-1-405b-instruct" in model:
            return "llama3.1-405b-instruct"
        elif "meta.llama3-1-70b-instruct" in model:
            return "llama3.1-70b-instruct"
        elif "meta.llama3-1-8b-instruct" in model:
            return "llama3.1-8b-instruct"
        elif "meta.llama3-70b-instruct" in model:
            return "llama3-70b-instruct"
        elif "meta.llama3-8b-instruct" in model:
            return "llama3-8b-instruct"
        elif "meta.llama2-70b-chat" in model:
            return "llama2-70b-chat"
        elif "meta.llama2-13b-chat" in model:
            return "llama2-13b-chat"
        elif "mistral.mistral-large" in model:
            return "mistral-large"
        elif "mistral.mistral-small" in model:
            return "mistral-small"
        elif "mistral.mixtral-8x7b-instruct" in model:
            return "mixtral-8x7b-instruct"
        elif "mistral.mistral-7b-instruct" in model:
            return "mistral-7b-instruct"
        else:
            # Return the model as-is if no mapping found
            return model

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
            "provider": self._extract_provider_from_model(model),
        }

    def _extract_provider_from_model(self, model: str) -> str:
        """Extract the model provider from the full model ID."""
        if "anthropic." in model:
            return "anthropic"
        elif "amazon." in model:
            return "amazon"
        elif "cohere." in model:
            return "cohere"
        elif "meta." in model:
            return "meta"
        elif "mistral." in model:
            return "mistral"
        elif "ai21." in model:
            return "ai21"
        elif "stability." in model:
            return "stability"
        else:
            return "unknown"

    def estimate_cost(
        self, model: str, input_text: str = "", estimated_output_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Estimate cost before making an API call.

        Args:
            model: The model ID
            input_text: Input text to estimate tokens for
            estimated_output_tokens: Estimated output tokens

        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimation (actual tokenization would be more accurate)
        # AWS typically uses ~4 characters per token for English text
        estimated_input_tokens = len(input_text) / 4

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
            "provider": self._extract_provider_from_model(model),
        }
