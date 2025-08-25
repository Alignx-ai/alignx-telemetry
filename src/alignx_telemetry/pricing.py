"""
Cost calculation utility for AI operations.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PricingCalculator:
    """
    Utility class for calculating costs of AI operations.
    """

    def __init__(self):
        self._pricing_data = None
        self._load_pricing_data()

    def _load_pricing_data(self) -> None:
        """Load pricing data from the pricing.json file."""
        try:
            # Get the path to the pricing.json file
            pricing_file = Path(__file__).parent / "pricing.json"

            if not pricing_file.exists():
                logger.warning(f"Pricing file not found at {pricing_file}")
                self._pricing_data = {}
                return

            with open(pricing_file, "r") as f:
                self._pricing_data = json.load(f)

            logger.debug(
                f"Loaded pricing data for {len(self._pricing_data)} categories"
            )

        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            self._pricing_data = {}

    def calculate_chat_cost(
        self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0, **kwargs
    ) -> Optional[float]:
        """
        Calculate cost for chat/completion operations.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-sonnet-20240229")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            **kwargs: Additional parameters (for future extensions)

        Returns:
            Cost in USD, or None if pricing not available
        """
        if not self._pricing_data:
            return None

        chat_pricing = self._pricing_data.get("chat", {})
        model_pricing = chat_pricing.get(model)

        if not model_pricing:
            logger.debug(f"No pricing data available for model: {model}")
            return None

        # Calculate cost per 1000 tokens
        prompt_cost = (prompt_tokens / 1000.0) * model_pricing.get("promptPrice", 0)
        completion_cost = (completion_tokens / 1000.0) * model_pricing.get(
            "completionPrice", 0
        )

        total_cost = prompt_cost + completion_cost

        logger.debug(
            f"Cost calculation for {model}: prompt={prompt_cost:.6f}, completion={completion_cost:.6f}, total={total_cost:.6f}"
        )

        return total_cost

    def calculate_embedding_cost(
        self, model: str, tokens: int = 0, **kwargs
    ) -> Optional[float]:
        """
        Calculate cost for embedding operations.

        Args:
            model: Model name (e.g., "text-embedding-ada-002")
            tokens: Number of tokens
            **kwargs: Additional parameters (for future extensions)

        Returns:
            Cost in USD, or None if pricing not available
        """
        if not self._pricing_data:
            return None

        embedding_pricing = self._pricing_data.get("embeddings", {})
        model_price = embedding_pricing.get(model)

        if model_price is None:
            logger.debug(f"No pricing data available for embedding model: {model}")
            return None

        # Calculate cost per 1000 tokens
        total_cost = (tokens / 1000.0) * model_price

        logger.debug(
            f"Embedding cost calculation for {model}: tokens={tokens}, cost={total_cost:.6f}"
        )

        return total_cost

    def calculate_image_cost(
        self, model: str, size: str = "1024x1024", quality: str = "standard", **kwargs
    ) -> Optional[float]:
        """
        Calculate cost for image generation operations.

        Args:
            model: Model name (e.g., "dall-e-3")
            size: Image size (e.g., "1024x1024")
            quality: Image quality (e.g., "standard", "hd")
            **kwargs: Additional parameters (for future extensions)

        Returns:
            Cost in USD, or None if pricing not available
        """
        if not self._pricing_data:
            return None

        image_pricing = self._pricing_data.get("images", {})
        model_pricing = image_pricing.get(model)

        if not model_pricing:
            logger.debug(f"No pricing data available for image model: {model}")
            return None

        quality_pricing = model_pricing.get(quality, {})
        if not quality_pricing:
            logger.debug(f"No pricing data available for quality: {quality}")
            return None

        size_price = quality_pricing.get(size)
        if size_price is None:
            logger.debug(f"No pricing data available for size: {size}")
            return None

        logger.debug(
            f"Image cost calculation for {model}: size={size}, quality={quality}, cost={size_price}"
        )

        return size_price

    def calculate_audio_cost(
        self, model: str, characters: int = 0, **kwargs
    ) -> Optional[float]:
        """
        Calculate cost for audio operations.

        Args:
            model: Model name (e.g., "tts-1")
            characters: Number of characters
            **kwargs: Additional parameters (for future extensions)

        Returns:
            Cost in USD, or None if pricing not available
        """
        if not self._pricing_data:
            return None

        audio_pricing = self._pricing_data.get("audio", {})
        model_price = audio_pricing.get(model)

        if model_price is None:
            logger.debug(f"No pricing data available for audio model: {model}")
            return None

        # Calculate cost per 1000 characters
        total_cost = (characters / 1000.0) * model_price

        logger.debug(
            f"Audio cost calculation for {model}: characters={characters}, cost={total_cost:.6f}"
        )

        return total_cost

    def calculate_cost(
        self, operation_type: str, model: str, usage_data: Dict[str, Any], **kwargs
    ) -> Optional[float]:
        """
        Generic cost calculation method that routes to appropriate specific method.

        Args:
            operation_type: Type of operation ("chat", "embedding", "image", "audio")
            model: Model name
            usage_data: Dictionary containing usage information
            **kwargs: Additional parameters

        Returns:
            Cost in USD, or None if pricing not available
        """
        if operation_type in ["chat", "completion", "chat_completion"]:
            return self.calculate_chat_cost(
                model=model,
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                **kwargs,
            )
        elif operation_type in ["embedding", "embeddings"]:
            return self.calculate_embedding_cost(
                model=model, tokens=usage_data.get("total_tokens", 0), **kwargs
            )
        elif operation_type in ["image", "image_generation"]:
            return self.calculate_image_cost(
                model=model,
                size=usage_data.get("size", "1024x1024"),
                quality=usage_data.get("quality", "standard"),
                **kwargs,
            )
        elif operation_type in ["audio", "tts", "text_to_speech"]:
            return self.calculate_audio_cost(
                model=model, characters=usage_data.get("characters", 0), **kwargs
            )
        else:
            logger.warning(
                f"Unknown operation type for cost calculation: {operation_type}"
            )
            return None

    def get_available_models(self, operation_type: str) -> list:
        """
        Get list of available models for a given operation type.

        Args:
            operation_type: Type of operation ("chat", "embedding", "image", "audio")

        Returns:
            List of available model names
        """
        if not self._pricing_data:
            return []

        if operation_type == "chat":
            return list(self._pricing_data.get("chat", {}).keys())
        elif operation_type == "embedding":
            return list(self._pricing_data.get("embeddings", {}).keys())
        elif operation_type == "image":
            return list(self._pricing_data.get("images", {}).keys())
        elif operation_type == "audio":
            return list(self._pricing_data.get("audio", {}).keys())
        else:
            return []


# Global instance for easy access
_pricing_calculator = PricingCalculator()


def calculate_cost(
    operation_type: str, model: str, usage_data: Dict[str, Any], **kwargs
) -> Optional[float]:
    """
    Convenience function for calculating costs.

    Args:
        operation_type: Type of operation ("chat", "embedding", "image", "audio")
        model: Model name
        usage_data: Dictionary containing usage information
        **kwargs: Additional parameters

    Returns:
        Cost in USD, or None if pricing not available
    """
    return _pricing_calculator.calculate_cost(
        operation_type, model, usage_data, **kwargs
    )


def get_available_models(operation_type: str) -> list:
    """
    Get list of available models for a given operation type.

    Args:
        operation_type: Type of operation ("chat", "embedding", "image", "audio")

    Returns:
        List of available model names
    """
    return _pricing_calculator.get_available_models(operation_type)


def get_llm_cost(
    provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0, **kwargs
) -> Optional[float]:
    """
    Get LLM cost for a specific provider and model.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        **kwargs: Additional parameters

    Returns:
        Cost in USD, or None if pricing not available
    """
    # Try to get cost for the model directly (most models are provider-agnostic in pricing data)
    cost = _pricing_calculator.calculate_chat_cost(
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        **kwargs,
    )

    if cost is not None:
        return cost

    # If direct model lookup fails, try provider-prefixed model names
    provider_model_variants = [
        f"{provider}/{model}",
        f"{provider}.{model}",
        f"{provider}:{model}",
        f"{provider}-{model}",
    ]

    for variant in provider_model_variants:
        cost = _pricing_calculator.calculate_chat_cost(
            model=variant,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            **kwargs,
        )
        if cost is not None:
            return cost

    return None


def is_cost_tracking_supported(provider: str, model: str) -> bool:
    """
    Check if cost tracking is supported for a provider and model.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name

    Returns:
        True if cost tracking is supported, False otherwise
    """
    if not _pricing_calculator._pricing_data:
        return False

    # Check if we have pricing data for this model
    chat_pricing = _pricing_calculator._pricing_data.get("chat", {})

    # Try direct model lookup
    if model in chat_pricing:
        return True

    # Try provider-prefixed variants
    provider_model_variants = [
        f"{provider}/{model}",
        f"{provider}.{model}",
        f"{provider}:{model}",
        f"{provider}-{model}",
    ]

    for variant in provider_model_variants:
        if variant in chat_pricing:
            return True

    return False
