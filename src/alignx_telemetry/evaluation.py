"""AlignX SDK Evaluation Module for Agent Response Evaluation"""

import asyncio
import functools
import inspect
import os
import threading
import httpx
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Thread-safe lock for lazy initialization
_initialization_lock = threading.Lock()


class PolicyTitle(Enum):
    FACTUALITY = "Factuality"
    CONTEXT = "Context"
    RELEVANCE = "Relevance"
    SAFETY = "Safety"
    SENTIMENT = "Sentiment"
    TOXICITY = "Toxicity"
    PRIVACY = "Privacy"
    SENSITIVITY = "Sensitivity"
    ACCESSIBILITY = "Accessibility"
    ENGAGEMENT = "Engagement"
    TRANSPARENCY = "Transparency"
    ERROR_DETECTION = "Error Detection"


@dataclass
class EvaluationPolicy:
    """Represents an evaluation policy with its configuration"""

    title: PolicyTitle
    description: Optional[str] = None
    threshold: float = 0.7


@dataclass
class EvaluationConfig:
    reference: Optional[str]
    reference_id: Optional[str]
    policies: List[EvaluationPolicy]


class EvaluationConfigurationError(Exception):
    """Raised when evaluation configuration is invalid or missing"""

    pass


class AsyncEvaluationClient:
    """Client for making async evaluation requests to AlignX"""

    def __init__(
        self,
        license_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        """
        Initialize the async evaluation client

        Args:
            license_key: AlignX self-contained license key (can be set via ALIGNX_LICENSE_KEY env var)
                        Format: alignx_[base64(org_id:random)]_[checksum]
            api_base_url: Base URL for AlignX API (can be set via ALIGNX_API_BASE_URL env var)
        """
        # Environment-first configuration with fallbacks
        self.license_key = self._resolve_license_key(license_key)
        self.api_base_url = self._resolve_api_base_url(api_base_url)
        self.timeout = httpx.Timeout(
            connect=15.0,
            read=20.0,
            write=10.0,
            pool=2.0,
        )

        # Validate configuration
        self._validate_configuration()

    def _resolve_license_key(self, provided_key: Optional[str]) -> str:
        """Resolve license key from parameter or environment with validation"""
        key = provided_key or os.getenv("ALIGNX_LICENSE_KEY")

        if not key:
            raise EvaluationConfigurationError(
                "License key must be provided either as parameter or via ALIGNX_LICENSE_KEY environment variable. "
                "Get your license key from the AlignX dashboard: https://app.alignx.ai/settings/api-keys"
            )

        # Validate license key format
        if not self._validate_license_key_format(key):
            raise EvaluationConfigurationError(
                f"Invalid license key format. Expected format: alignx_[encoded_data]_[checksum]. "
                f"Received: {key[:10]}... (truncated for security)"
            )

        return key

    def _resolve_api_base_url(self, provided_url: Optional[str]) -> str:
        """Resolve API base URL from parameter or environment with validation"""
        url = (
            provided_url or os.getenv("ALIGNX_API_BASE_URL") or "https://app.alignx.ai"
        )

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise EvaluationConfigurationError(
                f"Invalid API base URL format. Must start with http:// or https://. "
                f"Received: {url}"
            )

        return url.rstrip("/")  # Remove trailing slash

    def _validate_configuration(self) -> None:
        """Validate the complete configuration"""
        logger.debug(
            f"Initialized AlignX evaluation client with API base: {self.api_base_url}"
        )

    def _validate_license_key_format(self, license_key: str) -> bool:
        """Validate the license key follows the expected format"""
        try:
            parts = license_key.split("_")
            return len(parts) == 3 and parts[0] == "alignx"
        except Exception:
            return False

    async def evaluate_async(
        self,
        candidate: str,
        config: EvaluationConfig,
        timeout: Optional[httpx.Timeout] = None,
    ) -> Dict[str, Any]:
        """
        Submit async evaluation request to AlignX

        Args:
            candidate: The agent response text to evaluate
            config: Evaluation configuration with policies and reference

        Returns:
            Dictionary with evaluation status and message
        """
        try:
            async with httpx.AsyncClient(
                timeout=timeout if timeout else self.timeout
            ) as client:
                headers = {
                    "X-License-Key": self.license_key,
                    "Content-Type": "application/json",
                }

                payload = {
                    "value": candidate,
                    "reference": config.reference,
                    "reference_id": config.reference_id,
                    "policies": [
                        {
                            "title": (policy.title.value),
                            "threshold": policy.threshold,
                        }
                        for policy in config.policies
                    ],
                }

                # Remove None values
                payload = {k: v for k, v in payload.items() if v is not None}

                response = await client.post(
                    f"{self.api_base_url}/api/v1.0/evals/evaluate/async/sdk",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(
                        f"Evaluation request failed: {response.status_code} - {response.text}"
                    )
                    return {
                        "status": "error",
                        "message": f"Request failed with status {response.status_code}",
                    }

        except Exception as e:
            logger.error(f"Error during async evaluation: {str(e)}")
            return {"status": "error", "message": f"Request failed: {str(e)}"}


class AsyncEvaluator:
    """Main evaluator class for managing async evaluations"""

    def __init__(
        self,
        license_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        """Initialize the async evaluator"""
        self.client = AsyncEvaluationClient(license_key, api_base_url)

    def evaluate_agent_response(
        self,
        policies: List[EvaluationPolicy],
        reference: Optional[str] = None,
        reference_id: Optional[str] = None,
        extract_response: Optional[Callable] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        """
        Decorator to automatically evaluate agent responses

        Args:
            policies: List of policy names or EvaluationPolicy objects
            reference: Reference text for evaluation
            reference_id: Reference ID for RAG-based evaluation
            extract_response: Optional function to extract response text from function result
            timeout: httpx timeout configuration
        Usage:
            @evaluator.evaluate_agent_response(
                policies=["coherence", "relevance"],
                reference="Expected response content"
            )
            async def my_agent_function(prompt: str) -> str:
                # Your agent logic here
                return agent_response
        """

        def decorator(func: Callable) -> Callable:
            # Convert string policies to EvaluationPolicy objects
            policy_objects = []
            for policy in policies:
                if isinstance(policy, EvaluationPolicy):
                    policy_objects.append(policy)
                else:
                    raise ValueError(f"Invalid policy type: {type(policy)}")

            config = EvaluationConfig(
                policies=policy_objects,
                reference=reference,
                reference_id=reference_id,
            )

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Execute the original function
                    result = await func(*args, **kwargs)

                    # Extract response text
                    if extract_response:
                        candidate_text = extract_response(result)
                    elif isinstance(result, str):
                        candidate_text = result
                    elif hasattr(result, "text"):
                        candidate_text = result.text
                    elif hasattr(result, "content"):
                        candidate_text = result.content
                    else:
                        logger.warning(
                            f"Could not extract text from result type: {type(result)}"
                        )
                        raise ValueError(
                            f"Could not extract llm response from result: {result}"
                        )

                    # Submit async evaluation
                    try:
                        eval_result = await self.client.evaluate_async(
                            candidate_text, config
                        )
                        logger.info(
                            f"Evaluation submitted: {eval_result.get('status', 'unknown')}"
                        )

                        # Optionally attach evaluation info to result
                        if hasattr(result, "__dict__"):
                            logger.info(f"Evaluation result status: {eval_result}")

                    except Exception as e:
                        logger.error(f"Failed to submit evaluation: {str(e)}")

                    return result

                return async_wrapper

            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # Execute the original function
                    result = func(*args, **kwargs)

                    # Extract response text
                    if extract_response:
                        candidate_text = extract_response(result)
                    elif isinstance(result, str):
                        candidate_text = result
                    elif hasattr(result, "text"):
                        candidate_text = result.text
                    elif hasattr(result, "content"):
                        candidate_text = result.content
                    else:
                        logger.warning(
                            f"Could not extract text from result type: {type(result)}"
                        )
                        candidate_text = str(result)

                    # Submit async evaluation in background
                    try:
                        # Create new event loop if none exists
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        # Submit evaluation
                        task = loop.create_task(
                            self.client.evaluate_async(
                                candidate_text, config, timeout=timeout
                            )
                        )
                        logger.info("Evaluation submitted in background")

                    except Exception as e:
                        logger.error(f"Failed to submit evaluation: {str(e)}")

                    return result

                return sync_wrapper

        return decorator


# Global evaluator instance with lazy initialization
_global_evaluator: Optional[AsyncEvaluator] = None
_global_config: Dict[str, Any] = {}


def _ensure_evaluator_initialized() -> AsyncEvaluator:
    """
    Ensure the global evaluator is initialized using environment variables.
    This follows the lazy initialization pattern used by OpenAI and other SaaS providers.
    """
    global _global_evaluator

    if _global_evaluator is not None:
        return _global_evaluator

    with _initialization_lock:
        # Double-check pattern to prevent race conditions
        if _global_evaluator is not None:
            return _global_evaluator

        try:
            # Initialize with environment variables automatically
            _global_evaluator = AsyncEvaluator(
                license_key=_global_config.get("license_key"),
                api_base_url=_global_config.get("api_base_url"),
            )
            logger.debug("AlignX evaluator auto-initialized from environment variables")
            return _global_evaluator

        except EvaluationConfigurationError as e:
            logger.error(f"Failed to auto-initialize AlignX evaluator: {e}")
            raise e


def configure_evaluation(
    license_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> AsyncEvaluator:
    """
    Configure the global evaluator instance with explicit parameters.

    This is the modern equivalent of init_evaluator() but with better naming
    and environment variable integration.

    Args:
        license_key: AlignX self-contained license key (format: alignx_[encoded_data]_[checksum])
                    If not provided, will check ALIGNX_LICENSE_KEY environment variable
        api_base_url: API base URL. If not provided, will check ALIGNX_API_BASE_URL environment variable

    Returns:
        AsyncEvaluator instance

    Examples:
        # Configure explicitly
        configure_evaluation(license_key="alignx_abc123_xyz789")

        # Or set environment variables and configure without parameters
        # export ALIGNX_LICENSE_KEY="alignx_abc123_xyz789"
        configure_evaluation()
    """
    global _global_evaluator, _global_config

    # Store configuration for lazy initialization
    if license_key is not None:
        _global_config["license_key"] = license_key
    if api_base_url is not None:
        _global_config["api_base_url"] = api_base_url

    # Force re-initialization with new config
    _global_evaluator = None

    return _ensure_evaluator_initialized()


def get_evaluator() -> AsyncEvaluator:
    """
    Get the global evaluator instance, auto-initializing if needed.

    This function now auto-initializes using environment variables if the
    evaluator hasn't been explicitly configured, following the pattern
    used by OpenAI SDK and other modern SaaS providers.

    Returns:
        AsyncEvaluator instance

    Raises:
        EvaluationConfigurationError: If configuration is invalid or missing

    Examples:
        # With environment variables set
        evaluator = get_evaluator()  # Auto-initializes from env vars

        # Or configure first, then use
        configure_evaluation(license_key="alignx_abc123_xyz789")
        evaluator = get_evaluator()
    """
    return _ensure_evaluator_initialized()


def evaluate_agent_response(
    policies: List[EvaluationPolicy],
    reference: Optional[str] = None,
    reference_id: Optional[str] = None,
    timeout: httpx.Timeout = httpx.Timeout(
        connect=15.0,
        read=20.0,
        write=10.0,
        pool=2.0,
    ),
    extract_response: Optional[Callable] = None,
):
    """
    Convenience decorator that uses the global evaluator with auto-initialization.

    This function now automatically initializes the evaluator from environment
    variables if it hasn't been configured yet, removing the need for explicit
    init_evaluator() calls.

    Args:
        policies: List of policy names or EvaluationPolicy objects
        reference: Reference text for evaluation
        reference_id: Reference ID for RAG-based evaluation
        timeout: httpx timeout configuration
        extract_response: Optional function to extract response text from function result

    Usage:
        # Set environment variables (recommended)
        # export ALIGNX_LICENSE_KEY="alignx_abc123_xyz789"
        # export ALIGNX_API_BASE_URL="https://api.alignx.ai"  # optional

        # Use the decorator directly - auto-initializes from env vars
        @evaluate_agent_response(
            policies=["coherence", "relevance"],
            reference="Expected response"
        )
        async def my_agent(prompt: str) -> str:
            return "agent response"

        # Or configure programmatically first
        configure_evaluation(license_key="alignx_abc123_xyz789")

        @evaluate_agent_response(policies=["coherence"])
        async def my_agent(prompt: str) -> str:
            return "agent response"
    """
    evaluator = get_evaluator()  # Auto-initializes if needed
    return evaluator.evaluate_agent_response(
        policies=policies,
        reference=reference,
        reference_id=reference_id,
        extract_response=extract_response,
        timeout=timeout,
    )
