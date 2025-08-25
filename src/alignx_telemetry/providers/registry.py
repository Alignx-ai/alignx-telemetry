"""Provider instrumentation registry for AlignX.

This module provides centralized registration and management of all
provider instrumentations, inspired by the registry patterns in
OpenLLMetry and Opik.
"""

import logging
import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Any, Callable
from datetime import datetime
from pathlib import Path

from .base import AlignXBaseProviderInstrumentation

# Avoid circular import - use string annotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alignx_telemetry.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    """Detailed capabilities and metadata for a provider."""

    # Basic information
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"

    # Support matrix
    supports_streaming: bool = False
    supports_async: bool = False
    supports_embedding: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_function_calling: bool = False

    # Model information
    supported_models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None

    # Pricing and cost tracking
    supports_cost_tracking: bool = False
    cost_currency: str = "USD"
    cost_per_input_token: Dict[str, float] = field(default_factory=dict)
    cost_per_output_token: Dict[str, float] = field(default_factory=dict)

    # Technical requirements
    required_packages: List[str] = field(default_factory=list)
    optional_packages: List[str] = field(default_factory=list)
    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: List[str] = field(default_factory=list)

    # Documentation
    documentation_url: Optional[str] = None
    setup_instructions: Optional[str] = None

    # Status
    is_available: bool = False
    is_experimental: bool = False
    deprecation_warning: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class ProviderInstrumentationResult:
    """Result of provider instrumentation operation."""

    provider_name: str
    success: bool
    error_message: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    instrumentation_time_ms: float = 0.0
    capabilities: Optional[ProviderCapabilities] = None


class AlignXProviderRegistry:
    """Enhanced central registry for all provider instrumentations.

    This class manages the lifecycle of provider instrumentations and
    provides utilities for bulk operations, discovery, and metadata management.
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._registry: Dict[str, Type[AlignXBaseProviderInstrumentation]] = {}
        self._instrumented_providers: Dict[str, AlignXBaseProviderInstrumentation] = {}
        self._capabilities_cache: Dict[str, ProviderCapabilities] = {}
        self._telemetry_manager: Optional["TelemetryManager"] = None
        self._discovery_hooks: List[Callable] = []

    def register_provider(
        self,
        provider_name: str,
        instrumentation_class: Type[AlignXBaseProviderInstrumentation],
        capabilities: Optional[ProviderCapabilities] = None,
    ) -> None:
        """Register a provider instrumentation class.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            instrumentation_class: Provider instrumentation class
            capabilities: Optional provider capabilities metadata
        """
        if provider_name in self._registry:
            logger.warning(f"Provider {provider_name} already registered, overriding")

        self._registry[provider_name] = instrumentation_class

        # Store capabilities if provided
        if capabilities:
            self._capabilities_cache[provider_name] = capabilities
        else:
            # Try to auto-discover capabilities
            self._discover_provider_capabilities(provider_name, instrumentation_class)

        logger.debug(f"Registered provider: {provider_name}")

    def unregister_provider(self, provider_name: str) -> bool:
        """Unregister a provider instrumentation.

        Args:
            provider_name: Name of the provider to unregister

        Returns:
            True if provider was unregistered, False if not found
        """
        if provider_name not in self._registry:
            return False

        # Uninstrument if currently instrumented
        if provider_name in self._instrumented_providers:
            self.uninstrument_provider(provider_name)

        del self._registry[provider_name]
        logger.debug(f"Unregistered provider: {provider_name}")
        return True

    def get_registered_providers(self) -> List[str]:
        """Get list of all registered provider names.

        Returns:
            List of registered provider names
        """
        return list(self._registry.keys())

    def get_instrumented_providers(self) -> List[str]:
        """Get list of currently instrumented provider names.

        Returns:
            List of instrumented provider names
        """
        return list(self._instrumented_providers.keys())

    def is_provider_registered(self, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is registered
        """
        return provider_name in self._registry

    def is_provider_instrumented(self, provider_name: str) -> bool:
        """Check if a provider is currently instrumented.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is instrumented
        """
        return provider_name in self._instrumented_providers

    def instrument_provider(
        self, provider_name: str, telemetry_manager: "TelemetryManager", **kwargs: Any
    ) -> bool:
        """Instrument a specific provider.

        Args:
            provider_name: Name of the provider to instrument
            telemetry_manager: AlignX telemetry manager instance
            **kwargs: Additional configuration for the provider

        Returns:
            True if instrumentation succeeded, False otherwise
        """
        if provider_name not in self._registry:
            logger.error(f"Provider {provider_name} not registered")
            return False

        if provider_name in self._instrumented_providers:
            logger.debug(f"Provider {provider_name} already instrumented")
            return True

        try:
            # Create provider instrumentation instance
            instrumentation_class = self._registry[provider_name]
            instrumentation = instrumentation_class(telemetry_manager)

            # Apply any provider-specific configuration
            if kwargs:
                self._apply_provider_config(instrumentation, kwargs)

            # Perform instrumentation
            success = instrumentation.instrument()

            if success:
                self._instrumented_providers[provider_name] = instrumentation
                self._telemetry_manager = telemetry_manager
                logger.info(f"Successfully instrumented provider: {provider_name}")
            else:
                logger.error(f"Failed to instrument provider: {provider_name}")

            return success

        except Exception as e:
            logger.error(
                f"Error instrumenting provider {provider_name}: {e}", exc_info=True
            )
            return False

    def uninstrument_provider(self, provider_name: str) -> bool:
        """Remove instrumentation from a specific provider.

        Args:
            provider_name: Name of the provider to uninstrument

        Returns:
            True if uninstrumentation succeeded, False otherwise
        """
        if provider_name not in self._instrumented_providers:
            logger.debug(f"Provider {provider_name} not instrumented")
            return True

        try:
            instrumentation = self._instrumented_providers[provider_name]
            success = instrumentation.uninstrument()

            if success:
                del self._instrumented_providers[provider_name]
                logger.info(f"Successfully uninstrumented provider: {provider_name}")
            else:
                logger.error(f"Failed to uninstrument provider: {provider_name}")

            return success

        except Exception as e:
            logger.error(
                f"Error uninstrumenting provider {provider_name}: {e}", exc_info=True
            )
            return False

    def instrument_all_providers(
        self, telemetry_manager: "TelemetryManager", **kwargs: Any
    ) -> Dict[str, bool]:
        """Instrument all registered providers.

        Args:
            telemetry_manager: AlignX telemetry manager instance
            **kwargs: Global configuration for all providers

        Returns:
            Dictionary mapping provider names to instrumentation success status
        """
        results = {}

        for provider_name in self._registry:
            try:
                # Extract provider-specific config if available
                provider_config = kwargs.get(f"{provider_name}_config", {})
                global_config = {
                    k: v for k, v in kwargs.items() if not k.endswith("_config")
                }
                final_config = {**global_config, **provider_config}

                success = self.instrument_provider(
                    provider_name, telemetry_manager, **final_config
                )
                results[provider_name] = success

            except Exception as e:
                logger.error(f"Error instrumenting {provider_name}: {e}")
                results[provider_name] = False

        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Instrumented {successful}/{total} providers successfully")

        return results

    def uninstrument_all_providers(self) -> Dict[str, bool]:
        """Remove instrumentation from all providers.

        Returns:
            Dictionary mapping provider names to uninstrumentation success status
        """
        results = {}

        # Create a copy since we'll be modifying the dict
        providers_to_uninstrument = list(self._instrumented_providers.keys())

        for provider_name in providers_to_uninstrument:
            results[provider_name] = self.uninstrument_provider(provider_name)

        logger.info("Uninstrumented all providers")
        return results

    def get_provider_instrumentation(
        self, provider_name: str
    ) -> Optional[AlignXBaseProviderInstrumentation]:
        """Get the instrumentation instance for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instrumentation instance or None if not instrumented
        """
        return self._instrumented_providers.get(provider_name)

    def get_instrumentation_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all providers.

        Returns:
            Dictionary with detailed provider status information
        """
        return {
            "registered_providers": self.get_registered_providers(),
            "instrumented_providers": self.get_instrumented_providers(),
            "total_registered": len(self._registry),
            "total_instrumented": len(self._instrumented_providers),
            "instrumentation_rate": (
                len(self._instrumented_providers) / len(self._registry)
                if self._registry
                else 0.0
            ),
        }

    def _apply_provider_config(
        self, instrumentation: AlignXBaseProviderInstrumentation, config: Dict[str, Any]
    ) -> None:
        """Apply configuration to a provider instrumentation.

        Args:
            instrumentation: Provider instrumentation instance
            config: Configuration dictionary
        """
        # Apply configuration if the instrumentation supports it
        if hasattr(instrumentation, "configure"):
            instrumentation.configure(config)
        elif hasattr(instrumentation, "config"):
            instrumentation.config.update(config)

    def discover_available_providers(self) -> List[str]:
        """Dynamically discover available provider instrumentations.

        This method scans the providers package for available instrumentations
        and returns a list of provider names that could be instrumented.

        Returns:
            List of available provider names
        """
        available_providers = []

        # Scan the providers package directory
        try:
            providers_dir = Path(__file__).parent
            for item in providers_dir.iterdir():
                if (
                    item.is_dir()
                    and not item.name.startswith("_")
                    and not item.name.startswith(".")
                ):
                    # Check if it's a valid provider directory
                    init_file = item / "__init__.py"
                    instrumentation_file = item / "instrumentation.py"

                    if init_file.exists() and instrumentation_file.exists():
                        provider_name = item.name
                        available_providers.append(provider_name)
                        logger.debug(f"Discovered provider: {provider_name}")
        except Exception as e:
            logger.warning(f"Provider discovery failed: {e}")

        return sorted(available_providers)

    def get_provider_capabilities(
        self, provider_name: str
    ) -> Optional[ProviderCapabilities]:
        """Get detailed capabilities for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderCapabilities if available, None otherwise
        """
        # Check cache first
        if provider_name in self._capabilities_cache:
            return self._capabilities_cache[provider_name]

        # Try to discover capabilities if provider is registered
        if provider_name in self._registry:
            self._discover_provider_capabilities(
                provider_name, self._registry[provider_name]
            )
            return self._capabilities_cache.get(provider_name)

        return None

    def _discover_provider_capabilities(
        self,
        provider_name: str,
        instrumentation_class: Type[AlignXBaseProviderInstrumentation],
    ) -> None:
        """Auto-discover capabilities for a provider.

        Args:
            provider_name: Name of the provider
            instrumentation_class: Provider instrumentation class
        """
        try:
            # Create basic capabilities from known information
            capabilities = ProviderCapabilities(
                name=provider_name,
                display_name=provider_name.title(),
                description=f"{provider_name.title()} LLM provider instrumentation",
            )

            # Try to determine availability by checking for required packages
            if provider_name == "openai":
                capabilities.required_packages = ["openai"]
                capabilities.required_env_vars = ["OPENAI_API_KEY"]
                capabilities.supports_streaming = True
                capabilities.supports_async = True
                capabilities.supports_embedding = True
                capabilities.supports_vision = True
                capabilities.supports_audio = True
                capabilities.supports_function_calling = True
                capabilities.supports_cost_tracking = True
                capabilities.documentation_url = "https://platform.openai.com/docs"

            elif provider_name == "anthropic":
                capabilities.required_packages = ["anthropic"]
                capabilities.required_env_vars = ["ANTHROPIC_API_KEY"]
                capabilities.supports_streaming = True
                capabilities.supports_async = True
                capabilities.supports_vision = True
                capabilities.supports_function_calling = True
                capabilities.supports_cost_tracking = True
                capabilities.documentation_url = "https://docs.anthropic.com"

            elif provider_name == "google_genai":
                capabilities.required_packages = ["google-generativeai"]
                capabilities.required_env_vars = ["GOOGLE_API_KEY"]
                capabilities.supports_streaming = True
                capabilities.supports_async = True
                capabilities.supports_vision = True
                capabilities.supports_function_calling = True
                capabilities.supports_cost_tracking = True
                capabilities.documentation_url = "https://ai.google.dev/docs"

            elif provider_name == "bedrock":
                capabilities.required_packages = ["boto3"]
                capabilities.required_env_vars = [
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                ]
                capabilities.supports_streaming = True
                capabilities.supports_async = True
                capabilities.supports_cost_tracking = True
                capabilities.documentation_url = "https://docs.aws.amazon.com/bedrock/"

            # Check if required packages are available
            capabilities.is_available = self._check_package_availability(
                capabilities.required_packages
            )

            # Store in cache
            self._capabilities_cache[provider_name] = capabilities

        except Exception as e:
            logger.debug(f"Failed to discover capabilities for {provider_name}: {e}")

    def _check_package_availability(self, packages: List[str]) -> bool:
        """Check if required packages are available.

        Args:
            packages: List of package names to check

        Returns:
            True if all packages are available, False otherwise
        """
        for package in packages:
            try:
                importlib.import_module(package)
            except ImportError:
                return False
        return True

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including capabilities and discovery info.

        Returns:
            Dictionary with detailed registry status
        """
        available_providers = self.discover_available_providers()

        return {
            "registry_info": {
                "total_registered": len(self._registry),
                "total_instrumented": len(self._instrumented_providers),
                "total_available": len(available_providers),
                "has_telemetry_manager": self._telemetry_manager is not None,
            },
            "registered_providers": list(self._registry.keys()),
            "instrumented_providers": list(self._instrumented_providers.keys()),
            "available_providers": available_providers,
            "provider_capabilities": {
                name: capabilities.__dict__ if capabilities else None
                for name, capabilities in self._capabilities_cache.items()
            },
            "instrumentation_status": self.get_instrumentation_status(),
        }


# Global provider registry instance
_global_registry = AlignXProviderRegistry()


# Public registry management functions
def register_provider(
    provider_name: str, instrumentation_class: Type[AlignXBaseProviderInstrumentation]
) -> None:
    """Register a provider instrumentation class globally.

    Args:
        provider_name: Name of the provider
        instrumentation_class: Provider instrumentation class
    """
    _global_registry.register_provider(provider_name, instrumentation_class)


def unregister_provider(provider_name: str) -> bool:
    """Unregister a provider instrumentation globally.

    Args:
        provider_name: Name of the provider to unregister

    Returns:
        True if provider was unregistered, False if not found
    """
    return _global_registry.unregister_provider(provider_name)


def instrument_provider(
    provider_name: str, telemetry_manager: "TelemetryManager", **kwargs: Any
) -> bool:
    """Instrument a specific provider globally.

    Args:
        provider_name: Name of the provider to instrument
        telemetry_manager: AlignX telemetry manager instance
        **kwargs: Additional configuration for the provider

    Returns:
        True if instrumentation succeeded, False otherwise
    """
    return _global_registry.instrument_provider(
        provider_name, telemetry_manager, **kwargs
    )


def uninstrument_provider(provider_name: str) -> bool:
    """Remove instrumentation from a specific provider globally.

    Args:
        provider_name: Name of the provider to uninstrument

    Returns:
        True if uninstrumentation succeeded, False otherwise
    """
    return _global_registry.uninstrument_provider(provider_name)


def instrument_all_providers(
    telemetry_manager: "TelemetryManager", **kwargs: Any
) -> Dict[str, bool]:
    """Instrument all registered providers globally.

    Args:
        telemetry_manager: AlignX telemetry manager instance
        **kwargs: Global configuration for all providers

    Returns:
        Dictionary mapping provider names to instrumentation success status
    """
    return _global_registry.instrument_all_providers(telemetry_manager, **kwargs)


def uninstrument_all_providers() -> Dict[str, bool]:
    """Remove instrumentation from all providers globally.

    Returns:
        Dictionary mapping provider names to uninstrumentation success status
    """
    return _global_registry.uninstrument_all_providers()


def get_registered_providers() -> List[str]:
    """Get list of all registered provider names globally.

    Returns:
        List of registered provider names
    """
    return _global_registry.get_registered_providers()


def get_instrumented_providers() -> List[str]:
    """Get list of currently instrumented provider names globally.

    Returns:
        List of instrumented provider names
    """
    return _global_registry.get_instrumented_providers()


def get_instrumentation_status() -> Dict[str, Any]:
    """Get comprehensive status of all providers globally.

    Returns:
        Dictionary with detailed provider status information
    """
    return _global_registry.get_instrumentation_status()


def discover_available_providers() -> List[str]:
    """Discover available provider instrumentations globally.

    Returns:
        List of available provider names
    """
    return _global_registry.discover_available_providers()


def get_provider_capabilities(provider_name: str) -> Optional[ProviderCapabilities]:
    """Get capabilities for a specific provider globally.

    Args:
        provider_name: Name of the provider

    Returns:
        ProviderCapabilities if available, None otherwise
    """
    return _global_registry.get_provider_capabilities(provider_name)


def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive status including discovery and capabilities globally.

    Returns:
        Dictionary with detailed registry status
    """
    return _global_registry.get_comprehensive_status()


def list_providers() -> List[str]:
    """Alias for get_registered_providers for backward compatibility.

    Returns:
        List of registered provider names
    """
    return get_registered_providers()


# Provider instrumentations registry (will be populated by provider modules)
ALIGNX_PROVIDER_INSTRUMENTATIONS: Dict[str, Type[AlignXBaseProviderInstrumentation]] = (
    {}
)


def _auto_register_providers() -> None:
    """Automatically register all available provider instrumentations.

    This function attempts to import and register all provider instrumentations
    that are available in the providers package.
    """
    # Register OpenAI provider
    try:
        from .openai import AlignXOpenAIInstrumentation

        register_provider("openai", AlignXOpenAIInstrumentation)
        ALIGNX_PROVIDER_INSTRUMENTATIONS["openai"] = AlignXOpenAIInstrumentation
        logger.debug("Auto-registered provider: openai")
    except ImportError:
        logger.debug("OpenAI provider module not available")
    except Exception as e:
        logger.error(f"Error auto-registering OpenAI provider: {e}")

    # Register Anthropic provider
    try:
        from .anthropic import AlignXAnthropicInstrumentation

        register_provider("anthropic", AlignXAnthropicInstrumentation)
        ALIGNX_PROVIDER_INSTRUMENTATIONS["anthropic"] = AlignXAnthropicInstrumentation
        logger.debug("Auto-registered provider: anthropic")
    except ImportError:
        logger.debug("Anthropic provider module not available")
    except Exception as e:
        logger.error(f"Error auto-registering Anthropic provider: {e}")

    # Register Google GenAI provider
    try:
        from .google_genai import AlignXGoogleGenAIInstrumentation

        register_provider("google_genai", AlignXGoogleGenAIInstrumentation)
        ALIGNX_PROVIDER_INSTRUMENTATIONS["google_genai"] = (
            AlignXGoogleGenAIInstrumentation
        )
        logger.debug("Auto-registered provider: google_genai")
    except ImportError:
        logger.debug("Google GenAI provider module not available")
    except Exception as e:
        logger.error(f"Error auto-registering Google GenAI provider: {e}")

    # Register Bedrock provider
    try:
        from .bedrock import AlignXBedrockInstrumentation

        register_provider("bedrock", AlignXBedrockInstrumentation)
        ALIGNX_PROVIDER_INSTRUMENTATIONS["bedrock"] = AlignXBedrockInstrumentation
        logger.debug("Auto-registered provider: bedrock")
    except ImportError:
        logger.debug("Bedrock provider module not available")
    except Exception as e:
        logger.error(f"Error auto-registering Bedrock provider: {e}")

    # Future providers will be added here dynamically
    future_provider_modules = [
        "cohere",
        "groq",
        "together",
        "ollama",
    ]

    for provider_module in future_provider_modules:
        try:
            # Dynamically import provider module
            module = __import__(
                f"alignx_telemetry.providers.{provider_module}",
                fromlist=["instrumentation"],
            )

            # Look for instrumentation class
            instrumentation_class_name = (
                f"AlignX{provider_module.title()}Instrumentation"
            )
            if hasattr(module, instrumentation_class_name):
                instrumentation_class = getattr(module, instrumentation_class_name)
                register_provider(provider_module, instrumentation_class)
                ALIGNX_PROVIDER_INSTRUMENTATIONS[provider_module] = (
                    instrumentation_class
                )
                logger.debug(f"Auto-registered provider: {provider_module}")

        except ImportError:
            # Provider module not available (missing dependencies, etc.)
            logger.debug(f"Provider module {provider_module} not available")
        except Exception as e:
            logger.error(f"Error auto-registering provider {provider_module}: {e}")


# Auto-register providers on module import
try:
    _auto_register_providers()
except Exception as e:
    logger.debug(f"Error during auto-registration: {e}")
