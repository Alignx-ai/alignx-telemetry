"""Core implementation of the OpenTelemetry instrumentation system."""

import importlib
import os
import logging
import warnings
from typing import Dict, List, Any, Optional


from alignx_telemetry.strategy import get_instrumentation_strategy

# Logger for instrumentation module
logger = logging.getLogger(__name__)


# Dictionary mapping library names to their OpenTelemetry instrumentor package name
INSTRUMENTOR_MAPPING = {
    # Database libraries
    "psycopg2": "opentelemetry.instrumentation.psycopg2",
    "pymongo": "opentelemetry.instrumentation.pymongo",
    "redis": "opentelemetry.instrumentation.redis",
    "sqlite3": "opentelemetry.instrumentation.sqlite3",
    "sqlalchemy": "opentelemetry.instrumentation.sqlalchemy",
    "pymysql": "opentelemetry.instrumentation.pymysql",
    "asyncpg": "opentelemetry.instrumentation.asyncpg",
    "boto3sqs": "opentelemetry.instrumentation.boto3sqs",
    "botocore": "opentelemetry.instrumentation.botocore",
    "celery": "opentelemetry.instrumentation.celery",
    "elasticsearch": "opentelemetry.instrumentation.elasticsearch",
    "kafka": "opentelemetry.instrumentation.kafka",
    "pika": "opentelemetry.instrumentation.pika",
    # Web framework server libraries
    "django": "opentelemetry.instrumentation.django",
    "falcon": "opentelemetry.instrumentation.falcon",
    "fastapi": "opentelemetry.instrumentation.fastapi",
    "flask": "opentelemetry.instrumentation.flask",
    "pyramid": "opentelemetry.instrumentation.pyramid",
    # HTTP/transport server libraries
    "aiohttp_server": "opentelemetry.instrumentation.aiohttp_server",
    "grpc_aioserver": "opentelemetry.instrumentation.grpc_aioserver",
    "grpc_server": "opentelemetry.instrumentation.grpc_server",
    "starlette": "opentelemetry.instrumentation.starlette",
    "tornado": "opentelemetry.instrumentation.tornado",
    # HTTP/transport client libraries
    "aiohttp_client": "opentelemetry.instrumentation.aiohttp_client",
    "grpc_client": "opentelemetry.instrumentation.grpc_client",
    "grpc_aioclient": "opentelemetry.instrumentation.grpc_aioclient",
    "httpx": "opentelemetry.instrumentation.httpx",
    "requests": "opentelemetry.instrumentation.requests",
    "urllib": "opentelemetry.instrumentation.urllib",
    "urllib3": "opentelemetry.instrumentation.urllib3",
    # ========================================
    # Vector Databases
    # ========================================
    "chromadb": "opentelemetry.instrumentation.chromadb",
    "lancedb": "opentelemetry.instrumentation.lancedb",
    "marqo": "opentelemetry.instrumentation.marqo",
    "milvus": "opentelemetry.instrumentation.milvus",
    "pinecone": "opentelemetry.instrumentation.pinecone",
    "qdrant": "opentelemetry.instrumentation.qdrant",
    "weaviate": "opentelemetry.instrumentation.weaviate",
    # ========================================
    # AI Frameworks
    # ========================================
    "crewai": "opentelemetry.instrumentation.crewai",
    "haystack": "opentelemetry.instrumentation.haystack",
    "llamaindex": "opentelemetry.instrumentation.llamaindex",
    "litellm": "opentelemetry.instrumentation.litellm",
    "mem0": "opentelemetry.instrumentation.mem0",
    "autogen": "opentelemetry.instrumentation.autogen",
    "embedchain": "opentelemetry.instrumentation.embedchain",
    "guardrails": "opentelemetry.instrumentation.guardrails",
    "phidata": "opentelemetry.instrumentation.phidata",
    # Traditional libraries
    "logging": "opentelemetry.instrumentation.logging",
    "jinja2": "opentelemetry.instrumentation.jinja2",
    "system_metrics": "opentelemetry.instrumentation.system_metrics",
}


# ========================================
# DEPRECATED LLM Provider Instrumentors
# ========================================
# These LLM provider instrumentors are deprecated in favor of the new AlignX provider system.
# The new provider system offers:
# - Enhanced metrics collection with industry-standard observability
# - Framework-agnostic instrumentation
# - Zero duplicate spans
# - Advanced cost tracking and optimization
# - Streaming performance metrics
#
# Migration Guide:
# Old: instrument_specific("openai")
# New: Use alignx_telemetry.providers.instrument_all_providers() or instrument_provider("openai")
DEPRECATED_LLM_INSTRUMENTORS = {
    "alephalpha": "opentelemetry.instrumentation.alephalpha",
    "anthropic": "opentelemetry.instrumentation.anthropic",
    "bedrock": "opentelemetry.instrumentation.bedrock",
    "cohere": "opentelemetry.instrumentation.cohere",
    "google_generativeai": "opentelemetry.instrumentation.google_generativeai",
    "groq": "opentelemetry.instrumentation.groq",
    "mistralai": "opentelemetry.instrumentation.mistralai",
    "ollama": "opentelemetry.instrumentation.ollama",
    "openai": "opentelemetry.instrumentation.openai",
    "replicate": "opentelemetry.instrumentation.replicate",
    "sagemaker": "opentelemetry.instrumentation.sagemaker",
    "together": "opentelemetry.instrumentation.together",
    "transformers": "opentelemetry.instrumentation.transformers",
    "vertexai": "opentelemetry.instrumentation.vertexai",
    "watsonx": "opentelemetry.instrumentation.watsonx",
}

DEPRECATED_LLM_CLASS_MAPPING = {
    "alephalpha": "AlephAlphaInstrumentor",
    "anthropic": "AnthropicInstrumentor",
    "bedrock": "BedrockInstrumentor",
    "cohere": "CohereInstrumentor",
    "google_generativeai": "GoogleGenerativeAiInstrumentor",
    "groq": "GroqInstrumentor",
    "mistralai": "MistralAiInstrumentor",
    "ollama": "OllamaInstrumentor",
    "openai": "OpenAIInstrumentor",
    "replicate": "ReplicateInstrumentor",
    "sagemaker": "SageMakerInstrumentor",
    "together": "TogetherAiInstrumentor",
    "transformers": "TransformersInstrumentor",
    "vertexai": "VertexAIInstrumentor",
    "watsonx": "WatsonxInstrumentor",
}

# Map of library names to their instrumentor class names
INSTRUMENTOR_CLASS_MAPPING = {
    # Database libraries
    "psycopg2": "Psycopg2Instrumentor",
    "pymongo": "PymongoInstrumentor",
    "redis": "RedisInstrumentor",
    "sqlite3": "SQLite3Instrumentor",
    "sqlalchemy": "SQLAlchemyInstrumentor",
    "pymysql": "PyMySQLInstrumentor",
    "asyncpg": "AsyncPGInstrumentor",
    "boto3sqs": "Boto3SQSInstrumentor",
    "botocore": "BotocoreInstrumentor",
    "celery": "CeleryInstrumentor",
    "elasticsearch": "ElasticsearchInstrumentor",
    "kafka": "KafkaInstrumentor",
    "pika": "PikaInstrumentor",
    # Web framework server libraries
    "django": "DjangoInstrumentor",
    "falcon": "FalconInstrumentor",
    "fastapi": "FastAPIInstrumentor",
    "flask": "FlaskInstrumentor",
    "pyramid": "PyramidInstrumentor",
    # HTTP/transport server libraries
    "aiohttp_server": "AioHttpServerInstrumentor",
    "grpc_aioserver": "GrpcAioServerInstrumentor",
    "grpc_server": "GrpcServerInstrumentor",
    "starlette": "StarletteInstrumentor",
    "tornado": "TornadoInstrumentor",
    # HTTP/transport client libraries
    "aiohttp_client": "AioHttpClientInstrumentor",
    "grpc_client": "GrpcClientInstrumentor",
    "grpc_aioclient": "GrpcAioClientInstrumentor",
    "httpx": "HttpxInstrumentor",
    "requests": "RequestsInstrumentor",
    "urllib": "UrllibInstrumentor",
    "urllib3": "Urllib3Instrumentor",
    # ========================================
    # Vector Databases
    # ========================================
    "chromadb": "ChromaInstrumentor",
    "lancedb": "LanceInstrumentor",
    "marqo": "MarqoInstrumentor",
    "milvus": "MilvusInstrumentor",
    "pinecone": "PineconeInstrumentor",
    "qdrant": "QdrantInstrumentor",
    "weaviate": "WeaviateInstrumentor",
    # ========================================
    # AI Frameworks
    # ========================================
    "crewai": "CrewAIInstrumentor",
    "haystack": "HaystackInstrumentor",
    "llamaindex": "LlamaIndexInstrumentor",
    "litellm": "LiteLLMInstrumentor",
    "mem0": "Mem0Instrumentor",
    "autogen": "AutoGenInstrumentor",
    "embedchain": "EmbedChainInstrumentor",
    "guardrails": "GuardrailsInstrumentor",
    "phidata": "PhidataInstrumentor",
    # Traditional libraries
    "logging": "LoggingInstrumentor",
    "jinja2": "Jinja2Instrumentor",
    "system_metrics": "SystemMetricsInstrumentor",
}


def _get_env_var_name(library_name: str) -> str:
    """Generate environment variable name for library configuration."""
    return f"ALIGNX_INSTRUMENT_{library_name.upper()}"


def _get_config_from_env(library_name: str) -> Dict[str, Any]:
    """Get configuration for a library from environment variables."""
    env_var = _get_env_var_name(library_name)
    enabled = os.getenv(env_var, "true").lower() in ("true", "1", "yes")

    return {
        "enabled": enabled,
    }


def _load_instrumentor(library_name: str) -> Optional[Any]:
    """Load the instrumentor class for a library.

    Args:
        library_name: Name of the library

    Returns:
        The instrumentor class or None if not found
    """
    if library_name not in INSTRUMENTOR_MAPPING:
        return None

    module_name = INSTRUMENTOR_MAPPING[library_name]
    class_name = INSTRUMENTOR_CLASS_MAPPING.get(library_name)

    if not class_name:
        return None

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name, None)
    except (ImportError, AttributeError):
        logger.debug(f"Could not import instrumentor for {library_name}")
        return None


def instrument_langchain(telemetry_manager=None, **kwargs) -> bool:
    """Instrument LangChain application with AlignX embedded LangSmith implementation."""
    try:
        # Get the AlignX LangChain strategy
        strategy = get_instrumentation_strategy("langchain")
        if not strategy:
            logger.debug(f"No AlignX LangChain strategy found")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Configure strategy with AlignX-specific settings
        config = {
            "tracing_enabled": kwargs.get("tracing_enabled", True),
            "project_name": kwargs.get("project_name", "alignx-langchain"),
            "tags": kwargs.get("tags", ["alignx", "langchain"]),
            "metadata": kwargs.get("metadata", {"instrumented_by": "alignx_telemetry"}),
        }

        # Use strategy to instrument (no instrumentor class needed for custom implementation)
        success = strategy.instrument(None, config, **kwargs)

        if success:
            logger.info(
                "Successfully instrumented LangChain with AlignX embedded LangSmith"
            )
        else:
            logger.warning(
                "Failed to instrument LangChain with AlignX embedded LangSmith"
            )

        return success
    except Exception as e:
        logger.error(f"Failed to instrument LangChain: {e}", exc_info=True)
        return False


def instrument_langgraph(telemetry_manager=None, **kwargs) -> bool:
    """Instrument LangGraph application with AlignX embedded LangSmith implementation."""
    try:
        # Get the AlignX LangGraph strategy
        strategy = get_instrumentation_strategy("langgraph")
        if not strategy:
            logger.debug(f"No AlignX LangGraph strategy found")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Configure strategy with AlignX-specific settings for LangGraph
        config = {
            "tracing_enabled": kwargs.get("tracing_enabled", True),
            "project_name": kwargs.get("project_name", "alignx-langgraph"),
            "tags": kwargs.get("tags", ["alignx", "langgraph"]),
            "metadata": kwargs.get(
                "metadata",
                {
                    "instrumented_by": "alignx_telemetry",
                    "framework": "langgraph",
                    "supports_state_management": True,
                    "supports_graph_execution": True,
                },
            ),
        }

        # Use strategy to instrument
        success = strategy.instrument(None, config, **kwargs)

        if success:
            logger.info(f"Successfully instrumented LangGraph with AlignX LangSmith")

        return success
    except Exception as e:
        logger.error(f"Failed to instrument LangGraph: {e}", exc_info=True)
        return False


def instrument_llamaindex(telemetry_manager=None, **kwargs) -> bool:
    """Instrument LlamaIndex application with AlignX framework context enhancement."""
    try:
        # Get the AlignX LlamaIndex strategy
        strategy = get_instrumentation_strategy("llamaindex")
        if not strategy:
            logger.debug("No AlignX LlamaIndex strategy found")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Configure strategy with AlignX-specific settings for LlamaIndex
        config = {
            "tracing_enabled": kwargs.get("tracing_enabled", True),
            "project_name": kwargs.get("project_name", "alignx-llamaindex"),
            "tags": kwargs.get("tags", ["alignx", "llamaindex"]),
            "metadata": kwargs.get(
                "metadata",
                {
                    "instrumented_by": "alignx_telemetry",
                    "framework": "llamaindex",
                    "supports_rag": True,
                    "supports_retrieval": True,
                    "supports_synthesis": True,
                },
            ),
        }

        # Use strategy to instrument
        success = strategy.instrument(None, config, **kwargs)

        if success:
            logger.info(
                "Successfully instrumented LlamaIndex with AlignX framework context"
            )

        return success
    except Exception as e:
        logger.error(f"Failed to instrument LlamaIndex: {e}", exc_info=True)
        return False


def instrument_autogen(telemetry_manager=None, **kwargs) -> bool:
    """Instrument AutoGen application with AlignX framework context enhancement."""
    try:
        # Get the AlignX AutoGen strategy
        strategy = get_instrumentation_strategy("autogen")
        if not strategy:
            logger.debug("No AlignX AutoGen strategy found")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Configure strategy with AlignX-specific settings for AutoGen
        config = {
            "tracing_enabled": kwargs.get("tracing_enabled", True),
            "project_name": kwargs.get("project_name", "alignx-autogen"),
            "tags": kwargs.get("tags", ["alignx", "autogen"]),
            "metadata": kwargs.get(
                "metadata",
                {
                    "instrumented_by": "alignx_telemetry",
                    "framework": "autogen",
                    "supports_multi_agent": True,
                    "supports_conversation": True,
                    "supports_code_execution": True,
                },
            ),
        }

        # Use strategy to instrument
        success = strategy.instrument(None, config, **kwargs)

        if success:
            logger.info(
                "Successfully instrumented AutoGen with AlignX framework context"
            )

        return success
    except Exception as e:
        logger.error(f"Failed to instrument AutoGen: {e}", exc_info=True)
        return False


def instrument_crewai(telemetry_manager=None, **kwargs) -> bool:
    """Instrument CrewAI application with AlignX framework context enhancement."""
    try:
        # Get the AlignX CrewAI strategy
        strategy = get_instrumentation_strategy("crewai")
        if not strategy:
            logger.debug("No AlignX CrewAI strategy found")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Configure strategy with AlignX-specific settings for CrewAI
        config = {
            "tracing_enabled": kwargs.get("tracing_enabled", True),
            "project_name": kwargs.get("project_name", "alignx-crewai"),
            "tags": kwargs.get("tags", ["alignx", "crewai"]),
            "metadata": kwargs.get(
                "metadata",
                {
                    "instrumented_by": "alignx_telemetry",
                    "framework": "crewai",
                    "supports_crews": True,
                    "supports_tasks": True,
                    "supports_agents": True,
                },
            ),
        }

        # Use strategy to instrument
        success = strategy.instrument(None, config, **kwargs)

        if success:
            logger.info(
                "Successfully instrumented CrewAI with AlignX framework context"
            )

        return success
    except Exception as e:
        logger.error(f"Failed to instrument CrewAI: {e}", exc_info=True)
        return False


def instrument_specific(
    library_name: str, enabled: Optional[bool] = None, telemetry_manager=None, **kwargs
) -> bool:
    """Instrument a specific library with AlignX enhancements.

    Args:
        library_name: Name of the library to instrument
        enabled: Override whether to enable instrumentation
        telemetry_manager: AlignX telemetry manager for enhanced metrics
        **kwargs: Additional configuration parameters for the instrumentor
            For libraries that require an app instance (like FastAPI, Flask):
                app: The application instance to instrument
            For middleware-based libraries (like WSGI, ASGI):
                app: The application instance to wrap with middleware

    Returns:
        True if instrumentation was successful, False otherwise
    """

    # Check if this is a deprecated LLM provider instrumentor
    if library_name in DEPRECATED_LLM_INSTRUMENTORS:
        warnings.warn(
            f"LLM provider instrumentor '{library_name}' is deprecated. "
            f"Please use the new AlignX provider system instead:\n\n"
            f"# Old approach (deprecated):\n"
            f"# instrument_specific('{library_name}')\n\n"
            f"# New approach (recommended):\n"
            f"from alignx_telemetry.providers import instrument_provider, instrument_all_providers\n"
            f"instrument_provider('{library_name}', telemetry_manager)  # For specific provider\n"
            f"# OR\n"
            f"instrument_all_providers(telemetry_manager)  # For all available providers\n\n"
            f"The new provider system offers enhanced metrics, better performance, "
            f"and eliminates duplicate spans.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Try to auto-migrate to the new provider system
        try:
            from alignx_telemetry.providers import instrument_provider

            logger.info(f"Auto-migrating {library_name} to new provider system...")
            return instrument_provider(library_name, telemetry_manager, **kwargs)
        except ImportError:
            logger.error(
                f"Failed to auto-migrate {library_name} - provider system not available"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to auto-migrate {library_name}: {e}")
            return False

    # Get configuration from environment
    config = _get_config_from_env(library_name)

    # Override enabled status if specified
    if enabled is not None:
        config["enabled"] = enabled

    # Skip if disabled
    if not config.get("enabled", True):
        logger.info(f"Instrumentation disabled for {library_name}")
        return False

    # Add provided configuration
    config.update(kwargs)

    try:
        # Get the instrumentor class
        instrumentor_cls = _load_instrumentor(library_name)
        if not instrumentor_cls:
            logger.debug(f"Instrumentation for {library_name} is not available")
            return False

        # Get the strategy for the library
        strategy = get_instrumentation_strategy(library_name)
        if not strategy:
            logger.debug(f"No strategy found for {library_name}")
            return False

        # Add telemetry_manager to kwargs so it gets passed to the strategy
        if telemetry_manager is not None:
            kwargs["telemetry_manager"] = telemetry_manager

        # Use strategy to instrument with instrumentor class
        success = strategy.instrument(instrumentor_cls, config, **kwargs)

        if success:
            logger.info(f"Successfully instrumented {library_name}")

        return success

    except ImportError as e:
        logger.debug(f"Library {library_name} not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to instrument {library_name}: {e}")
        return False


def instrument_all(
    excluded_libraries: Optional[List[str]] = None,
    included_libraries: Optional[List[str]] = None,
    telemetry_manager=None,
    **kwargs,
) -> Dict[str, bool]:
    """Instrument all available libraries.

    Args:
        excluded_libraries: List of libraries to exclude from instrumentation
        included_libraries: List of libraries to specifically include (if None, all are included)
        telemetry_manager: AlignX telemetry manager for enhanced hooks
        **kwargs: Additional configuration for all instrumentors
            For app-based libraries, you can provide apps in the format:
                {"fastapi": {"app": fastapi_app}, "flask": {"app": flask_app}}

    Returns:
        Dictionary mapping library names to whether they were successfully instrumented
    """
    # Default excluded libraries are the ones that are automatically instrumented
    # when auto_instrument=True
    # This is because these libraries need 'app' parameter to be passed in
    default_excluded_libraries = ["fastapi", "flask", "starlette"]
    excluded = set(excluded_libraries or default_excluded_libraries)
    included = set(included_libraries or get_instrumentors())

    results = {}

    # Instrument custom AlignX implementations first
    results["langchain"] = instrument_langchain(
        telemetry_manager=telemetry_manager, **kwargs.get("langchain", {})
    )
    results["langgraph"] = instrument_langgraph(
        telemetry_manager=telemetry_manager, **kwargs.get("langgraph", {})
    )

    # Instrument Phase 3 framework modules
    results["llamaindex"] = instrument_llamaindex(
        telemetry_manager=telemetry_manager, **kwargs.get("llamaindex", {})
    )
    results["autogen"] = instrument_autogen(
        telemetry_manager=telemetry_manager, **kwargs.get("autogen", {})
    )
    results["crewai"] = instrument_crewai(
        telemetry_manager=telemetry_manager, **kwargs.get("crewai", {})
    )

    for library_name in INSTRUMENTOR_MAPPING:
        if library_name in excluded or library_name not in included:
            results[library_name] = False
            continue

        # Get library-specific kwargs if they exist
        lib_kwargs = {}

        # Check for direct kwargs format: kwargs[library_name]
        if library_name in kwargs:
            lib_kwargs.update(kwargs[library_name])

        # Instrument the library
        results[library_name] = instrument_specific(
            library_name, telemetry_manager=telemetry_manager, **lib_kwargs
        )

    return results


def get_instrumentors() -> List[str]:
    """Get a list of all available instrumentor names."""
    return list(INSTRUMENTOR_MAPPING.keys())


def is_instrumentation_supported(library_name: str) -> bool:
    """Check if instrumentation is supported for a library."""
    return library_name in INSTRUMENTOR_MAPPING
