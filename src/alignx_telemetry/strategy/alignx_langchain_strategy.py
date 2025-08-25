"""AlignX LangChain Strategy - Custom LangSmith implementation without API dependencies."""

import logging
import os
from typing import Dict, Any, Type, Optional

from alignx_telemetry.strategy.strategies import InstrumentationStrategy

# Logger for LangChain strategy
logger = logging.getLogger(__name__)


class AlignXLangChainStrategy(InstrumentationStrategy):
    """Enhanced LangChain strategy using embedded AlignX LangSmith tracing.

    This strategy provides LangChain instrumentation without dependencies on
    the external LangSmith API. All tracing data flows through AlignX's
    OpenTelemetry infrastructure.
    """

    def __init__(self):
        """Initialize the LangChain strategy."""
        super().__init__()
        self._instrumented = False

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        """Instrument LangChain using AlignX's embedded LangSmith implementation.

        Args:
            instrumentor_cls: Not used for LangChain (can be None)
            config: Configuration dictionary
            **kwargs: Additional configuration including telemetry_manager

        Returns:
            True if instrumentation was successful, False otherwise
        """
        if self._instrumented:
            logger.debug("LangChain already instrumented with AlignX LangSmith")
            return True

        try:
            # Get telemetry manager for OpenTelemetry integration
            telemetry_manager = kwargs.get("telemetry_manager")

            # Check if langchain is available
            try:
                import langchain_core
            except ImportError:
                logger.warning(
                    "LangChain instrumentation requires 'langchain-core' package. "
                    "Install it with 'pip install langchain-core'"
                )
                return False

            # Configure AlignX LangSmith tracing
            tracing_enabled = config.get("tracing_enabled", True)
            project_name = config.get("project_name", "alignx-langchain")

            # Store configuration for callback handler
            self._current_project_name = project_name
            self._current_tags = config.get("tags", ["alignx", "langchain"])
            self._current_metadata = config.get(
                "metadata", {"created_from": "alignx_telemetry"}
            )

            # Check if the standard environment variables are set
            otel_env_enabled = os.getenv(
                "OTEL_PYTHON_INSTRUMENT_LANGCHAIN", ""
            ).lower() in ("true", "1", "yes")

            # Use standard env vars if set, otherwise use config
            tracing_enabled = otel_env_enabled or tracing_enabled

            if not tracing_enabled:
                logger.debug("LangChain instrumentation is disabled")
                return False

            # Set up AlignX LangSmith configuration
            from alignx_telemetry.langsmith import configure
            from alignx_telemetry.langsmith.run_helpers import (
                set_callback_handler,
                TracingCallbackHandler,
            )

            # Configure the embedded LangSmith
            configure(
                enabled=True,
                project_name=project_name,
                tags=config.get("tags", ["alignx", "langchain"]),
                metadata=config.get("metadata", {"created_from": "alignx_telemetry"}),
            )

            # Set up callback handler to integrate with AlignX telemetry
            if telemetry_manager:
                callback_handler = TracingCallbackHandler(
                    telemetry_manager=telemetry_manager
                )
                set_callback_handler(callback_handler)
                logger.info("AlignX LangSmith callback handler configured")

            # Instrument LangChain callbacks to use AlignX tracing
            self._instrument_langchain_callbacks()

            self._instrumented = True
            logger.info("Successfully instrumented LangChain with AlignX LangSmith")
            return True

        except Exception as e:
            logger.error(
                f"Failed to instrument LangChain with AlignX LangSmith: {e}",
                exc_info=True,
            )
            return False

    def _instrument_langchain_callbacks(self) -> None:
        """Instrument LangChain callback system to use AlignX tracing."""
        try:
            # Import LangChain components
            from langchain_core.callbacks import BaseCallbackManager
            from alignx_telemetry.langsmith.callbacks import (
                AlignXLangChainCallbackHandler,
            )

            # Create our custom callback handler with enhanced LangGraph support
            alignx_handler = AlignXLangChainCallbackHandler(
                project_name=getattr(self, "_current_project_name", "alignx-langchain"),
                tags=getattr(self, "_current_tags", ["alignx", "langchain"]),
                metadata=getattr(
                    self, "_current_metadata", {"created_from": "alignx_telemetry"}
                ),
            )

            # Store handler for potential LangGraph graph definition tracking
            self._alignx_handler = alignx_handler

            # Monkey patch the BaseCallbackManager to include our handler
            original_init = BaseCallbackManager.__init__

            def patched_init(
                self,
                handlers=None,
                inheritable_handlers=None,
                parent_run_id=None,
                **kwargs,
            ):
                handlers = handlers or []
                # Add our AlignX handler to the list
                handlers.append(alignx_handler)
                return original_init(
                    self, handlers, inheritable_handlers, parent_run_id, **kwargs
                )

            BaseCallbackManager.__init__ = patched_init
            logger.debug("LangChain BaseCallbackManager patched with AlignX handler")

        except Exception as e:
            logger.warning(f"Failed to patch LangChain callbacks: {e}")
            # Fall back to environment variable approach
            self._setup_langsmith_environment_variables()

    def _setup_langsmith_environment_variables(self) -> None:
        """Set up LangSmith environment variables for basic tracing."""
        try:
            # Set environment variables that LangChain recognizes
            os.environ["LANGSMITH_TRACING"] = "true"
            # Don't set LANGSMITH_API_KEY to avoid API calls
            # LangChain will create traces locally without uploading

            logger.info("LangSmith environment variables configured for local tracing")

        except Exception as e:
            logger.error(f"Failed to configure LangSmith environment: {e}")

    def create_enhanced_callback_handler(self, graph=None, **kwargs):
        """Create an enhanced callback handler with optional graph definition tracking.

        Args:
            graph: Optional LangGraph Graph object for definition tracking.
            **kwargs: Additional configuration.

        Returns:
            AlignXLangChainCallbackHandler instance.
        """
        try:
            from alignx_telemetry.langsmith.callbacks import (
                AlignXLangChainCallbackHandler,
            )

            return AlignXLangChainCallbackHandler(
                project_name=getattr(self, "_current_project_name", "alignx-langchain"),
                tags=getattr(self, "_current_tags", ["alignx", "langchain"]),
                metadata=getattr(
                    self, "_current_metadata", {"created_from": "alignx_telemetry"}
                ),
                graph=graph,  # This will enable graph definition tracking
            )
        except Exception as e:
            logger.error(f"Failed to create enhanced callback handler: {e}")
            return None


class AlignXLangGraphStrategy(AlignXLangChainStrategy):
    """Enhanced LangGraph strategy using AlignX LangSmith tracing."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        """Instrument LangGraph using AlignX's embedded LangSmith implementation."""
        # Check if langgraph is available
        try:
            import langgraph
        except ImportError:
            logger.warning(
                "LangGraph instrumentation requires 'langgraph' package. "
                "Install it with 'pip install langgraph'"
            )
            return False

        # Add LangGraph-specific tags
        config = dict(config)  # Copy to avoid modifying original
        tags = config.get("tags", ["alignx", "langchain"])
        if "langgraph" not in tags:
            tags.append("langgraph")
        config["tags"] = tags

        # Update metadata for LangGraph
        metadata = config.get("metadata", {})
        metadata.update(
            {
                "created_from": "alignx_telemetry",
                "framework": "langgraph",
                "supports_state_management": True,
                "supports_graph_execution": True,
            }
        )
        config["metadata"] = metadata

        # Use the same instrumentation as LangChain but with LangGraph enhancements
        success = super().instrument(instrumentor_cls, config, **kwargs)

        if success:
            logger.info("Successfully instrumented LangGraph with AlignX LangSmith")

            # Additional LangGraph-specific setup
            self._setup_langgraph_enhancements()

        return success

    def _setup_langgraph_enhancements(self):
        """Set up LangGraph-specific enhancements."""
        try:
            # Import LangGraph components for enhanced integration
            from langgraph.graph import StateGraph
            from langgraph.pregel import Pregel

            # Monkey patch StateGraph.compile to automatically track graph definitions
            original_compile = StateGraph.compile
            alignx_strategy = self

            def enhanced_compile(self, **kwargs):
                """Enhanced compile method that tracks graph definitions."""
                # Call original compile
                compiled_graph = original_compile(self, **kwargs)

                # Try to extract and store graph definition
                try:
                    if hasattr(compiled_graph, "get_graph"):
                        graph_def = compiled_graph.get_graph(xray=True)
                        if hasattr(alignx_strategy, "_alignx_handler"):
                            # Update handler with graph definition
                            alignx_strategy._alignx_handler.default_metadata[
                                "_alignx_graph_definition"
                            ] = {
                                "format": "mermaid",
                                "data": (
                                    graph_def.draw_mermaid()
                                    if hasattr(graph_def, "draw_mermaid")
                                    else str(graph_def)
                                ),
                            }
                            if (
                                "langgraph"
                                not in alignx_strategy._alignx_handler.default_tags
                            ):
                                alignx_strategy._alignx_handler.default_tags.append(
                                    "langgraph"
                                )
                except Exception as e:
                    logger.debug(
                        f"Failed to extract graph definition during compile: {e}"
                    )

                return compiled_graph

            StateGraph.compile = enhanced_compile
            logger.debug(
                "LangGraph StateGraph.compile enhanced with graph definition tracking"
            )

        except Exception as e:
            logger.warning(f"Failed to set up LangGraph enhancements: {e}")
