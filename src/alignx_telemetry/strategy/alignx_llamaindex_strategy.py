"""
AlignX LlamaIndex Framework Strategy

This strategy provides framework-aware instrumentation for LlamaIndex applications.
It leverages LlamaIndex's dispatcher pattern to add AlignX-specific context to provider-level spans.
"""

import logging
from typing import Any, Dict, Optional

from ..providers.context import AlignXFrameworkContext
from .strategies import InstrumentationStrategy

logger = logging.getLogger(__name__)

# Optional LlamaIndex imports with fallbacks
try:
    from llama_index.core.instrumentation import get_dispatcher
    from llama_index.core.instrumentation.dispatcher import Dispatcher
    from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
    from llama_index.core.instrumentation.span.base import BaseSpan
    from llama_index.core.instrumentation.events.span import SpanDropEvent
    import inspect
    from typing import List

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    get_dispatcher = None
    Dispatcher = None
    BaseSpanHandler = None
    BaseSpan = None
    SpanDropEvent = None
    inspect = None


class AlignXLlamaIndexSpanHandler(BaseSpanHandler if LLAMAINDEX_AVAILABLE else object):
    """Custom span handler that adds AlignX framework context to provider-level spans."""

    def __init__(self):
        if LLAMAINDEX_AVAILABLE:
            super().__init__()
        self.alignx_context = AlignXFrameworkContext()

    def new_span(
        self,
        id_: str,
        bound_args: "inspect.BoundArguments",
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional["BaseSpan"]:
        """Create a new span and set AlignX framework context."""
        if not LLAMAINDEX_AVAILABLE:
            return None

        # Extract framework context from the span
        framework_context = self._extract_llamaindex_context(
            id_, bound_args, instance, tags
        )

        # Set the framework context for provider instrumentation
        if framework_context:
            self.alignx_context.set_framework_context(**framework_context)

        # Don't create actual spans - let provider instrumentation handle that
        return None

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: "inspect.BoundArguments",
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional["BaseSpan"]:
        """Prepare to exit span and clear AlignX context."""
        if not LLAMAINDEX_AVAILABLE:
            return None

        # Clear framework context when exiting major operations
        if self._is_major_operation(id_):
            self.alignx_context.clear_framework_context()

        return None

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: "inspect.BoundArguments",
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional["BaseSpan"]:
        """Prepare to drop span and clear AlignX context."""
        if not LLAMAINDEX_AVAILABLE:
            return None

        # Clear framework context on error
        self.alignx_context.clear_framework_context()
        return None

    def _extract_llamaindex_context(
        self,
        span_id: str,
        bound_args: "inspect.BoundArguments",
        instance: Optional[Any],
        tags: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract LlamaIndex-specific context from span data."""
        context = {
            "framework": "llamaindex",
            "span_id": span_id,
        }

        # Extract operation type from span ID or instance
        if "synthesize" in span_id.lower():
            context["operation"] = "synthesize"
        elif "retrieve" in span_id.lower():
            context["operation"] = "retrieve"
        elif "query" in span_id.lower():
            context["operation"] = "query"
        elif "index" in span_id.lower():
            context["operation"] = "index"
        elif "embed" in span_id.lower():
            context["operation"] = "embed"
        else:
            context["operation"] = "unknown"

        # Extract workflow information from instance
        if instance:
            instance_type = type(instance).__name__
            context["component"] = instance_type

            # Check for specific LlamaIndex components
            if hasattr(instance, "callback_manager"):
                context["has_callbacks"] = True
            if hasattr(instance, "_llm"):
                context["has_llm"] = True
            if hasattr(instance, "_embed_model"):
                context["has_embeddings"] = True

        # Extract additional metadata from tags
        if tags:
            context["tags"] = {
                k: v for k, v in tags.items() if isinstance(v, (str, int, float, bool))
            }

        # Extract query information from bound arguments
        if bound_args and bound_args.arguments:
            args = bound_args.arguments
            if "query" in args:
                query = args["query"]
                if hasattr(query, "query_str"):
                    context["query_type"] = "QueryBundle"
                elif isinstance(query, str):
                    context["query_type"] = "string"
                    context["query_length"] = len(query)

            if "nodes" in args:
                nodes = args["nodes"]
                if hasattr(nodes, "__len__"):
                    context["node_count"] = len(nodes)

        return context

    def _is_major_operation(self, span_id: str) -> bool:
        """Check if this is a major operation that should clear context."""
        major_ops = ["synthesize", "query", "retrieve", "index"]
        return any(op in span_id.lower() for op in major_ops)


class AlignXLlamaIndexStrategy(InstrumentationStrategy):
    """AlignX strategy for LlamaIndex framework instrumentation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.span_handler: Optional[AlignXLlamaIndexSpanHandler] = None
        self.dispatcher: Optional["Dispatcher"] = None

    @property
    def name(self) -> str:
        return "AlignXLlamaIndexStrategy"

    @property
    def framework_name(self) -> str:
        return "llamaindex"

    def is_available(self) -> bool:
        """Check if LlamaIndex is available for instrumentation."""
        return LLAMAINDEX_AVAILABLE

    def instrument(self, **kwargs) -> bool:
        """
        Instrument LlamaIndex to add AlignX framework context.

        This leverages LlamaIndex's dispatcher system to add framework-specific
        context to provider-level spans without creating duplicate spans.
        """
        if not LLAMAINDEX_AVAILABLE:
            logger.warning(
                "LlamaIndex is not installed. Skipping LlamaIndex instrumentation."
            )
            return False

        try:
            # Get the root dispatcher
            self.dispatcher = get_dispatcher("root")

            # Create and add our custom span handler
            self.span_handler = AlignXLlamaIndexSpanHandler()
            self.dispatcher.add_span_handler(self.span_handler)

            logger.info(
                "AlignX LlamaIndex framework instrumentation enabled. "
                "Framework context will be added to provider-level spans."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to instrument LlamaIndex: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove LlamaIndex instrumentation."""
        if not LLAMAINDEX_AVAILABLE or not self.dispatcher or not self.span_handler:
            return True

        try:
            # Remove our span handler from the dispatcher
            if self.span_handler in self.dispatcher.span_handlers:
                self.dispatcher.span_handlers.remove(self.span_handler)

            self.span_handler = None
            self.dispatcher = None

            logger.info("AlignX LlamaIndex framework instrumentation disabled.")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument LlamaIndex: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the LlamaIndex instrumentation."""
        metadata = {
            "strategy": self.name,
            "framework": self.framework_name,
            "available": self.is_available(),
            "instrumented": self.span_handler is not None,
        }

        if LLAMAINDEX_AVAILABLE:
            try:
                import llama_index

                metadata["llama_index_version"] = llama_index.__version__
            except (ImportError, AttributeError):
                metadata["llama_index_version"] = "unknown"

        return metadata
