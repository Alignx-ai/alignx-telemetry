"""Context management for AlignX provider instrumentation.

This module provides thread-safe context management for framework information
and instrumentation suppression, inspired by OpenLLMetry's suppression patterns.
"""

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class AlignXFrameworkContext:
    """Thread-safe context management for framework information.

    This class manages framework context that gets propagated to provider
    instrumentation, enabling rich observability without span duplication.

    Inspired by:
    - OpenLLMetry's SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY pattern
    - Opik's context management for framework integration
    - LangChain's callback context propagation
    """

    # Context variables for framework information
    _current_framework: ContextVar[Optional[str]] = ContextVar(
        "alignx_framework", default=None
    )
    _current_workflow_id: ContextVar[Optional[str]] = ContextVar(
        "alignx_workflow_id", default=None
    )
    _current_node_name: ContextVar[Optional[str]] = ContextVar(
        "alignx_node_name", default=None
    )
    _current_agent_name: ContextVar[Optional[str]] = ContextVar(
        "alignx_agent_name", default=None
    )
    _current_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
        "alignx_metadata", default=None
    )

    # Suppression context variables
    _suppressed_providers: ContextVar[Optional[Set[str]]] = ContextVar(
        "alignx_suppressed_providers", default=None
    )
    _global_suppression: ContextVar[bool] = ContextVar(
        "alignx_global_suppression", default=False
    )

    @classmethod
    def set_framework_context(
        cls,
        framework: str,
        workflow_id: Optional[str] = None,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set framework context for downstream provider calls.

        Args:
            framework: Framework name (e.g., "langchain", "llamaindex", "crewai")
            workflow_id: Unique workflow identifier
            node_name: Specific component/node name
            agent_name: Agent name (for multi-agent frameworks)
            metadata: Additional metadata
        """
        cls._current_framework.set(framework)

        if workflow_id:
            cls._current_workflow_id.set(workflow_id)

        if node_name:
            cls._current_node_name.set(node_name)

        if agent_name:
            cls._current_agent_name.set(agent_name)

        if metadata:
            current_metadata = cls._current_metadata.get({})
            updated_metadata = {**current_metadata, **metadata}
            cls._current_metadata.set(updated_metadata)

        logger.debug(
            f"Set framework context: {framework}, workflow: {workflow_id}, node: {node_name}"
        )

    @classmethod
    def get_current_context(cls) -> Dict[str, Optional[str]]:
        """Get current framework context.

        Returns:
            Dictionary containing current context information
        """
        return {
            "framework": cls._current_framework.get(),
            "workflow_id": cls._current_workflow_id.get(),
            "node_name": cls._current_node_name.get(),
            "agent_name": cls._current_agent_name.get(),
            "metadata": cls._current_metadata.get({}),
        }

    @classmethod
    def clear_context(cls) -> None:
        """Clear all framework context."""
        cls._current_framework.set(None)
        cls._current_workflow_id.set(None)
        cls._current_node_name.set(None)
        cls._current_agent_name.set(None)
        cls._current_metadata.set({})

        logger.debug("Cleared framework context")

    @classmethod
    @contextmanager
    def framework_context(
        cls,
        framework: str,
        workflow_id: Optional[str] = None,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for framework-aware operations.

        Args:
            framework: Framework name
            workflow_id: Unique workflow identifier
            node_name: Specific component/node name
            agent_name: Agent name
            metadata: Additional metadata

        Example:
            with AlignXFrameworkContext.framework_context(
                framework="langchain",
                workflow_id="rag_pipeline_001",
                node_name="answer_generation"
            ):
                # Any LLM calls here will have framework context
                response = openai_client.chat.completions.create(...)
        """
        # Store current context
        previous_context = cls.get_current_context()

        try:
            # Set new context
            cls.set_framework_context(
                framework=framework,
                workflow_id=workflow_id,
                node_name=node_name,
                agent_name=agent_name,
                metadata=metadata,
            )
            yield
        finally:
            # Restore previous context
            cls._restore_context(previous_context)

    @classmethod
    async def async_framework_context(
        cls,
        framework: str,
        coro,
        workflow_id: Optional[str] = None,
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Async context manager for framework-aware operations.

        Args:
            framework: Framework name
            coro: Coroutine to execute with context
            workflow_id: Unique workflow identifier
            node_name: Specific component/node name
            agent_name: Agent name
            metadata: Additional metadata

        Returns:
            Result of the coroutine

        Example:
            result = await AlignXFrameworkContext.async_framework_context(
                framework="langchain",
                coro=async_llm_call(),
                workflow_id="async_rag_001"
            )
        """
        # Copy current context
        ctx = copy_context()

        # Set framework context in the copied context
        def set_context():
            cls.set_framework_context(
                framework=framework,
                workflow_id=workflow_id,
                node_name=node_name,
                agent_name=agent_name,
                metadata=metadata,
            )

        ctx.run(set_context)

        # Run coroutine with the framework context
        return await asyncio.create_task(coro, context=ctx)

    @classmethod
    def _restore_context(cls, context: Dict[str, Any]) -> None:
        """Restore previous context state.

        Args:
            context: Previous context to restore
        """
        cls._current_framework.set(context.get("framework"))
        cls._current_workflow_id.set(context.get("workflow_id"))
        cls._current_node_name.set(context.get("node_name"))
        cls._current_agent_name.set(context.get("agent_name"))
        cls._current_metadata.set(context.get("metadata", {}))

    # Suppression methods for avoiding duplicate spans

    @classmethod
    def suppress_provider_instrumentation(cls, provider: str) -> None:
        """Suppress instrumentation for a specific provider.

        This is used to avoid duplicate spans when framework-level
        instrumentation is handling the same LLM calls.

        Args:
            provider: Provider name to suppress
        """
        current_suppressed = cls._suppressed_providers.get(set())
        updated_suppressed = current_suppressed | {provider}
        cls._suppressed_providers.set(updated_suppressed)

        logger.debug(f"Suppressed provider instrumentation: {provider}")

    @classmethod
    def unsuppress_provider_instrumentation(cls, provider: str) -> None:
        """Remove suppression for a specific provider.

        Args:
            provider: Provider name to unsuppress
        """
        current_suppressed = cls._suppressed_providers.get(set())
        updated_suppressed = current_suppressed - {provider}
        cls._suppressed_providers.set(updated_suppressed)

        logger.debug(f"Unsuppressed provider instrumentation: {provider}")

    @classmethod
    def should_suppress_provider_instrumentation(cls, provider: str) -> bool:
        """Check if provider instrumentation should be suppressed.

        Args:
            provider: Provider name to check

        Returns:
            True if instrumentation should be suppressed
        """
        # Check global suppression
        if cls._global_suppression.get(False):
            return True

        # Check provider-specific suppression
        suppressed_providers = cls._suppressed_providers.get(set())
        return provider in suppressed_providers

    @classmethod
    @contextmanager
    def suppress_provider(cls, provider: str):
        """Context manager to temporarily suppress a provider.

        Args:
            provider: Provider name to suppress

        Example:
            with AlignXFrameworkContext.suppress_provider("openai"):
                # OpenAI instrumentation will be suppressed here
                response = some_framework_call_that_uses_openai()
        """
        cls.suppress_provider_instrumentation(provider)
        try:
            yield
        finally:
            cls.unsuppress_provider_instrumentation(provider)

    @classmethod
    @contextmanager
    def suppress_all_providers(cls):
        """Context manager to temporarily suppress all provider instrumentation.

        Example:
            with AlignXFrameworkContext.suppress_all_providers():
                # All provider instrumentation will be suppressed
                response = framework_with_own_tracing()
        """
        cls._global_suppression.set(True)
        try:
            yield
        finally:
            cls._global_suppression.set(False)

    @classmethod
    def get_suppression_status(cls) -> Dict[str, Any]:
        """Get current suppression status for debugging.

        Returns:
            Dictionary with suppression information
        """
        return {
            "global_suppression": cls._global_suppression.get(False),
            "suppressed_providers": list(cls._suppressed_providers.get(set())),
        }


class AlignXTraceContext:
    """Extended trace context management for complex workflows.

    This class provides additional context management for trace correlation
    and workflow-level observability.
    """

    _current_trace_id: ContextVar[Optional[str]] = ContextVar(
        "alignx_trace_id", default=None
    )
    _current_span_id: ContextVar[Optional[str]] = ContextVar(
        "alignx_span_id", default=None
    )
    _workflow_type: ContextVar[Optional[str]] = ContextVar(
        "alignx_workflow_type", default=None
    )

    @classmethod
    def set_trace_context(
        cls,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
    ) -> None:
        """Set trace context information.

        Args:
            trace_id: Current trace ID
            span_id: Current span ID
            workflow_type: Type of workflow (e.g., "rag", "agent", "chain")
        """
        if trace_id:
            cls._current_trace_id.set(trace_id)
        if span_id:
            cls._current_span_id.set(span_id)
        if workflow_type:
            cls._workflow_type.set(workflow_type)

    @classmethod
    def get_trace_context(cls) -> Dict[str, Optional[str]]:
        """Get current trace context.

        Returns:
            Dictionary with trace context information
        """
        return {
            "trace_id": cls._current_trace_id.get(),
            "span_id": cls._current_span_id.get(),
            "workflow_type": cls._workflow_type.get(),
        }

    @classmethod
    def clear_trace_context(cls) -> None:
        """Clear trace context."""
        cls._current_trace_id.set(None)
        cls._current_span_id.set(None)
        cls._workflow_type.set(None)
