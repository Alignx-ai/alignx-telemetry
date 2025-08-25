"""
AlignX AutoGen Framework Strategy

This strategy provides framework-aware instrumentation for AutoGen applications.
It integrates with AutoGen's OpenTelemetry-based tracing to add AlignX-specific context.
"""

import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

from ..providers.context import AlignXFrameworkContext
from .strategies import InstrumentationStrategy

logger = logging.getLogger(__name__)

# Optional AutoGen imports with fallbacks
try:
    from autogen_core._telemetry import TraceHelper, MessageRuntimeTracingConfig
    from autogen_core._telemetry._genai import (
        GenAiOperationNameValues,
        GENAI_SYSTEM_AUTOGEN,
        GEN_AI_AGENT_ID,
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_DESCRIPTION,
        GEN_AI_OPERATION_NAME,
        GEN_AI_SYSTEM,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_DESCRIPTION,
    )
    import wrapt

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    TraceHelper = None
    MessageRuntimeTracingConfig = None
    GenAiOperationNameValues = None
    GENAI_SYSTEM_AUTOGEN = None
    wrapt = None


class AlignXAutoGenContext:
    """Context manager for AutoGen framework information."""

    def __init__(self):
        self.alignx_context = AlignXFrameworkContext()
        self._active_agents: Dict[str, Dict[str, Any]] = {}
        self._active_tools: Dict[str, Dict[str, Any]] = {}

    def set_agent_context(
        self, agent_id: str, agent_name: str, description: str = None
    ):
        """Set the current agent context."""
        agent_info = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "description": description,
        }
        self._active_agents[agent_id] = agent_info

        # Set framework context for provider instrumentation
        self.alignx_context.set_framework_context(
            framework="autogen",
            agent_id=agent_id,
            agent_name=agent_name,
            operation="agent_operation",
        )

    def set_tool_context(self, tool_name: str, description: str = None):
        """Set the current tool context."""
        tool_info = {
            "tool_name": tool_name,
            "description": description,
        }
        self._active_tools[tool_name] = tool_info

        # Update framework context for tool execution
        current_context = self.alignx_context.get_framework_context()
        if current_context:
            current_context.update(
                {"tool_name": tool_name, "operation": "tool_execution"}
            )
            self.alignx_context.set_framework_context(**current_context)

    def clear_agent_context(self, agent_id: str):
        """Clear specific agent context."""
        self._active_agents.pop(agent_id, None)
        if not self._active_agents:
            self.alignx_context.clear_framework_context()

    def clear_tool_context(self, tool_name: str):
        """Clear specific tool context."""
        self._active_tools.pop(tool_name, None)

    def get_current_context(self) -> Dict[str, Any]:
        """Get the current AutoGen context."""
        return {
            "framework": "autogen",
            "active_agents": list(self._active_agents.keys()),
            "active_tools": list(self._active_tools.keys()),
            "agent_details": self._active_agents,
            "tool_details": self._active_tools,
        }


class AlignXAutoGenStrategy(InstrumentationStrategy):
    """AlignX strategy for AutoGen framework instrumentation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.context_manager = AlignXAutoGenContext()
        self._instrumented = False

    @property
    def name(self) -> str:
        return "AlignXAutoGenStrategy"

    @property
    def framework_name(self) -> str:
        return "autogen"

    def is_available(self) -> bool:
        """Check if AutoGen is available for instrumentation."""
        return AUTOGEN_AVAILABLE

    def instrument(self, **kwargs) -> bool:
        """
        Instrument AutoGen to add AlignX framework context.

        This hooks into AutoGen's telemetry system to add framework-specific
        context to provider-level spans.
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning(
                "AutoGen is not installed. Skipping AutoGen instrumentation."
            )
            return False

        if self._instrumented:
            logger.debug("AutoGen is already instrumented.")
            return True

        try:
            self._instrument_autogen_telemetry()
            self._instrument_agent_operations()
            self._instrument_tool_operations()

            self._instrumented = True
            logger.info(
                "AlignX AutoGen framework instrumentation enabled. "
                "Framework context will be added to provider-level spans."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to instrument AutoGen: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove AutoGen instrumentation."""
        if not AUTOGEN_AVAILABLE or not self._instrumented:
            return True

        try:
            # AutoGen uses OpenTelemetry directly, so we don't need to actively uninstrument
            # Our context manager will simply stop providing context
            self.context_manager.alignx_context.clear_framework_context()
            self._instrumented = False

            logger.info("AlignX AutoGen framework instrumentation disabled.")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument AutoGen: {e}")
            return False

    def _instrument_autogen_telemetry(self):
        """Hook into AutoGen's telemetry system."""
        if not AUTOGEN_AVAILABLE:
            return

        # Monkey patch TraceHelper to add our context
        original_trace_block = TraceHelper.trace_block

        @contextmanager
        def enhanced_trace_block(self, operation, destination, parent, **kwargs):
            # Extract AutoGen operation context
            operation_name = str(operation)
            self.context_manager._extract_autogen_operation_context(
                operation_name, kwargs
            )

            # Use original trace_block
            with original_trace_block(
                self, operation, destination, parent, **kwargs
            ) as span:
                yield span

        TraceHelper.trace_block = enhanced_trace_block

    def _instrument_agent_operations(self):
        """Instrument AutoGen agent operations."""
        if not AUTOGEN_AVAILABLE:
            return

        try:
            # Try to instrument common AutoGen agent classes
            self._wrap_agent_methods()
        except Exception as e:
            logger.debug(f"Could not instrument AutoGen agents: {e}")

    def _instrument_tool_operations(self):
        """Instrument AutoGen tool operations."""
        if not AUTOGEN_AVAILABLE:
            return

        try:
            # Try to instrument tool execution
            self._wrap_tool_methods()
        except Exception as e:
            logger.debug(f"Could not instrument AutoGen tools: {e}")

    def _wrap_agent_methods(self):
        """Wrap AutoGen agent methods to add context."""
        try:
            # Try to wrap common agent patterns from autogen-agentchat
            import autogen_agentchat

            def wrap_agent_method(wrapped, instance, args, kwargs):
                # Extract agent context
                agent_id = getattr(instance, "id", getattr(instance, "name", "unknown"))
                agent_name = getattr(
                    instance,
                    "name",
                    getattr(instance, "__class__.__name__", "UnknownAgent"),
                )
                description = getattr(instance, "description", None)

                self.context_manager.set_agent_context(
                    agent_id, agent_name, description
                )

                try:
                    result = wrapped(*args, **kwargs)
                    return result
                finally:
                    self.context_manager.clear_agent_context(agent_id)

            # Wrap common agent methods
            for class_name in ["Agent", "AssistantAgent", "UserProxyAgent"]:
                try:
                    agent_class = getattr(autogen_agentchat, class_name, None)
                    if agent_class and hasattr(agent_class, "on_messages"):
                        wrapt.wrap_function_wrapper(
                            agent_class, "on_messages", wrap_agent_method
                        )
                except (AttributeError, ImportError):
                    continue

        except ImportError:
            logger.debug("autogen-agentchat not available for instrumentation")

    def _wrap_tool_methods(self):
        """Wrap AutoGen tool methods to add context."""
        try:
            # Try to wrap tool execution patterns
            import autogen_core.tools

            def wrap_tool_method(wrapped, instance, args, kwargs):
                # Extract tool context
                tool_name = getattr(
                    instance,
                    "name",
                    getattr(instance, "__class__.__name__", "UnknownTool"),
                )
                description = getattr(instance, "description", None)

                self.context_manager.set_tool_context(tool_name, description)

                try:
                    result = wrapped(*args, **kwargs)
                    return result
                finally:
                    self.context_manager.clear_tool_context(tool_name)

            # Wrap tool execution if available
            if hasattr(autogen_core.tools, "Tool"):
                wrapt.wrap_function_wrapper(
                    autogen_core.tools.Tool, "__call__", wrap_tool_method
                )

        except ImportError:
            logger.debug("autogen-core tools not available for instrumentation")

    def _extract_autogen_operation_context(
        self, operation_name: str, kwargs: Dict[str, Any]
    ):
        """Extract context from AutoGen operation."""
        context = {"framework": "autogen", "operation": operation_name}

        # Extract additional context from kwargs
        if "attributes" in kwargs:
            attributes = kwargs["attributes"] or {}

            # Extract agent information
            if GEN_AI_AGENT_ID in attributes:
                context["agent_id"] = attributes[GEN_AI_AGENT_ID]
            if GEN_AI_AGENT_NAME in attributes:
                context["agent_name"] = attributes[GEN_AI_AGENT_NAME]
            if GEN_AI_AGENT_DESCRIPTION in attributes:
                context["agent_description"] = attributes[GEN_AI_AGENT_DESCRIPTION]

            # Extract tool information
            if GEN_AI_TOOL_NAME in attributes:
                context["tool_name"] = attributes[GEN_AI_TOOL_NAME]
            if GEN_AI_TOOL_DESCRIPTION in attributes:
                context["tool_description"] = attributes[GEN_AI_TOOL_DESCRIPTION]

            # Extract operation information
            if GEN_AI_OPERATION_NAME in attributes:
                context["operation_type"] = attributes[GEN_AI_OPERATION_NAME]

        # Set framework context for provider instrumentation
        self.context_manager.alignx_context.set_framework_context(**context)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the AutoGen instrumentation."""
        metadata = {
            "strategy": self.name,
            "framework": self.framework_name,
            "available": self.is_available(),
            "instrumented": self._instrumented,
        }

        if AUTOGEN_AVAILABLE:
            try:
                import autogen_core

                metadata["autogen_core_version"] = autogen_core.__version__
            except (ImportError, AttributeError):
                metadata["autogen_core_version"] = "unknown"

            try:
                import autogen_agentchat

                metadata["autogen_agentchat_version"] = autogen_agentchat.__version__
            except (ImportError, AttributeError):
                metadata["autogen_agentchat_version"] = "unknown"

        # Add current context information
        metadata["current_context"] = self.context_manager.get_current_context()

        return metadata
