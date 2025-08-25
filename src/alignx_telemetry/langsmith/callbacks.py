"""LangChain callback handlers for AlignX LangSmith integration - Enhanced for LangGraph support."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from alignx_telemetry.langsmith.run_helpers import (
    get_current_run_tree,
    _PARENT_RUN_TREE,
    _create_run_tree_for_function,
    get_callback_handler,
    tracing_context,
)
from alignx_telemetry.langsmith.run_trees import RunTree
from alignx_telemetry.langsmith import utils

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import (
        ChatGeneration,
        LLMResult,
        Generation,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Provide stubs for when LangChain is not available
    class BaseCallbackHandler:
        pass

    class BaseMessage:
        pass

    class ChatGeneration:
        pass

    class LLMResult:
        pass

    class Generation:
        pass


class AlignXLangChainCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that integrates with AlignX LangSmith tracing.

    Enhanced to support both LangChain and LangGraph execution patterns.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        graph: Optional[Any] = None,  # LangGraph Graph object
    ):
        """Initialize the callback handler.

        Args:
            project_name: Project name for runs.
            tags: Default tags to apply to runs.
            metadata: Default metadata to apply to runs.
            graph: Optional LangGraph Graph object to track graph definition.
        """
        super().__init__()
        self.project_name = project_name or "alignx-langchain"
        self.default_tags = tags or ["alignx", "langchain"]
        self.default_metadata = metadata or {}
        self.run_map: Dict[UUID, RunTree] = {}

        # Handle LangGraph graph definition
        if graph:
            try:
                # Try to extract graph definition in Mermaid format
                if hasattr(graph, "draw_mermaid"):
                    self.default_metadata["_alignx_graph_definition"] = {
                        "format": "mermaid",
                        "data": graph.draw_mermaid(),
                    }
                    self.default_tags.append("langgraph")
                elif hasattr(graph, "get_graph"):
                    # Handle compiled LangGraph apps
                    internal_graph = graph.get_graph(xray=True)
                    if hasattr(internal_graph, "draw_mermaid"):
                        self.default_metadata["_alignx_graph_definition"] = {
                            "format": "mermaid",
                            "data": internal_graph.draw_mermaid(),
                        }
                        self.default_tags.append("langgraph")
            except Exception as e:
                logger.debug(f"Failed to extract graph definition: {e}")

    def _get_run_type(self, serialized: Dict[str, Any]) -> str:
        """Determine run type from serialized information."""
        # Extract class information
        class_info = serialized.get("id", [])
        if isinstance(class_info, list) and class_info:
            class_name = class_info[-1].lower()

            # Special handling for LangGraph
            if "langgraph" in class_name:
                return "chain"  # LangGraph executions are chain-type
            # Map LangChain classes to run types
            elif "llm" in class_name or "chat" in class_name:
                return "llm"
            elif "tool" in class_name or "retriever" in class_name:
                return "tool"
            elif "prompt" in class_name:
                return "prompt"
            elif "parser" in class_name:
                return "parser"
            else:
                return "chain"

        return "chain"

    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Safely serialize inputs for storage."""
        try:
            return utils.safe_serialize(inputs)
        except Exception as e:
            logger.debug(f"Failed to serialize inputs: {e}")
            return {"error": f"serialization_failed: {e}"}

    def _serialize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Safely serialize outputs for storage."""
        try:
            return utils.safe_serialize(outputs)
        except Exception as e:
            logger.debug(f"Failed to serialize outputs: {e}")
            return {"error": f"serialization_failed: {e}"}

    def _get_run_name(self, serialized: Dict[str, Any]) -> str:
        """Extract run name from serialized information."""
        # Try to get name from various fields
        if "name" in serialized:
            return serialized["name"]

        # Extract from class path
        class_info = serialized.get("id", [])
        if isinstance(class_info, list) and class_info:
            return class_info[-1]

        return "Unknown"

    def _get_or_create_parent_run(
        self, parent_run_id: Optional[UUID]
    ) -> Optional[RunTree]:
        """Get parent run, with special handling for LangGraph context restoration."""
        if parent_run_id and parent_run_id in self.run_map:
            return self.run_map[parent_run_id]
        elif not parent_run_id:
            # For LangGraph, check if there's a current run tree in context
            current_run = get_current_run_tree()
            if current_run:
                return current_run
        return None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts running."""
        try:
            # Get parent run with enhanced context handling
            parent_run = self._get_or_create_parent_run(parent_run_id)

            # Create run tree
            run_tree = RunTree(
                id=run_id,
                name=self._get_run_name(serialized),
                run_type="llm",
                inputs=self._serialize_inputs({"prompts": prompts}),
                parent_run=parent_run,
                tags=(tags or []) + self.default_tags,
                project_name=self.project_name,
                extra={**(metadata or {}), **self.default_metadata},
            )

            self.run_map[run_id] = run_tree

            # Enhanced context management for LangGraph
            if not parent_run:
                _PARENT_RUN_TREE.set(run_tree)

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_start(run_tree)

        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chat model starts."""
        try:
            # Convert messages to serializable format
            serialized_messages = []
            for message_list in messages:
                serialized_list = []
                for msg in message_list:
                    if hasattr(msg, "dict"):
                        serialized_list.append(msg.dict())
                    else:
                        serialized_list.append(str(msg))
                serialized_messages.append(serialized_list)

            # Get parent run with enhanced context handling
            parent_run = self._get_or_create_parent_run(parent_run_id)

            # Create run tree
            run_tree = RunTree(
                id=run_id,
                name=self._get_run_name(serialized),
                run_type="llm",
                inputs=self._serialize_inputs({"messages": serialized_messages}),
                parent_run=parent_run,
                tags=(tags or []) + self.default_tags,
                project_name=self.project_name,
                extra={**(metadata or {}), **self.default_metadata},
            )

            self.run_map[run_id] = run_tree

            # Enhanced context management for LangGraph
            if not parent_run:
                _PARENT_RUN_TREE.set(run_tree)

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_start(run_tree)

        except Exception as e:
            logger.error(f"Error in on_chat_model_start: {e}", exc_info=True)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM ends successfully."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # Extract outputs from LLM result
            outputs = {"generations": []}
            if hasattr(response, "generations"):
                for generation_list in response.generations:
                    gen_outputs = []
                    for gen in generation_list:
                        if hasattr(gen, "dict"):
                            gen_outputs.append(gen.dict())
                        else:
                            gen_outputs.append({"text": str(gen)})
                    outputs["generations"].append(gen_outputs)

            # Add token usage if available
            if hasattr(response, "llm_output") and response.llm_output:
                if "token_usage" in response.llm_output:
                    outputs["token_usage"] = response.llm_output["token_usage"]

            # End the run
            run_tree.end(outputs=self._serialize_outputs(outputs))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_end(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM encounters an error."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # End the run with error
            run_tree.end(error=str(error))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_error(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_llm_error: {e}", exc_info=True)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain starts - enhanced for LangGraph support."""
        try:
            # Get parent run with enhanced context handling
            parent_run = self._get_or_create_parent_run(parent_run_id)

            # Determine if this is a LangGraph execution
            run_name = self._get_run_name(serialized)
            is_langgraph = run_name == "LangGraph" or "langgraph" in run_name.lower()

            # Enhanced metadata for LangGraph
            enhanced_metadata = {**(metadata or {}), **self.default_metadata}
            if is_langgraph:
                enhanced_metadata["execution_type"] = "langgraph"
                # Add state information if available
                if isinstance(inputs, dict) and "input" in inputs:
                    enhanced_metadata["state_keys"] = (
                        list(inputs["input"].keys())
                        if isinstance(inputs["input"], dict)
                        else None
                    )

            # Create run tree
            run_tree = RunTree(
                id=run_id,
                name=run_name,
                run_type=self._get_run_type(serialized),
                inputs=self._serialize_inputs(inputs),
                parent_run=parent_run,
                tags=(tags or []) + self.default_tags,
                project_name=self.project_name,
                extra=enhanced_metadata,
            )

            self.run_map[run_id] = run_tree

            # Enhanced context management for LangGraph
            if not parent_run:
                _PARENT_RUN_TREE.set(run_tree)

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_start(run_tree)

        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}", exc_info=True)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain ends successfully."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # Enhanced output processing for LangGraph
            processed_outputs = outputs
            if isinstance(outputs, dict) and "output" in outputs:
                # LangGraph often wraps outputs in an "output" key
                processed_outputs = outputs

            # End the run
            run_tree.end(outputs=self._serialize_outputs(processed_outputs))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_end(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}", exc_info=True)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain encounters an error."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # End the run with error
            run_tree.end(error=str(error))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_error(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}", exc_info=True)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts."""
        try:
            # Get parent run with enhanced context handling
            parent_run = self._get_or_create_parent_run(parent_run_id)

            # Create run tree
            run_tree = RunTree(
                id=run_id,
                name=self._get_run_name(serialized),
                run_type="tool",
                inputs=self._serialize_inputs({"input": input_str}),
                parent_run=parent_run,
                tags=(tags or []) + self.default_tags,
                project_name=self.project_name,
                extra={**(metadata or {}), **self.default_metadata},
            )

            self.run_map[run_id] = run_tree

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_start(run_tree)

        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}", exc_info=True)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends successfully."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # End the run
            run_tree.end(outputs=self._serialize_outputs({"output": output}))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_end(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}", exc_info=True)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool encounters an error."""
        try:
            if run_id not in self.run_map:
                logger.warning(f"Run {run_id} not found in run_map")
                return

            run_tree = self.run_map[run_id]

            # End the run with error
            run_tree.end(error=str(error))

            # Notify callback handler
            callback_handler = get_callback_handler()
            if callback_handler:
                callback_handler.on_run_error(run_tree)

            # Clean up
            del self.run_map[run_id]

        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}", exc_info=True)

    def flush(self):
        """Flush any pending operations - compatibility method."""
        # For compatibility with tests and other integrations
        pass

    def created_traces(self):
        """Return list of created traces - compatibility method."""
        # For compatibility with tests
        return []
