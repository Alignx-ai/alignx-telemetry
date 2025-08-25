"""Decorator and helpers for creating run trees from functions - Enhanced AlignX LangSmith implementation."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import uuid
import warnings
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import copy_context
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import ParamSpec

from alignx_telemetry.langsmith import run_trees, schemas, utils
from alignx_telemetry.langsmith._internal import _otel_utils

if TYPE_CHECKING:
    from types import TracebackType

LOGGER = logging.getLogger(__name__)

# Context variables for distributed tracing
_PARENT_RUN_TREE = contextvars.ContextVar[Optional[run_trees.RunTree]](
    "_PARENT_RUN_TREE", default=None
)
_PROJECT_NAME = contextvars.ContextVar[Optional[str]]("_PROJECT_NAME", default=None)
_TAGS = contextvars.ContextVar[Optional[list[str]]]("_TAGS", default=None)
_METADATA = contextvars.ContextVar[Optional[dict[str, Any]]]("_METADATA", default=None)
_TRACING_ENABLED = contextvars.ContextVar[Optional[Union[bool, Literal["local"]]]](
    "_TRACING_ENABLED", default=None
)

_CONTEXT_KEYS: dict[str, contextvars.ContextVar] = {
    "parent": _PARENT_RUN_TREE,
    "project_name": _PROJECT_NAME,
    "tags": _TAGS,
    "metadata": _METADATA,
    "enabled": _TRACING_ENABLED,
    "replicas": run_trees._REPLICAS,
    "distributed_parent_id": run_trees._DISTRIBUTED_PARENT_ID,
}

P = ParamSpec("P")
T = TypeVar("T")

# Valid run types
_VALID_RUN_TYPES = {
    "chain", "llm", "tool", "prompt", "retriever", "embedding", "parser"
}


def get_current_run_tree() -> Optional[run_trees.RunTree]:
    """Get the current run tree from context."""
    return _PARENT_RUN_TREE.get()


def get_tracing_context(
    context: Optional[contextvars.Context] = None,
) -> dict[str, Any]:
    """Get the current tracing context."""
    if context is None:
        return {
            "parent": _PARENT_RUN_TREE.get(),
            "project_name": _PROJECT_NAME.get(),
            "tags": _TAGS.get(),
            "metadata": _METADATA.get(),
            "enabled": _TRACING_ENABLED.get(),
            "replicas": run_trees._REPLICAS.get(),
            "distributed_parent_id": run_trees._DISTRIBUTED_PARENT_ID.get(),
        }

    # Extract from provided context
    result = {}
    for key, context_var in _CONTEXT_KEYS.items():
        try:
            result[key] = context.get(context_var)
        except LookupError:
            result[key] = None
    return result


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    enabled = _TRACING_ENABLED.get()
    if enabled is None:
        # Check environment variables
        import os
        env_enabled = os.getenv("LANGSMITH_TRACING", "").lower()
        if env_enabled in ("true", "1", "yes"):
            return True
        # Default to enabled if AlignX telemetry is available
        return True
    if isinstance(enabled, bool):
        return enabled
    return enabled == "local"


def _get_parent_run(context: dict[str, Any]) -> Optional[run_trees.RunTree]:
    """Extract parent run from context."""
    parent = context.get("parent")
    if isinstance(parent, run_trees.RunTree):
        return parent
    elif isinstance(parent, dict):
        # Handle distributed tracing headers
        return None  # For now, we'll handle this later
    elif isinstance(parent, str):
        # Handle dotted order string
        return None  # For now, we'll handle this later
    return None


@contextlib.contextmanager
def tracing_context(
    *,
    enabled: Optional[bool] = None,
    project_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    parent: Optional[Union[run_trees.RunTree, Mapping, str]] = None,
    replicas: Optional[Sequence[run_trees.WriteReplica]] = None,
    distributed_parent_id: Optional[str] = None,
    **kwargs: Any,
):
    """Context manager for setting tracing configuration.

    Args:
        enabled: Whether tracing is enabled in this context.
        project_name: Project name for runs in this context.
        tags: Tags to apply to runs in this context.
        metadata: Metadata to apply to runs in this context.
        parent: Parent run tree for runs in this context.
        replicas: Write replicas for distributed tracing.
        distributed_parent_id: Distributed parent ID for tracing.
        **kwargs: Additional arguments (for compatibility).
    """
    if kwargs:
        warnings.warn(
            f"Unrecognized keyword arguments: {kwargs}.",
            DeprecationWarning,
        )

    current_context = get_tracing_context()
    parent_run = _get_parent_run({"parent": parent}) if parent else None
    
    # Store current values and set new ones
    tokens = []

    if enabled is not None:
        tokens.append(_TRACING_ENABLED.set(enabled))
    if project_name is not None:
        tokens.append(_PROJECT_NAME.set(project_name))
    if tags is not None:
        tokens.append(_TAGS.set(tags))
    if metadata is not None:
        tokens.append(_METADATA.set(metadata))
    if parent_run is not None:
        tokens.append(_PARENT_RUN_TREE.set(parent_run))
    if replicas is not None:
        tokens.append(run_trees._REPLICAS.set(replicas))
    if distributed_parent_id is not None:
        tokens.append(run_trees._DISTRIBUTED_PARENT_ID.set(distributed_parent_id))

    try:
        yield
    finally:
        # Reset all tokens
        for token in reversed(tokens):
            token.__exit__(None, None, None)


class TracingCallbackHandler:
    """Callback handler for integrating with AlignX telemetry."""

    def __init__(self, telemetry_manager=None):
        """Initialize with optional telemetry manager for AlignX integration."""
        self.telemetry_manager = telemetry_manager

    def on_run_start(self, run_tree: run_trees.RunTree) -> None:
        """Called when a run starts."""
        if not is_tracing_enabled():
            return

        # Create OpenTelemetry span if telemetry manager is available
        if self.telemetry_manager and _otel_utils.is_otel_available():
            try:
                span = _otel_utils.create_otel_span_from_run_tree(run_tree)
                if span:
                    # Store span reference for later
                    run_tree._otel_span = span
            except Exception as e:
                LOGGER.debug(f"Failed to create OpenTelemetry span: {e}")

    def on_run_end(self, run_tree: run_trees.RunTree) -> None:
        """Called when a run ends successfully."""
        if not is_tracing_enabled():
            return

        # End OpenTelemetry span if available
        if hasattr(run_tree, "_otel_span") and run_tree._otel_span:
            try:
                _otel_utils.finalize_otel_span(run_tree._otel_span, run_tree)
            except Exception as e:
                LOGGER.debug(f"Failed to end OpenTelemetry span: {e}")

    def on_run_error(self, run_tree: run_trees.RunTree) -> None:
        """Called when a run errors."""
        if not is_tracing_enabled():
            return

        # End OpenTelemetry span with error if available
        if hasattr(run_tree, "_otel_span") and run_tree._otel_span:
            try:
                _otel_utils.finalize_otel_span(run_tree._otel_span, run_tree)
            except Exception as e:
                LOGGER.debug(f"Failed to end OpenTelemetry span with error: {e}")


# Global callback handler
_CALLBACK_HANDLER: Optional[TracingCallbackHandler] = None


def set_callback_handler(handler: TracingCallbackHandler) -> None:
    """Set the global callback handler."""
    global _CALLBACK_HANDLER
    _CALLBACK_HANDLER = handler


def get_callback_handler() -> Optional[TracingCallbackHandler]:
    """Get the current callback handler."""
    return _CALLBACK_HANDLER


def _get_run_name(func: Callable, name: Optional[str] = None) -> str:
    """Get the name for a run from a function."""
    if name:
        return name
    return utils.get_function_name(func)


def _get_run_type_for_function(func: Callable) -> str:
    """Infer run type from function characteristics."""
    func_name = utils.get_function_name(func).lower()

    # Simple heuristics for run type detection
    if any(
        keyword in func_name for keyword in ["llm", "chat", "completion", "generate"]
    ):
        return "llm"
    elif any(keyword in func_name for keyword in ["tool", "search", "retriev"]):
        return "tool"
    elif any(keyword in func_name for keyword in ["chain", "workflow", "pipeline"]):
        return "chain"
    else:
        return "chain"  # Default to chain


def _create_run_tree_for_function(
    func: Callable,
    args: tuple,
    kwargs: dict,
    name: Optional[str] = None,
    run_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    project_name: Optional[str] = None,
    process_inputs: Optional[Callable[[dict], dict]] = None,
    langsmith_extra: Optional[dict] = None,
) -> run_trees.RunTree:
    """Create a run tree for a function call."""
    # Get current context
    parent_run = get_current_run_tree()
    current_project = project_name or _PROJECT_NAME.get()
    current_tags = list(tags or [])
    current_tags.extend(_TAGS.get() or [])
    current_metadata = dict(_METADATA.get() or {})
    current_metadata.update(metadata or {})
    
    # Handle langsmith_extra if provided
    if langsmith_extra:
        current_tags.extend(langsmith_extra.get("tags", []))
        current_metadata.update(langsmith_extra.get("metadata", {}))
        if langsmith_extra.get("project_name"):
            current_project = langsmith_extra["project_name"]

    # Determine inputs
    inputs = {}
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)

        # Process inputs if custom processor provided
        if process_inputs:
            try:
                inputs = process_inputs(inputs)
            except Exception as e:
                LOGGER.debug(f"Failed to process inputs: {e}")

        # Serialize inputs safely
        inputs = utils.safe_serialize(inputs)
    except Exception as e:
        LOGGER.debug(f"Failed to capture inputs for {func}: {e}")
        inputs = {
            "args": utils.safe_serialize(args),
            "kwargs": utils.safe_serialize(kwargs),
        }

    # Create run tree
    run_tree = run_trees.RunTree(
        name=_get_run_name(func, name),
        run_type=run_type or _get_run_type_for_function(func),
        inputs=inputs,
        parent_run=parent_run,
        tags=current_tags,
        project_name=current_project,
        extra=current_metadata,
    )

    return run_tree


@overload
def traceable(
    func: Callable[P, T],
) -> Callable[P, T]: ...


@overload
def traceable(
    run_type: str = "chain",
    *,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    project_name: Optional[str] = None,
    process_inputs: Optional[Callable[[dict], dict]] = None,
    process_outputs: Optional[Callable[..., dict]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def traceable(
    *args: Any,
    **kwargs: Any,
) -> Union[Callable[P, T], Callable[[Callable[P, T]], Callable[P, T]]]:
    """Trace a function with AlignX LangSmith.

    Args:
        run_type: The type of run to create. Examples: llm, chain, tool, prompt,
            retriever, etc. Defaults to "chain".
        name: The name of the run. Defaults to the function name.
        metadata: The metadata to add to the run. Defaults to None.
        tags: The tags to add to the run. Defaults to None.
        project_name: The name of the project to log the run to. Defaults to None.
        process_inputs: Custom serialization / processing function for inputs.
        process_outputs: Custom serialization / processing function for outputs.

    Returns:
        Decorated function or decorator.

    Examples:
        Basic usage:

        .. code-block:: python

            @traceable
            def my_function(x: float, y: float) -> float:
                return x + y

            my_function(5, 6)

        Specifying a run type and name:

        .. code-block:: python

            @traceable(name="CustomName", run_type="tool")
            def another_function(a: float, b: float) -> float:
                return a * b

            another_function(5, 6)

        With custom metadata and tags:

        .. code-block:: python

            @traceable(
                metadata={"version": "1.0"}, 
                tags=["beta", "test"]
            )
            def tagged_function(x):
                return x**2

            tagged_function(5)

        Manually passing langsmith_extra:

        .. code-block:: python

            @traceable
            def manual_extra_function(x):
                return x**2

            manual_extra_function(
                5, 
                langsmith_extra={"metadata": {"version": "1.0"}}
            )
    """
    run_type = cast(
        str,
        (
            args[0]
            if args and isinstance(args[0], str)
            else (kwargs.pop("run_type", None) or "chain")
        ),
    )
    
    if run_type not in _VALID_RUN_TYPES:
        warnings.warn(
            f"Unrecognized run_type: {run_type}. Must be one of: {_VALID_RUN_TYPES}. "
            f"Did you mean @traceable(name='{run_type}')?"
        )
    
    if len(args) > 1:
        warnings.warn(
            "The `traceable()` decorator only accepts one positional argument, "
            "which should be the run_type. All other arguments should be passed "
            "as keyword arguments."
        )

    name = kwargs.pop("name", None)
    metadata = kwargs.pop("metadata", None)
    tags = kwargs.pop("tags", None)
    project_name = kwargs.pop("project_name", None)
    process_inputs = kwargs.pop("process_inputs", None)
    process_outputs = kwargs.pop("process_outputs", None)

    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> T:
                if not is_tracing_enabled():
                    return await f(*func_args, **func_kwargs)

                # Extract langsmith_extra if provided
                langsmith_extra = func_kwargs.pop("langsmith_extra", None)

                run_tree = _create_run_tree_for_function(
                    f, func_args, func_kwargs, name, run_type, tags, 
                    metadata, project_name, process_inputs, langsmith_extra
                )

                # Set as current run
                token = _PARENT_RUN_TREE.set(run_tree)

                # Notify callback handler
                if _CALLBACK_HANDLER:
                    _CALLBACK_HANDLER.on_run_start(run_tree)

                try:
                    result = await f(*func_args, **func_kwargs)
                    
                    # Process outputs if custom processor provided
                    outputs = result
                    if process_outputs:
                        try:
                            outputs = process_outputs(result)
                        except Exception as e:
                            LOGGER.debug(f"Failed to process outputs: {e}")
                    
                    # Capture outputs
                    run_tree.end(outputs=utils.safe_serialize(outputs))

                    # Notify callback handler
                    if _CALLBACK_HANDLER:
                        _CALLBACK_HANDLER.on_run_end(run_tree)

                    return result
                except Exception as e:
                    run_tree.end(error=str(e))

                    # Notify callback handler
                    if _CALLBACK_HANDLER:
                        _CALLBACK_HANDLER.on_run_error(run_tree)

                    raise
                finally:
                    _PARENT_RUN_TREE.reset(token)

            return async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> T:
                if not is_tracing_enabled():
                    return f(*func_args, **func_kwargs)

                # Extract langsmith_extra if provided
                langsmith_extra = func_kwargs.pop("langsmith_extra", None)

                run_tree = _create_run_tree_for_function(
                    f, func_args, func_kwargs, name, run_type, tags, 
                    metadata, project_name, process_inputs, langsmith_extra
                )

                # Set as current run
                token = _PARENT_RUN_TREE.set(run_tree)

                # Notify callback handler
                if _CALLBACK_HANDLER:
                    _CALLBACK_HANDLER.on_run_start(run_tree)

                try:
                    result = f(*func_args, **func_kwargs)
                    
                    # Process outputs if custom processor provided
                    outputs = result
                    if process_outputs:
                        try:
                            outputs = process_outputs(result)
                        except Exception as e:
                            LOGGER.debug(f"Failed to process outputs: {e}")
                    
                    # Capture outputs
                    run_tree.end(outputs=utils.safe_serialize(outputs))

                    # Notify callback handler
                    if _CALLBACK_HANDLER:
                        _CALLBACK_HANDLER.on_run_end(run_tree)

                    return result
                except Exception as e:
                    run_tree.end(error=str(e))

                    # Notify callback handler
                    if _CALLBACK_HANDLER:
                        _CALLBACK_HANDLER.on_run_error(run_tree)

                    raise
                finally:
                    _PARENT_RUN_TREE.reset(token)

            return sync_wrapper

    # Handle both @traceable and @traceable() usage
    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], str):
        # Direct decoration: @traceable
        return decorator(args[0])
    else:
        # Parameterized decoration: @traceable(...)
        return decorator


# Alias for backward compatibility
trace = traceable