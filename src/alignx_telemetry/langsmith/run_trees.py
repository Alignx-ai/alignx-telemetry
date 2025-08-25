"""Core RunTree implementation for AlignX LangSmith tracing - without API client dependencies."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Union, cast
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

from typing_extensions import TypedDict

try:
    from pydantic.v1 import Field, root_validator  # type: ignore[import]
except ImportError:
    from pydantic import (  # type: ignore[assignment, no-redef]
        Field,
        root_validator,
    )

from alignx_telemetry.langsmith import schemas as ls_schemas
from alignx_telemetry.langsmith import utils

logger = logging.getLogger(__name__)


class WriteReplica(TypedDict, total=False):
    """Configuration for write replicas."""

    project_name: Optional[str]
    updates: Optional[dict]


# LangSmith constants
LANGSMITH_PREFIX = "langsmith-"
LANGSMITH_DOTTED_ORDER = sys.intern(f"{LANGSMITH_PREFIX}trace")
LANGSMITH_DOTTED_ORDER_BYTES = LANGSMITH_DOTTED_ORDER.encode("utf-8")
LANGSMITH_METADATA = sys.intern(f"{LANGSMITH_PREFIX}metadata")
LANGSMITH_TAGS = sys.intern(f"{LANGSMITH_PREFIX}tags")
LANGSMITH_PROJECT = sys.intern(f"{LANGSMITH_PREFIX}project")
LANGSMITH_REPLICAS = sys.intern(f"{LANGSMITH_PREFIX}replicas")
OVERRIDE_OUTPUTS = sys.intern("__omit_auto_outputs")
NOT_PROVIDED = cast(None, object())

_LOCK = threading.Lock()

# Context variables for distributed tracing
_REPLICAS = contextvars.ContextVar[Optional[Sequence[WriteReplica]]](
    "_REPLICAS", default=None
)

_DISTRIBUTED_PARENT_ID = contextvars.ContextVar[Optional[str]](
    "_DISTRIBUTED_PARENT_ID", default=None
)

_SENTINEL = cast(None, object())
TIMESTAMP_LENGTH = 36


def configure(
    enabled: Optional[bool] = _SENTINEL,
    project_name: Optional[str] = _SENTINEL,
    tags: Optional[list[str]] = _SENTINEL,
    metadata: Optional[dict[str, Any]] = _SENTINEL,
):
    """Configure global AlignX LangSmith tracing context.

    This function allows you to set global configuration options for LangSmith
    tracing that will be applied to all subsequent traced operations.

    Args:
        enabled: Whether tracing is enabled.
        project_name: Default project name for runs.
        tags: Default tags to apply to all runs.
        metadata: Default metadata to apply to all runs.
    """
    from alignx_telemetry.langsmith import run_helpers

    # Set up context variables for tracing configuration
    if enabled is not _SENTINEL:
        run_helpers._TRACING_ENABLED.set(enabled)
    if project_name is not _SENTINEL:
        run_helpers._PROJECT_NAME.set(project_name)
    if tags is not _SENTINEL:
        run_helpers._TAGS.set(tags)
    if metadata is not _SENTINEL:
        run_helpers._METADATA.set(metadata)


class RunTree:
    """A run tree represents a single run or operation, which can have child runs.

    This is a simplified version that focuses on tracing structure without API dependencies.
    """

    def __init__(
        self,
        id: Optional[Union[UUID, str]] = None,
        name: Optional[str] = None,
        inputs: Optional[dict] = None,
        run_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        extra: Optional[dict] = None,
        error: Optional[str] = None,
        outputs: Optional[dict] = None,
        parent_run: Optional[RunTree] = None,
        child_runs: Optional[list[RunTree]] = None,
        tags: Optional[list[str]] = None,
        trace_id: Optional[Union[UUID, str]] = None,
        dotted_order: Optional[str] = None,
        project_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize a RunTree.

        Args:
            id: Unique identifier for the run.
            name: Name of the run.
            inputs: Input data for the run.
            run_type: Type of run (llm, chain, tool, etc.).
            start_time: When the run started.
            end_time: When the run ended.
            extra: Additional metadata.
            error: Error message if the run failed.
            outputs: Output data from the run.
            parent_run: Parent RunTree if this is a child run.
            child_runs: List of child RunTrees.
            tags: Tags associated with the run.
            trace_id: Trace identifier.
            dotted_order: Hierarchical ordering string.
            project_name: Project name for the run.
            **kwargs: Additional arguments.
        """
        self.id = UUID(id) if isinstance(id, str) else (id or uuid4())
        self.name = name or "Unknown"
        self.inputs = inputs or {}
        self.run_type = run_type or "chain"
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = end_time
        self.extra = extra or {}
        self.error = error
        self.outputs = outputs
        self.parent_run = parent_run
        self.child_runs = child_runs or []
        self.tags = tags or []
        self.trace_id = (
            UUID(trace_id) if isinstance(trace_id, str) else (trace_id or self.id)
        )
        self.dotted_order = dotted_order or self._generate_dotted_order()
        self.project_name = project_name

        # Track in parent if provided
        if self.parent_run:
            self.parent_run.child_runs.append(self)

    def _generate_dotted_order(self) -> str:
        """Generate a dotted order string for hierarchical ordering."""
        if self.parent_run:
            parent_order = self.parent_run.dotted_order or ""
            child_index = len(self.parent_run.child_runs)
            return (
                f"{parent_order}.{child_index:010d}"
                if parent_order
                else f"{child_index:010d}"
            )
        else:
            return f"{int(self.start_time.timestamp() * 1000000):010d}"

    def add_child(self, child: RunTree) -> None:
        """Add a child run to this run."""
        child.parent_run = self
        child.trace_id = self.trace_id
        child.dotted_order = child._generate_dotted_order()
        self.child_runs.append(child)

    def end(
        self,
        outputs: Optional[dict] = None,
        error: Optional[str] = None,
        end_time: Optional[datetime] = None,
    ) -> RunTree:
        """End the run and set outputs or error.

        Args:
            outputs: Output data from the run.
            error: Error message if the run failed.
            end_time: When the run ended.

        Returns:
            Self for method chaining.
        """
        self.end_time = end_time or datetime.now(timezone.utc)
        if outputs is not None:
            self.outputs = outputs
        if error is not None:
            self.error = error
        return self

    def create_child(
        self,
        name: str,
        run_type: str,
        inputs: Optional[dict] = None,
        **kwargs: Any,
    ) -> RunTree:
        """Create a child run.

        Args:
            name: Name of the child run.
            run_type: Type of the child run.
            inputs: Input data for the child run.
            **kwargs: Additional arguments.

        Returns:
            New child RunTree.
        """
        child = RunTree(
            name=name,
            run_type=run_type,
            inputs=inputs,
            parent_run=self,
            trace_id=self.trace_id,
            project_name=self.project_name,
            **kwargs,
        )
        return child

    def to_dict(self) -> dict[str, Any]:
        """Convert the run tree to a dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "inputs": self.inputs,
            "run_type": self.run_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "extra": self.extra,
            "error": self.error,
            "outputs": self.outputs,
            "parent_run_id": str(self.parent_run.id) if self.parent_run else None,
            "child_run_ids": [str(child.id) for child in self.child_runs],
            "tags": self.tags,
            "trace_id": str(self.trace_id),
            "dotted_order": self.dotted_order,
            "project_name": self.project_name,
        }

    def to_opentelemetry_span_attributes(self) -> dict[str, Any]:
        """Convert run data to OpenTelemetry span attributes."""
        attributes = {
            "langsmith.run.id": str(self.id),
            "langsmith.run.name": self.name,
            "langsmith.run.type": self.run_type,
            "langsmith.trace.id": str(self.trace_id),
        }

        if self.project_name:
            attributes["langsmith.project.name"] = self.project_name

        if self.tags:
            attributes["langsmith.run.tags"] = json.dumps(self.tags)

        if self.parent_run:
            attributes["langsmith.parent.run.id"] = str(self.parent_run.id)

        # Add custom metadata
        if self.extra:
            for key, value in self.extra.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"langsmith.metadata.{key}"] = value
                else:
                    attributes[f"langsmith.metadata.{key}"] = json.dumps(value)

        return attributes

    def __enter__(self) -> RunTree:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and end the run."""
        if exc_type is not None:
            self.end(error=f"{exc_type.__name__}: {exc_val}")
        else:
            self.end()

    def __repr__(self) -> str:
        """String representation of the RunTree."""
        return (
            f"RunTree(id={self.id}, name='{self.name}', "
            f"run_type='{self.run_type}', children={len(self.child_runs)})"
        )
