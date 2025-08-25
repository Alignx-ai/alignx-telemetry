"""Core schemas for AlignX LangSmith tracing - stripped down version."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)
from uuid import UUID

from typing_extensions import NotRequired, TypedDict

try:
    from pydantic.v1 import (
        BaseModel,
        Field,  # type: ignore[import]
        PrivateAttr,
        StrictBool,
        StrictFloat,
        StrictInt,
    )
except ImportError:
    from pydantic import (  # type: ignore[assignment]
        BaseModel,
        Field,
        PrivateAttr,
        StrictBool,
        StrictFloat,
        StrictInt,
    )

from pathlib import Path
from typing_extensions import Literal

SCORE_TYPE = Union[StrictBool, StrictInt, StrictFloat, None]
VALUE_TYPE = Union[dict, str, None]


class Attachment(NamedTuple):
    """Annotated type that will be stored as an attachment if used.

    Examples:
        Create an attachment from bytes:
            Attachment(("my_file.txt", b"file_content"))

        Create an attachment from a file path:
            Attachment(("my_file.txt", Path("path/to/file.txt")))

        Create an attachment with a specific MIME type:
            Attachment(("my_image.png", image_bytes, "image/png"))
    """

    name: str
    data: Union[bytes, str, Path]
    mime_type: Optional[str] = None


class RunTypeEnum(str, Enum):
    """Enum for run types."""

    tool = "tool"
    chain = "chain"
    llm = "llm"
    retriever = "retriever"
    embedding = "embedding"
    prompt = "prompt"
    parser = "parser"


# Minimal Run schema for tracing
class Run(BaseModel):
    """Run schema for storing and retrieving run information."""

    id: UUID = Field(..., description="The unique identifier for the run.")
    name: str = Field(..., description="The name of the run.")
    start_time: datetime = Field(..., description="The start time of the run.")
    run_type: RunTypeEnum = Field(..., description="The type of run.")

    # Optional fields
    end_time: Optional[datetime] = Field(None, description="The end time of the run.")
    extra: Optional[dict] = Field(None, description="Additional metadata for the run.")
    error: Optional[str] = Field(
        None, description="The error message if the run failed."
    )
    inputs: Optional[dict] = Field(None, description="The inputs to the run.")
    outputs: Optional[dict] = Field(None, description="The outputs of the run.")
    parent_run_id: Optional[UUID] = Field(None, description="The ID of the parent run.")
    trace_id: Optional[UUID] = Field(None, description="The trace ID for the run.")
    dotted_order: Optional[str] = Field(
        None, description="The dotted order for the run."
    )
    session_id: Optional[UUID] = Field(None, description="The session ID for the run.")
    tags: Optional[list[str]] = Field(None, description="The tags for the run.")


class RunWithAnnotationQueueInfo(Run):
    """Run with annotation queue information."""

    last_reviewed_time: Optional[datetime] = Field(
        None, description="The last time the run was reviewed."
    )
    added_at: Optional[datetime] = Field(
        None, description="The time the run was added to the queue."
    )


# Simple data structures for tracing
@runtime_checkable
class TracingCallbackHandler(Protocol):
    """Protocol for tracing callback handlers."""

    def on_run_start(self, run: Run) -> None:
        """Called when a run starts."""
        ...

    def on_run_end(self, run: Run) -> None:
        """Called when a run ends."""
        ...

    def on_run_error(self, run: Run) -> None:
        """Called when a run errors."""
        ...


# OpenTelemetry compatibility
class TraceableConfig(TypedDict, total=False):
    """Configuration for traceable functions."""

    name: Optional[str]
    run_type: Optional[str]
    tags: Optional[list[str]]
    metadata: Optional[dict[str, Any]]
    reduce_fn: Optional[Any]
    project_name: Optional[str]


# Common utility functions
def ensure_iso_format(dt: datetime) -> str:
    """Ensure a datetime is in ISO format with timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def parse_iso_format(iso_string: str) -> datetime:
    """Parse an ISO format string to a datetime."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))


# Error types for tracing
class TracingError(Exception):
    """Base exception for tracing errors."""

    pass


class RunTreeError(TracingError):
    """Exception raised when there's an error with run trees."""

    pass
