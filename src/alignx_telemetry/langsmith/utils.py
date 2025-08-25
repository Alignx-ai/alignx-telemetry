"""Utility functions for AlignX LangSmith tracing."""

import json
import logging
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def ensure_uuid(value: Any) -> UUID:
    """Ensure a value is a UUID."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ValueError(f"Cannot convert {value} to UUID")


def dumps_json(obj: Any) -> str:
    """Dump an object to JSON with UUID serialization support."""

    def json_serializer(obj: Any) -> str:
        if isinstance(obj, UUID):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(obj, default=json_serializer, ensure_ascii=False)


def sanitize_for_otel(value: Any) -> Any:
    """Sanitize a value for OpenTelemetry attributes."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [sanitize_for_otel(item) for item in value]
    if isinstance(value, dict):
        return {k: sanitize_for_otel(v) for k, v in value.items()}
    # Convert other types to strings
    return str(value)


def get_runtime_and_metrics() -> dict[str, Any]:
    """Get basic runtime information for tracing."""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
    }


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    try:
        import opentelemetry

        return True
    except ImportError:
        return False


def safe_serialize(obj: Any, max_depth: int = 10) -> Any:
    """Safely serialize an object, handling circular references."""
    if max_depth <= 0:
        return f"<max_depth_exceeded: {type(obj).__name__}>"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, UUID):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item, max_depth - 1) for item in obj]

    if isinstance(obj, dict):
        return {str(k): safe_serialize(v, max_depth - 1) for k, v in obj.items()}

    # For other objects, try to get useful attributes
    try:
        if hasattr(obj, "__dict__"):
            return {
                "__type__": type(obj).__name__,
                "__dict__": safe_serialize(obj.__dict__, max_depth - 1),
            }
        else:
            return str(obj)
    except Exception as e:
        return f"<serialization_error: {type(obj).__name__}: {e}>"


class ContextThreadPoolExecutor:
    """Simple context-aware thread pool executor placeholder."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the executor."""
        self.max_workers = max_workers or 4

    def submit(self, fn, *args, **kwargs):
        """Submit a function for execution."""
        # For now, just execute synchronously
        # In a full implementation, this would use threading
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing function {fn}: {e}")
            raise


def get_function_name(func) -> str:
    """Get a human-readable name for a function."""
    if hasattr(func, "__name__"):
        return func.__name__
    if hasattr(func, "__class__"):
        return func.__class__.__name__
    return str(func)
