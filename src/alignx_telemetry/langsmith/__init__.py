"""AlignX LangSmith Tracing Core - Stripped down version focused on tracing only."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alignx_telemetry.langsmith.run_helpers import (
        get_current_run_tree,
        get_tracing_context,
        trace,
        traceable,
        tracing_context,
    )
    from alignx_telemetry.langsmith.run_trees import RunTree

# Version - aligned with AlignX telemetry
__version__ = "1.0.0-alignx"
version = __version__


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return version
    elif name == "RunTree":
        from alignx_telemetry.langsmith.run_trees import RunTree

        return RunTree
    elif name == "trace":
        from alignx_telemetry.langsmith.run_helpers import trace

        return trace
    elif name == "traceable":
        from alignx_telemetry.langsmith.run_helpers import traceable

        return traceable
    elif name == "tracing_context":
        from alignx_telemetry.langsmith.run_helpers import tracing_context

        return tracing_context
    elif name == "get_tracing_context":
        from alignx_telemetry.langsmith.run_helpers import get_tracing_context

        return get_tracing_context
    elif name == "get_current_run_tree":
        from alignx_telemetry.langsmith.run_helpers import get_current_run_tree

        return get_current_run_tree
    elif name == "configure":
        from alignx_telemetry.langsmith.run_trees import configure

        return configure

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RunTree",
    "__version__",
    "traceable",
    "trace",
    "tracing_context",
    "get_tracing_context",
    "get_current_run_tree",
    "configure",
]
