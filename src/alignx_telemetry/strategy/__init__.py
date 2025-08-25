"""Strategy for instrumenting libraries with OpenTelemetry."""

from alignx_telemetry.strategy.strategies import (
    InstrumentationStrategy,
    StandardInstrumentationStrategy,
)
from alignx_telemetry.strategy.alignx_langchain_strategy import (
    AlignXLangChainStrategy,
    AlignXLangGraphStrategy,
)
from alignx_telemetry.strategy.alignx_llamaindex_strategy import (
    AlignXLlamaIndexStrategy,
)
from alignx_telemetry.strategy.alignx_autogen_strategy import (
    AlignXAutoGenStrategy,
)
from alignx_telemetry.strategy.alignx_crewai_strategy import (
    AlignXCrewAIStrategy,
)

from alignx_telemetry.strategy.web_framework_strategy import (
    EnhancedFastAPIStrategy,
    EnhancedFlaskStrategy,
    EnhancedStarletteStrategy,
)


INSTRUMENTATION_STRATEGY_REGISTRY = {
    "default": StandardInstrumentationStrategy(),
    # Web frameworks
    "fastapi": EnhancedFastAPIStrategy(),
    "flask": EnhancedFlaskStrategy(),
    "starlette": EnhancedStarletteStrategy(),
    # LLM frameworks - AlignX custom implementations
    "langchain": AlignXLangChainStrategy(),
    "langgraph": AlignXLangGraphStrategy(),
    "llamaindex": AlignXLlamaIndexStrategy(),
    "autogen": AlignXAutoGenStrategy(),
    "crewai": AlignXCrewAIStrategy(),
}


def get_instrumentation_strategy(library_name: str) -> InstrumentationStrategy:
    """Get the appropriate instrumentation strategy for a library.

    Args:
        library_name: Name of the library

    Returns:
        The instrumentation strategy for the library
    """
    return INSTRUMENTATION_STRATEGY_REGISTRY.get(
        library_name, INSTRUMENTATION_STRATEGY_REGISTRY["default"]
    )


__all__ = [
    # Core functions
    "get_instrumentation_strategy",
]
