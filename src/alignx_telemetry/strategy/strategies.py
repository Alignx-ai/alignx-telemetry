"""Instrumentation strategies for different OpenTelemetry instrumentors."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type


# Logger for instrumentation strategies
logger = logging.getLogger(__name__)


class InstrumentationStrategy(ABC):
    """Abstract base class for instrumentation strategies."""

    @abstractmethod
    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        """Instrument using the provided instrumentor class and configuration.

        Args:
            instrumentor_cls: The instrumentor class to use
            config: Configuration dictionary
            **kwargs: Additional keyword arguments for specific strategies

        Returns:
            True if instrumentation was successful, False otherwise
        """
        pass


class StandardInstrumentationStrategy(InstrumentationStrategy):
    """Strategy for standard instrumentors that follow the pattern: Instrumentor().instrument(**config)."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        try:

            instrumentor = instrumentor_cls()
            instrumentor.instrument(**config)
            return True
        except Exception as e:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: {e}"
            )
            return False
