from typing import Dict, Any, Optional, Callable, Type
from alignx_telemetry.strategy.strategies import (
    InstrumentationStrategy,
    StandardInstrumentationStrategy,
)
import logging

# Import the ASGI middleware for correlation processing
from alignx_telemetry.middleware import AlignXCorrelationMiddleware
from alignx_telemetry.trace_correlation import AlignXTraceCorrelator

# Logger for web framework strategy
logger = logging.getLogger(__name__)


class CorrelationMixin:
    """Mixin that adds correlation processing capabilities to instrumentation strategies.

    This mixin uses the centralized AlignXTraceCorrelator instead of duplicating code.
    """

    def __init__(self):
        super().__init__()
        self._trace_correlator = AlignXTraceCorrelator()

    def _create_correlation_request_hook(
        self, original_hook: Optional[Callable] = None
    ) -> Callable:
        """Create a request hook using the centralized trace correlator.

        Args:
            original_hook: Original request hook to chain (optional)

        Returns:
            Enhanced request hook function from the centralized correlator
        """
        return self._trace_correlator.create_correlation_request_hook(original_hook)

    def _create_correlation_response_hook(
        self, original_hook: Optional[Callable] = None
    ) -> Callable:
        """Create a response hook using the centralized trace correlator.

        Args:
            original_hook: Original response hook to chain (optional)

        Returns:
            Enhanced response hook function from the centralized correlator
        """
        return self._trace_correlator.create_correlation_response_hook(original_hook)


class EnhancedFastAPIStrategy(CorrelationMixin, InstrumentationStrategy):
    """Enhanced FastAPI strategy with automatic correlation processing."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        app = kwargs.get("app")
        if not app:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: 'app' parameter is required"
            )
            return False

        try:
            # Add correlation middleware BEFORE instrumenting
            # This ensures correlation context is set before OpenTelemetry instrumentation runs
            app.add_middleware(AlignXCorrelationMiddleware)

            # Now instrument the app - correlation context will already be set
            instrumentor_cls.instrument_app(app, **config)

            logger.info(
                "Successfully instrumented FastAPI with AlignX correlation middleware"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: {e}"
            )
            return False


class EnhancedFlaskStrategy(CorrelationMixin, InstrumentationStrategy):
    """Enhanced Flask strategy with automatic correlation processing."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        app = kwargs.get("app")
        if not app:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: 'app' parameter is required"
            )
            return False

        try:
            # Auto-inject correlation hook using centralized correlator
            original_hook = config.get("request_hook")
            config["request_hook"] = self._create_correlation_request_hook(
                original_hook
            )

            # Also add response hook if not present
            if "response_hook" not in config:
                config["response_hook"] = self._create_correlation_response_hook()

            instrumentor = instrumentor_cls()
            instrumentor.instrument_app(app, **config)
            logger.info(
                "Successfully instrumented Flask with automatic correlation processing"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: {e}"
            )
            return False


class EnhancedStarletteStrategy(CorrelationMixin, InstrumentationStrategy):
    """Enhanced Starlette strategy with automatic correlation processing."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        app = kwargs.get("app")
        if not app:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: 'app' parameter is required"
            )
            return False

        try:
            # Auto-inject correlation hook using centralized correlator
            original_hook = config.get("server_request_hook")
            config["server_request_hook"] = self._create_correlation_request_hook(
                original_hook
            )

            # Also add response hook if not present
            if "server_response_hook" not in config:
                config["server_response_hook"] = (
                    self._create_correlation_response_hook()
                )

            instrumentor_cls.instrument_app(app, **config)
            logger.info(
                "Successfully instrumented Starlette with automatic correlation processing"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: {e}"
            )
            return False


class AutoInstrumentationStrategy(CorrelationMixin, StandardInstrumentationStrategy):
    """Enhanced standard strategy that auto-detects and instruments web frameworks with correlation."""

    def instrument(
        self, instrumentor_cls: Type[Any], config: Dict[str, Any], **kwargs
    ) -> bool:
        try:
            # Try to auto-detect if this is a web framework instrumentor that supports hooks
            instrumentor = instrumentor_cls()

            # Check if this instrumentor supports request hooks
            if hasattr(instrumentor, "instrument") and self._supports_request_hooks(
                instrumentor_cls
            ):
                # Auto-inject correlation hook for supported frameworks using centralized correlator
                original_hook = config.get("server_request_hook") or config.get(
                    "request_hook"
                )
                hook_key = (
                    "server_request_hook"
                    if "server_request_hook" in str(instrumentor_cls)
                    else "request_hook"
                )
                config[hook_key] = self._create_correlation_request_hook(original_hook)

                # Add response hook too
                response_hook_key = (
                    "server_response_hook"
                    if "server_request_hook" in str(instrumentor_cls)
                    else "response_hook"
                )
                if response_hook_key not in config:
                    config[response_hook_key] = self._create_correlation_response_hook()

                logger.info(
                    f"Auto-injected correlation processing for {instrumentor_cls.__name__}"
                )

            instrumentor.instrument(**config)
            return True
        except Exception as e:
            logger.warning(
                f"Failed to instrument with {instrumentor_cls.__name__}: {e}"
            )
            return False

    def _supports_request_hooks(self, instrumentor_cls: Type[Any]) -> bool:
        """Check if an instrumentor supports request hooks for correlation."""
        instrumentor_name = instrumentor_cls.__name__.lower()
        # List of instrumentors that support request hooks
        supported = ["fastapi", "flask", "starlette", "django", "tornado", "aiohttp"]
        return any(framework in instrumentor_name for framework in supported)
