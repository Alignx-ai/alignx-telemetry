"""Instrumentation for auto-instrumenting supported libraries with OpenTelemetry."""

from alignx_telemetry.instrumentation.instrumentators import (
    instrument_all,
    instrument_specific,
    get_instrumentors,
    is_instrumentation_supported,
)

__all__ = [
    "instrument_all",
    "instrument_specific",
    "get_instrumentors",
    "is_instrumentation_supported",
]
