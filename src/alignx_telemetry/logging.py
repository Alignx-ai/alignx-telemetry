"""Logging instrumentation for AlignX Telemetry."""

import logging
import os
from typing import Optional, Union

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import (
    LoggingInstrumentor as OTelLoggingInstrumentor,
)


# Default clean format that avoids redundancy
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] - %(message)s"


class LoggingInstrumentor:
    """Instrumentor for logging integration with OpenTelemetry.

    This class sets up the OpenTelemetry logging provider, exporters,
    and handler to integrate with Python's standard logging.
    """

    def __init__(self):
        """Initialize the logging instrumentor."""
        self._logger_provider = None
        self._handler = None
        self._instrumented = False
        self._otel_logging_instrumentor = None

    def instrument(
        self,
        resource: Resource,
        log_level: Optional[Union[int, str]] = logging.INFO,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        log_format: Optional[str] = None,
    ) -> None:
        """Instrument the Python logging module to send logs via OpenTelemetry.

        Args:
            resource: Existing OpenTelemetry resource to use.
            log_level: The log level to set for the root logger. Defaults to INFO.
            otlp_endpoint: OTLP exporter endpoint. If None, OTLP export is disabled.
            console_export: Whether to add a console exporter for logs.
            log_format: Custom log format to use. If None, uses OTEL_PYTHON_LOG_FORMAT env var
                or falls back to a clean default format.
        """
        if self._instrumented:
            logging.warning("Logging already instrumented.")
            return

        # Get otlp_endpoint from env vars if not provided
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_LOGS")

        # Convert log_level from string to int if needed
        if isinstance(log_level, str):
            level_int = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level_int = log_level or logging.INFO

        # 1. Create and set LoggerProvider
        self._logger_provider = LoggerProvider(resource=resource)
        set_logger_provider(self._logger_provider)

        # 2. Configure Exporters and Processors
        if otlp_endpoint:
            otlp_exporter = OTLPLogExporter(endpoint=otlp_endpoint)
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(otlp_exporter)
            )
            logging.info(f"Configured OTLP Log Exporter to: {otlp_endpoint}")

        if console_export:
            console_exporter = ConsoleLogExporter()
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(console_exporter)
            )
            logging.info("Configured Console Log Exporter.")

        # 3. Create and attach LoggingHandler
        # Set handler level to NOTSET; filtering is done by the logger level itself
        self._handler = LoggingHandler(
            level=logging.NOTSET, logger_provider=self._logger_provider
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self._handler)

        # 4. Set root logger level
        root_logger.setLevel(level_int)
        logging.info(f"Set root logger level to: {logging.getLevelName(level_int)}")

        # 5. Configure log format
        # Get format from parameter, env var, or default to clean format
        log_format = log_format or os.getenv(
            "OTEL_PYTHON_LOG_FORMAT", DEFAULT_LOG_FORMAT
        )

        # Use OTel logging instrumentor to inject trace context
        self._otel_logging_instrumentor = OTelLoggingInstrumentor()
        self._otel_logging_instrumentor.instrument(
            set_logging_format=True, logging_format=log_format, log_level=level_int
        )

        self._instrumented = True
        logging.info("Logging instrumentation complete.")

    def get_logger(
        self, name: str, level: Optional[Union[int, str]] = None
    ) -> logging.Logger:
        """Get a logger with the specified name and level.

        Args:
            name: Name of the logger
            level: Log level for the logger (optional)

        Returns:
            A logger instance
        """
        if not self._instrumented:
            raise RuntimeError("Logging not instrumented. Call instrument() first.")

        logger = logging.getLogger(name)

        if level is not None:
            # Convert level from string to int if needed
            if isinstance(level, str):
                level_int = getattr(logging, level.upper(), None)
                if level_int is not None:
                    logger.setLevel(level_int)
            else:
                logger.setLevel(level)

        return logger

    def shutdown(self) -> None:
        """Uninstrument the Python logging module and shut down the provider."""
        if not self._instrumented or not self._handler or not self._logger_provider:
            return

        # Uninstrument OTel logging if used
        if self._otel_logging_instrumentor:
            self._otel_logging_instrumentor.uninstrument()
            self._otel_logging_instrumentor = None

        root_logger = logging.getLogger()
        if self._handler in root_logger.handlers:
            root_logger.removeHandler(self._handler)
        self._handler = None

        # Shutdown the logger provider
        self._logger_provider.shutdown()
        set_logger_provider(None)  # Reset global provider
        self._logger_provider = None

        self._instrumented = False
        logging.info("Logging instrumentation shutdown complete.")
