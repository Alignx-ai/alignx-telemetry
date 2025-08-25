"""AlignX Google GenAI Instrumentation

Provides universal instrumentation for Google's Generative AI models by wrapping
the core API methods at the lowest level to ensure all calls are captured
regardless of the framework used.

Supports both google-generativeai and google-genai packages.
"""

import logging
import time
from typing import Any, Dict, Optional, Union

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse

    GOOGLE_GENERATIVEAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENERATIVEAI_AVAILABLE = False
    genai = None
    GenerateContentResponse = None

try:
    import google.genai as google_genai
    from google.genai.types import GenerateContentResponse as NewGenerateContentResponse

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    google_genai = None
    NewGenerateContentResponse = None

import wrapt
from opentelemetry.trace import SpanKind

from ..base import AlignXBaseProviderInstrumentation, AlignXProviderCallData
from ..context import AlignXFrameworkContext, AlignXTraceContext
from ..metrics import AlignXProviderMetrics
from .metrics_extractor import GoogleGenAIMetricsExtractor
from .response_parser import GoogleGenAIResponseParser
from .cost_calculator import GoogleGenAICostCalculator

logger = logging.getLogger(__name__)


class AlignXGoogleGenAIInstrumentation(AlignXBaseProviderInstrumentation):
    """Google GenAI-specific instrumentation implementation."""

    @property
    def provider_name(self) -> str:
        return "google_genai"

    @property
    def instrumentation_targets(self) -> Dict[str, Dict[str, str]]:
        """Define the methods to instrument for Google GenAI."""
        targets = {}

        if GOOGLE_GENERATIVEAI_AVAILABLE:
            targets.update(
                {
                    "google.generativeai.generative_models.GenerativeModel": {
                        "generate_content": "google.generativeai.generate_content",
                        "generate_content_async": "google.generativeai.generate_content_async",
                        "count_tokens": "google.generativeai.count_tokens",
                        "count_tokens_async": "google.generativeai.count_tokens_async",
                    }
                }
            )

        if GOOGLE_GENAI_AVAILABLE:
            targets.update(
                {
                    "google.genai.GenerativeModel": {
                        "generate_content": "google.genai.generate_content",
                        "generate_content_async": "google.genai.generate_content_async",
                    }
                }
            )

        return targets

    def _perform_instrumentation(self) -> bool:
        """Perform the actual instrumentation by wrapping Google GenAI methods."""
        if not GOOGLE_GENERATIVEAI_AVAILABLE and not GOOGLE_GENAI_AVAILABLE:
            logger.warning(
                "Google GenAI packages not available, skipping instrumentation"
            )
            return False

        try:
            # Initialize utility classes
            self.metrics_extractor = GoogleGenAIMetricsExtractor()
            self.response_parser = GoogleGenAIResponseParser()
            self.cost_calculator = GoogleGenAICostCalculator()

            # Instrument google-generativeai package (legacy)
            if GOOGLE_GENERATIVEAI_AVAILABLE:
                self._instrument_legacy_package()

            # Instrument google-genai package (new)
            if GOOGLE_GENAI_AVAILABLE:
                self._instrument_new_package()

            logger.info("Successfully instrumented Google GenAI provider")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument Google GenAI provider: {e}")
            return False

    def _instrument_legacy_package(self):
        """Instrument the legacy google-generativeai package."""
        try:
            # Wrap GenerativeModel.generate_content (sync)
            wrapt.wrap_function_wrapper(
                "google.generativeai.generative_models",
                "GenerativeModel.generate_content",
                self._wrap_generate_content,
            )

            # Wrap GenerativeModel.generate_content_async (async)
            wrapt.wrap_function_wrapper(
                "google.generativeai.generative_models",
                "GenerativeModel.generate_content_async",
                self._wrap_generate_content_async,
            )

            # Wrap count_tokens methods
            try:
                wrapt.wrap_function_wrapper(
                    "google.generativeai.generative_models",
                    "GenerativeModel.count_tokens",
                    self._wrap_count_tokens,
                )
                wrapt.wrap_function_wrapper(
                    "google.generativeai.generative_models",
                    "GenerativeModel.count_tokens_async",
                    self._wrap_count_tokens_async,
                )
            except AttributeError:
                logger.debug("count_tokens methods not available in this version")

            logger.debug("Successfully instrumented google-generativeai package")

        except Exception as e:
            logger.error(f"Failed to instrument google-generativeai package: {e}")

    def _instrument_new_package(self):
        """Instrument the new google-genai package."""
        try:
            # Wrap GenerativeModel.generate_content (sync)
            wrapt.wrap_function_wrapper(
                "google.genai",
                "GenerativeModel.generate_content",
                self._wrap_new_generate_content,
            )

            # Wrap GenerativeModel.generate_content_async (async)
            wrapt.wrap_function_wrapper(
                "google.genai",
                "GenerativeModel.generate_content_async",
                self._wrap_new_generate_content_async,
            )

            logger.debug("Successfully instrumented google-genai package")

        except Exception as e:
            logger.error(f"Failed to instrument google-genai package: {e}")

    def _wrap_generate_content(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous generate_content method (legacy package)."""
        return self._handle_generate_content_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="generate_content",
            package="google-generativeai",
            is_async=False,
        )

    async def _wrap_generate_content_async(self, wrapped, instance, args, kwargs):
        """Wrap the asynchronous generate_content method (legacy package)."""
        return await self._handle_generate_content_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="generate_content_async",
            package="google-generativeai",
            is_async=True,
        )

    def _wrap_new_generate_content(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous generate_content method (new package)."""
        return self._handle_generate_content_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="generate_content",
            package="google-genai",
            is_async=False,
        )

    async def _wrap_new_generate_content_async(self, wrapped, instance, args, kwargs):
        """Wrap the asynchronous generate_content method (new package)."""
        return await self._handle_generate_content_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="generate_content_async",
            package="google-genai",
            is_async=True,
        )

    def _wrap_count_tokens(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous count_tokens method."""
        return self._handle_count_tokens_call(
            wrapped, instance, args, kwargs, operation="count_tokens", is_async=False
        )

    async def _wrap_count_tokens_async(self, wrapped, instance, args, kwargs):
        """Wrap the asynchronous count_tokens method."""
        return await self._handle_count_tokens_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="count_tokens_async",
            is_async=True,
        )

    def _handle_generate_content_call(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        operation: str,
        package: str,
        is_async: bool = False,
    ):
        """Handle instrumentation for generate_content calls."""

        # Check if instrumentation should be suppressed
        if AlignXTraceContext.should_suppress():
            if is_async:
                return wrapped(*args, **kwargs)
            else:
                return wrapped(*args, **kwargs)

        # Extract call data
        call_data = self.extract_call_data(args, kwargs, operation, instance, package)

        # Start span
        span = self._create_span(call_data, SpanKind.CLIENT)

        if not span:
            return (
                wrapped(*args, **kwargs) if not is_async else wrapped(*args, **kwargs)
            )

        start_time = time.time()

        try:
            # Make the actual API call
            if is_async:
                response = wrapped(*args, **kwargs)
            else:
                response = wrapped(*args, **kwargs)

            # Handle streaming response
            if self._is_streaming_response(response):
                if is_async:
                    return self._wrap_async_stream_response(
                        response, span, call_data, start_time
                    )
                else:
                    return self._wrap_stream_response(
                        response, span, call_data, start_time
                    )

            # Handle regular response
            end_time = time.time()
            response_data = self.extract_response_data(response)
            metrics = self.calculate_metrics(
                call_data, response_data, start_time, end_time
            )

            self._finalize_span(span, call_data, response_data, metrics)

            return response

        except Exception as e:
            end_time = time.time()
            self._handle_exception(span, call_data, e, start_time, end_time)
            raise

    def _handle_count_tokens_call(
        self, wrapped, instance, args, kwargs, operation: str, is_async: bool = False
    ):
        """Handle instrumentation for count_tokens calls."""

        # Check if instrumentation should be suppressed
        if AlignXTraceContext.should_suppress():
            if is_async:
                return wrapped(*args, **kwargs)
            else:
                return wrapped(*args, **kwargs)

        # Extract call data
        call_data = self.extract_call_data(
            args, kwargs, operation, instance, "google-generativeai"
        )

        # Start span
        span = self._create_span(call_data, SpanKind.CLIENT)

        if not span:
            return (
                wrapped(*args, **kwargs) if not is_async else wrapped(*args, **kwargs)
            )

        start_time = time.time()

        try:
            # Make the actual API call
            if is_async:
                response = wrapped(*args, **kwargs)
            else:
                response = wrapped(*args, **kwargs)

            end_time = time.time()
            response_data = self.extract_response_data(response)
            metrics = self.calculate_metrics(
                call_data, response_data, start_time, end_time
            )

            self._finalize_span(span, call_data, response_data, metrics)

            return response

        except Exception as e:
            end_time = time.time()
            self._handle_exception(span, call_data, e, start_time, end_time)
            raise

    def _is_streaming_response(self, response) -> bool:
        """Check if the response is a streaming response."""
        # Check for streaming indicators
        if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            return True
        if hasattr(response, "__aiter__"):
            return True
        return False

    def _wrap_stream_response(self, response, span, call_data, start_time):
        """Wrap streaming response to capture metrics when stream completes."""
        original_iterator = iter(response)
        collected_chunks = []

        def stream_wrapper():
            nonlocal collected_chunks
            try:
                for chunk in original_iterator:
                    collected_chunks.append(chunk)
                    yield chunk

                # Stream completed, finalize span
                end_time = time.time()
                combined_response = self.response_parser.combine_streaming_chunks(
                    collected_chunks
                )
                response_data = self.extract_response_data(combined_response)
                metrics = self.calculate_metrics(
                    call_data, response_data, start_time, end_time
                )
                self._finalize_span(span, call_data, response_data, metrics)

            except Exception as e:
                end_time = time.time()
                self._handle_exception(span, call_data, e, start_time, end_time)
                raise

        # Replace the response iterator
        response._iterator = stream_wrapper()
        return response

    async def _wrap_async_stream_response(self, response, span, call_data, start_time):
        """Wrap async streaming response to capture metrics when stream completes."""
        original_aiterator = response.__aiter__()
        collected_chunks = []

        async def async_stream_wrapper():
            nonlocal collected_chunks
            try:
                async for chunk in original_aiterator:
                    collected_chunks.append(chunk)
                    yield chunk

                # Stream completed, finalize span
                end_time = time.time()
                combined_response = self.response_parser.combine_streaming_chunks(
                    collected_chunks
                )
                response_data = self.extract_response_data(combined_response)
                metrics = self.calculate_metrics(
                    call_data, response_data, start_time, end_time
                )
                self._finalize_span(span, call_data, response_data, metrics)

            except Exception as e:
                end_time = time.time()
                self._handle_exception(span, call_data, e, start_time, end_time)
                raise

        # Replace the response async iterator
        response._async_iterator = async_stream_wrapper()
        return response

    def extract_call_data(
        self,
        args: tuple,
        kwargs: dict,
        operation: str = "unknown",
        instance=None,
        package: str = "unknown",
    ) -> AlignXProviderCallData:
        """Extract structured data from Google GenAI API call arguments."""
        return self.metrics_extractor.extract_call_data(
            args, kwargs, operation, instance, package
        )

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract structured data from Google GenAI API response."""
        return self.response_parser.extract_response_data(response)

    def calculate_metrics(
        self,
        call_data: AlignXProviderCallData,
        response_data: Dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> AlignXProviderMetrics:
        """Calculate comprehensive metrics for the Google GenAI API call."""

        # Get framework context
        framework_context = AlignXFrameworkContext.get_current()

        # Calculate timing metrics
        duration = end_time - start_time
        ttft = response_data.get("ttft")  # Time to first token for streaming

        # Calculate cost
        cost = self.cost_calculator.calculate_cost(
            model=call_data.model,
            input_tokens=response_data.get("input_tokens", 0),
            output_tokens=response_data.get("output_tokens", 0),
        )

        return AlignXProviderMetrics(
            # Provider info
            provider="google_genai",
            model=call_data.model,
            operation=call_data.operation,
            # Usage metrics
            input_tokens=response_data.get("input_tokens", 0),
            output_tokens=response_data.get("output_tokens", 0),
            total_tokens=response_data.get("input_tokens", 0)
            + response_data.get("output_tokens", 0),
            # Cost metrics
            cost_usd=cost,
            # Performance metrics
            duration_ms=duration * 1000,
            ttft_ms=ttft * 1000 if ttft else None,
            # Framework context (if available)
            framework=framework_context.framework if framework_context else None,
            workflow_id=framework_context.workflow_id if framework_context else None,
            node_name=framework_context.node_name if framework_context else None,
            agent_name=framework_context.agent_name if framework_context else None,
            # Additional metadata
            metadata={
                "finish_reason": response_data.get("finish_reason"),
                "safety_ratings": response_data.get("safety_ratings"),
                "package": call_data.args.get("package", "unknown"),
                **(framework_context.metadata if framework_context else {}),
            },
        )
