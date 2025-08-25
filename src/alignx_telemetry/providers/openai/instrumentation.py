"""OpenAI provider instrumentation implementation.

This module implements OpenAI instrumentation at the lowest level (OpenAI.request)
to capture ALL OpenAI API usage patterns, inspired by OpenInference and OpenLLMetry.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Iterator, AsyncIterator, Union, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..base import AlignXBaseProviderInstrumentation, AlignXProviderCallData
from ..metrics import AlignXProviderMetrics
from .metrics_extractor import OpenAIMetricsExtractor
from .response_parser import OpenAIResponseParser
from .cost_calculator import OpenAICostCalculator

logger = logging.getLogger(__name__)


class AlignXOpenAIInstrumentation(AlignXBaseProviderInstrumentation):
    """OpenAI provider instrumentation for AlignX.

    Instruments OpenAI at the request level to capture all usage patterns:
    - Vanilla OpenAI SDK usage
    - LangChain ChatOpenAI calls
    - LlamaIndex OpenAI calls
    - CrewAI OpenAI calls
    - AutoGen OpenAI calls
    - Any framework or abstraction that uses OpenAI SDK

    Inspired by:
    - OpenInference OpenAI instrumentation patterns
    - OpenLLMetry request-level wrapping
    - Opik OpenAI tracking approach
    """

    @property
    def provider_name(self) -> str:
        """Provider name for OpenAI."""
        return "openai"

    @property
    def instrumentation_targets(self) -> Dict[str, str]:
        """OpenAI instrumentation targets."""
        return {
            "openai.OpenAI.request": "_wrap_openai_request",
            "openai.AsyncOpenAI.request": "_wrap_async_openai_request",
        }

    def __init__(self, telemetry_manager):
        """Initialize OpenAI instrumentation.

        Args:
            telemetry_manager: AlignX telemetry manager instance
        """
        super().__init__(telemetry_manager)

        # Initialize helper components
        self.metrics_extractor = OpenAIMetricsExtractor()
        self.response_parser = OpenAIResponseParser()
        self.cost_calculator = OpenAICostCalculator()

        # Track whether OpenAI is available
        self._openai_available = False
        self._openai_module = None

    def _perform_instrumentation(self) -> bool:
        """Perform OpenAI SDK instrumentation.

        Returns:
            True if instrumentation succeeded, False otherwise
        """
        try:
            # Import OpenAI SDK
            import openai

            self._openai_module = openai
            self._openai_available = True

            # Import wrapt for method wrapping
            from wrapt import wrap_function_wrapper

            # Store original methods for restoration
            self._original_methods["openai.OpenAI.request"] = openai.OpenAI.request
            self._original_methods["openai.AsyncOpenAI.request"] = (
                openai.AsyncOpenAI.request
            )

            # Wrap synchronous request method
            wrap_function_wrapper(
                module="openai",
                name="OpenAI.request",
                wrapper=self._wrap_openai_request,
            )

            # Wrap asynchronous request method
            wrap_function_wrapper(
                module="openai",
                name="AsyncOpenAI.request",
                wrapper=self._wrap_async_openai_request,
            )

            self.logger.info("Successfully instrumented OpenAI SDK at request level")
            return True

        except ImportError:
            self.logger.debug("OpenAI SDK not available for instrumentation")
            return False
        except Exception as e:
            self.logger.error(f"Error instrumenting OpenAI SDK: {e}", exc_info=True)
            return False

    def _wrap_openai_request(self, wrapped, instance, args, kwargs):
        """Wrap synchronous OpenAI request method.

        Args:
            wrapped: Original method
            instance: OpenAI instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Response from OpenAI API
        """
        # Check if instrumentation should be suppressed
        if self.should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Extract call data
        call_data = self.extract_call_data("request", args, kwargs)

        # Create OpenTelemetry span
        span = self.create_provider_span(call_data)

        try:
            with trace.use_span(span, end_on_exit=False):
                # Execute the request
                call_data.start_time = time.time()
                response = wrapped(*args, **kwargs)
                call_data.end_time = time.time()

                # Handle streaming vs non-streaming responses
                if self._is_streaming_response(response):
                    return self._handle_streaming_response(span, call_data, response)
                else:
                    # Parse response data
                    self.extract_response_data(call_data, response)

                    # Calculate and emit metrics
                    metrics = self.calculate_metrics(call_data)
                    self.metrics_emitter.emit_llm_metrics(metrics)

                    # Finalize span
                    self.finalize_span(span, call_data)

                    return response

        except Exception as e:
            # Handle errors
            call_data.error = e
            call_data.error_type = type(e).__name__
            call_data.end_time = time.time()

            # Calculate error metrics
            metrics = self.calculate_metrics(call_data)
            self.metrics_emitter.emit_llm_metrics(metrics)

            # Finalize span with error
            self.finalize_span(span, call_data)

            raise

    async def _wrap_async_openai_request(self, wrapped, instance, args, kwargs):
        """Wrap asynchronous OpenAI request method.

        Args:
            wrapped: Original async method
            instance: AsyncOpenAI instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Response from OpenAI API
        """
        # Check if instrumentation should be suppressed
        if self.should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Extract call data
        call_data = self.extract_call_data("request", args, kwargs)

        # Create OpenTelemetry span
        span = self.create_provider_span(call_data)

        try:
            with trace.use_span(span, end_on_exit=False):
                # Execute the async request
                call_data.start_time = time.time()
                response = await wrapped(*args, **kwargs)
                call_data.end_time = time.time()

                # Handle streaming vs non-streaming responses
                if self._is_async_streaming_response(response):
                    return self._handle_async_streaming_response(
                        span, call_data, response
                    )
                else:
                    # Parse response data
                    self.extract_response_data(call_data, response)

                    # Calculate and emit metrics
                    metrics = self.calculate_metrics(call_data)
                    self.metrics_emitter.emit_llm_metrics(metrics)

                    # Finalize span
                    self.finalize_span(span, call_data)

                    return response

        except Exception as e:
            # Handle errors
            call_data.error = e
            call_data.error_type = type(e).__name__
            call_data.end_time = time.time()

            # Calculate error metrics
            metrics = self.calculate_metrics(call_data)
            self.metrics_emitter.emit_llm_metrics(metrics)

            # Finalize span with error
            self.finalize_span(span, call_data)

            raise

    def extract_call_data(
        self, method: str, args: tuple, kwargs: dict
    ) -> AlignXProviderCallData:
        """Extract call data from OpenAI request arguments.

        Args:
            method: Method name
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            AlignXProviderCallData: Extracted call data
        """
        try:
            return self.metrics_extractor.extract_call_data(method, args, kwargs)
        except Exception as e:
            self.logger.warning(f"Error extracting call data: {e}")
            return AlignXProviderCallData(method="unknown", model="unknown", inputs={})

    def extract_response_data(
        self, call_data: AlignXProviderCallData, response: Any
    ) -> None:
        """Extract response data from OpenAI API response.

        Args:
            call_data: Call data to update
            response: OpenAI response object
        """
        try:
            self.response_parser.parse_response(call_data, response)
        except Exception as e:
            self.logger.warning(f"Error parsing response data: {e}")

    def calculate_metrics(
        self, call_data: AlignXProviderCallData
    ) -> AlignXProviderMetrics:
        """Calculate metrics from call data.

        Args:
            call_data: Complete call data

        Returns:
            AlignXProviderMetrics: Calculated metrics
        """
        try:
            # Import context to avoid circular imports
            from ..context import AlignXFrameworkContext

            # Get framework context
            framework_context = AlignXFrameworkContext.get_current_context()

            # Extract token counts from response
            input_tokens = (
                call_data.response_metadata.get("input_tokens", 0)
                if call_data.response_metadata
                else 0
            )
            output_tokens = (
                call_data.response_metadata.get("output_tokens", 0)
                if call_data.response_metadata
                else 0
            )

            # Calculate cost
            cost_usd = self.cost_calculator.calculate_cost(
                model=call_data.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Calculate latency
            latency_ms = 0.0
            if call_data.start_time and call_data.end_time:
                latency_ms = (call_data.end_time - call_data.start_time) * 1000

            # Calculate time to first token for streaming
            ttft_ms = None
            if call_data.time_to_first_token and call_data.start_time:
                ttft_ms = (call_data.time_to_first_token - call_data.start_time) * 1000

            # Get application context from telemetry manager
            app_name = getattr(self.telemetry_manager, "service_name", "unknown")
            environment = getattr(self.telemetry_manager, "environment", "unknown")

            return AlignXProviderMetrics(
                provider="openai",
                model=call_data.model,
                operation=call_data.method,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                framework=framework_context.get("framework"),
                workflow_id=framework_context.get("workflow_id"),
                node_name=framework_context.get("node_name"),
                agent_name=framework_context.get("agent_name"),
                application_name=app_name,
                environment=environment,
                error_type=call_data.error_type,
                success=call_data.error is None,
                metadata=call_data.provider_metadata or {},
            )

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            # Return basic metrics on error
            return AlignXProviderMetrics(
                provider="openai",
                model=call_data.model,
                operation=call_data.method,
                application_name="unknown",
                environment="unknown",
                success=call_data.error is None,
            )

    def _is_streaming_response(self, response: Any) -> bool:
        """Check if response is a streaming response.

        Args:
            response: OpenAI response object

        Returns:
            True if response is streaming
        """
        # Check for streaming response patterns
        return (
            hasattr(response, "__iter__")
            and not isinstance(response, (str, bytes, dict))
            and hasattr(response, "__next__")
        )

    def _is_async_streaming_response(self, response: Any) -> bool:
        """Check if response is an async streaming response.

        Args:
            response: OpenAI response object

        Returns:
            True if response is async streaming
        """
        return hasattr(response, "__aiter__") and hasattr(response, "__anext__")

    def _handle_streaming_response(
        self,
        span: trace.Span,
        call_data: AlignXProviderCallData,
        response_iter: Iterator,
    ) -> Iterator:
        """Handle streaming response with real-time metrics.

        Args:
            span: OpenTelemetry span
            call_data: Call data
            response_iter: Streaming response iterator

        Returns:
            Wrapped iterator that emits metrics
        """
        accumulated_response = []
        first_chunk_time = None

        def streaming_wrapper():
            nonlocal first_chunk_time

            try:
                for chunk in response_iter:
                    # Record time to first token
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        call_data.time_to_first_token = first_chunk_time

                    # Accumulate response for final processing
                    accumulated_response.append(chunk)

                    yield chunk

                # Process final accumulated response
                call_data.end_time = time.time()
                self._process_accumulated_streaming_response(
                    call_data, accumulated_response
                )

                # Calculate and emit final metrics
                metrics = self.calculate_metrics(call_data)
                self.metrics_emitter.emit_llm_metrics(metrics)

            except Exception as e:
                call_data.error = e
                call_data.error_type = type(e).__name__
                call_data.end_time = time.time()

                metrics = self.calculate_metrics(call_data)
                self.metrics_emitter.emit_llm_metrics(metrics)
                raise
            finally:
                self.finalize_span(span, call_data)

        return streaming_wrapper()

    async def _handle_async_streaming_response(
        self,
        span: trace.Span,
        call_data: AlignXProviderCallData,
        response_iter: AsyncIterator,
    ) -> AsyncIterator:
        """Handle async streaming response with real-time metrics.

        Args:
            span: OpenTelemetry span
            call_data: Call data
            response_iter: Async streaming response iterator

        Returns:
            Wrapped async iterator that emits metrics
        """
        accumulated_response = []
        first_chunk_time = None

        async def async_streaming_wrapper():
            nonlocal first_chunk_time

            try:
                async for chunk in response_iter:
                    # Record time to first token
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        call_data.time_to_first_token = first_chunk_time

                    # Accumulate response for final processing
                    accumulated_response.append(chunk)

                    yield chunk

                # Process final accumulated response
                call_data.end_time = time.time()
                self._process_accumulated_streaming_response(
                    call_data, accumulated_response
                )

                # Calculate and emit final metrics
                metrics = self.calculate_metrics(call_data)
                self.metrics_emitter.emit_llm_metrics(metrics)

            except Exception as e:
                call_data.error = e
                call_data.error_type = type(e).__name__
                call_data.end_time = time.time()

                metrics = self.calculate_metrics(call_data)
                self.metrics_emitter.emit_llm_metrics(metrics)
                raise
            finally:
                self.finalize_span(span, call_data)

        return async_streaming_wrapper()

    def _process_accumulated_streaming_response(
        self, call_data: AlignXProviderCallData, chunks: list
    ) -> None:
        """Process accumulated streaming response chunks.

        Args:
            call_data: Call data to update
            chunks: List of response chunks
        """
        try:
            # Combine chunks into final response data
            combined_response = self.response_parser.combine_streaming_chunks(chunks)
            self.extract_response_data(call_data, combined_response)
        except Exception as e:
            self.logger.warning(f"Error processing streaming response: {e}")

    def _add_response_attributes(
        self, span: trace.Span, call_data: AlignXProviderCallData
    ) -> None:
        """Add OpenAI-specific response attributes to span.

        Args:
            span: Span to add attributes to
            call_data: Call data with response information
        """
        try:
            if call_data.response_metadata:
                # Add token usage attributes
                if "input_tokens" in call_data.response_metadata:
                    span.set_attribute(
                        "gen_ai.usage.input_tokens",
                        call_data.response_metadata["input_tokens"],
                    )

                if "output_tokens" in call_data.response_metadata:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        call_data.response_metadata["output_tokens"],
                    )

                # Add cost information
                input_tokens = call_data.response_metadata.get("input_tokens", 0)
                output_tokens = call_data.response_metadata.get("output_tokens", 0)
                cost = self.cost_calculator.calculate_cost(
                    call_data.model, input_tokens, output_tokens
                )
                span.set_attribute("alignx.cost_usd", cost)

                # Add performance metrics
                if call_data.start_time and call_data.end_time:
                    latency_ms = (call_data.end_time - call_data.start_time) * 1000
                    span.set_attribute("alignx.latency_ms", latency_ms)

                if call_data.time_to_first_token and call_data.start_time:
                    ttft_ms = (
                        call_data.time_to_first_token - call_data.start_time
                    ) * 1000
                    span.set_attribute("alignx.time_to_first_token_ms", ttft_ms)

        except Exception as e:
            self.logger.warning(f"Error adding response attributes: {e}")
