"""AlignX Anthropic Instrumentation

Provides universal instrumentation for Anthropic's Claude models by wrapping
the core API methods at the lowest level to ensure all calls are captured
regardless of the framework used.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

try:
    import anthropic
    from anthropic.types import Message, TextBlock, MessageParam
    from anthropic._streaming import Stream, AsyncStream

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Message = None
    TextBlock = None
    MessageParam = None
    Stream = None
    AsyncStream = None

import wrapt
from opentelemetry.trace import SpanKind

from ..base import AlignXBaseProviderInstrumentation, AlignXProviderCallData
from ..context import AlignXFrameworkContext, AlignXTraceContext
from ..metrics import AlignXProviderMetrics
from .metrics_extractor import AnthropicMetricsExtractor
from .response_parser import AnthropicResponseParser
from .cost_calculator import AnthropicCostCalculator

logger = logging.getLogger(__name__)


class AlignXAnthropicInstrumentation(AlignXBaseProviderInstrumentation):
    """Anthropic-specific instrumentation implementation."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def instrumentation_targets(self) -> Dict[str, Dict[str, str]]:
        """Define the methods to instrument for Anthropic."""
        if not ANTHROPIC_AVAILABLE:
            return {}

        return {
            "anthropic.resources.messages.Messages": {
                "create": "anthropic.messages.create",
                "stream": "anthropic.messages.stream",
            },
            "anthropic.resources.completions.Completions": {
                "create": "anthropic.completions.create",
            },
        }

    def _perform_instrumentation(self) -> bool:
        """Perform the actual instrumentation by wrapping Anthropic methods."""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not available, skipping instrumentation")
            return False

        try:
            # Initialize utility classes
            self.metrics_extractor = AnthropicMetricsExtractor()
            self.response_parser = AnthropicResponseParser()
            self.cost_calculator = AnthropicCostCalculator()

            # Wrap Messages.create (main chat API)
            wrapt.wrap_function_wrapper(
                "anthropic.resources.messages",
                "Messages.create",
                self._wrap_anthropic_messages_create,
            )

            # Wrap Messages.stream (streaming chat API)
            wrapt.wrap_function_wrapper(
                "anthropic.resources.messages",
                "Messages.stream",
                self._wrap_anthropic_messages_stream,
            )

            # Wrap AsyncMessages.create (async chat API)
            try:
                wrapt.wrap_function_wrapper(
                    "anthropic.resources.messages",
                    "AsyncMessages.create",
                    self._wrap_async_anthropic_messages_create,
                )

                # Wrap AsyncMessages.stream (async streaming chat API)
                wrapt.wrap_function_wrapper(
                    "anthropic.resources.messages",
                    "AsyncMessages.stream",
                    self._wrap_async_anthropic_messages_stream,
                )
            except AttributeError:
                # Async methods might not be available in all versions
                logger.debug(
                    "Async Anthropic methods not available for instrumentation"
                )

            # Wrap legacy Completions.create if available
            try:
                wrapt.wrap_function_wrapper(
                    "anthropic.resources.completions",
                    "Completions.create",
                    self._wrap_anthropic_completions_create,
                )
            except AttributeError:
                # Completions API might be deprecated or not available
                logger.debug(
                    "Anthropic Completions API not available for instrumentation"
                )

            logger.info("Successfully instrumented Anthropic provider")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument Anthropic provider: {e}")
            return False

    def _wrap_anthropic_messages_create(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous Messages.create method."""
        return self._handle_anthropic_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="messages.create",
            is_async=False,
            is_streaming=False,
        )

    def _wrap_anthropic_messages_stream(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous Messages.stream method."""
        return self._handle_anthropic_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="messages.stream",
            is_async=False,
            is_streaming=True,
        )

    async def _wrap_async_anthropic_messages_create(
        self, wrapped, instance, args, kwargs
    ):
        """Wrap the asynchronous Messages.create method."""
        return await self._handle_anthropic_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="messages.create",
            is_async=True,
            is_streaming=False,
        )

    async def _wrap_async_anthropic_messages_stream(
        self, wrapped, instance, args, kwargs
    ):
        """Wrap the asynchronous Messages.stream method."""
        return await self._handle_anthropic_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="messages.stream",
            is_async=True,
            is_streaming=True,
        )

    def _wrap_anthropic_completions_create(self, wrapped, instance, args, kwargs):
        """Wrap the synchronous Completions.create method (legacy API)."""
        return self._handle_anthropic_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation="completions.create",
            is_async=False,
            is_streaming=False,
        )

    def _handle_anthropic_call(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        operation: str,
        is_async: bool = False,
        is_streaming: bool = False,
    ):
        """Handle instrumentation for any Anthropic API call."""

        # Check if instrumentation should be suppressed
        if AlignXTraceContext.should_suppress():
            if is_async:
                return wrapped(*args, **kwargs)
            else:
                return wrapped(*args, **kwargs)

        # Extract call data
        call_data = self.extract_call_data(args, kwargs, operation)

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
            if is_streaming:
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

    def _wrap_stream_response(self, response, span, call_data, start_time):
        """Wrap streaming response to capture metrics when stream completes."""
        if not isinstance(response, Stream):
            return response

        # Create a wrapper that will capture metrics
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
        if not isinstance(response, AsyncStream):
            return response

        # Create an async wrapper that will capture metrics
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
        self, args: tuple, kwargs: dict, operation: str = "unknown"
    ) -> AlignXProviderCallData:
        """Extract structured data from Anthropic API call arguments."""
        return self.metrics_extractor.extract_call_data(args, kwargs, operation)

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract structured data from Anthropic API response."""
        return self.response_parser.extract_response_data(response)

    def calculate_metrics(
        self,
        call_data: AlignXProviderCallData,
        response_data: Dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> AlignXProviderMetrics:
        """Calculate comprehensive metrics for the Anthropic API call."""

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
            provider="anthropic",
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
                "stop_reason": response_data.get("stop_reason"),
                "finish_reason": response_data.get("finish_reason"),
                **(framework_context.metadata if framework_context else {}),
            },
        )
