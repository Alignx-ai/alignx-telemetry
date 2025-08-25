"""AlignX Bedrock Instrumentation

Provides universal instrumentation for AWS Bedrock foundation models by wrapping
the boto3 bedrock-runtime client methods to ensure all calls are captured
regardless of the framework used.

Supports multiple foundation models through Bedrock.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Union

try:
    import boto3
    from botocore.client import BaseClient
    from botocore.response import StreamingBody

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    BaseClient = None
    StreamingBody = None

import wrapt
from opentelemetry.trace import SpanKind

from ..base import AlignXBaseProviderInstrumentation, AlignXProviderCallData
from ..context import AlignXFrameworkContext, AlignXTraceContext
from ..metrics import AlignXProviderMetrics
from .metrics_extractor import BedrockMetricsExtractor
from .response_parser import BedrockResponseParser
from .cost_calculator import BedrockCostCalculator

logger = logging.getLogger(__name__)


class AlignXBedrockInstrumentation(AlignXBaseProviderInstrumentation):
    """Bedrock-specific instrumentation implementation."""

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def instrumentation_targets(self) -> Dict[str, Dict[str, str]]:
        """Define the methods to instrument for Bedrock."""
        if not BOTO3_AVAILABLE:
            return {}

        return {
            "botocore.client.BaseClient": {
                "invoke_model": "bedrock.invoke_model",
                "invoke_model_with_response_stream": "bedrock.invoke_model_stream",
                "converse": "bedrock.converse",
                "converse_stream": "bedrock.converse_stream",
            }
        }

    def _perform_instrumentation(self) -> bool:
        """Perform the actual instrumentation by wrapping Bedrock methods."""
        if not BOTO3_AVAILABLE:
            logger.warning(
                "boto3 package not available, skipping Bedrock instrumentation"
            )
            return False

        try:
            # Initialize utility classes
            self.metrics_extractor = BedrockMetricsExtractor()
            self.response_parser = BedrockResponseParser()
            self.cost_calculator = BedrockCostCalculator()

            # Wrap client creation to instrument bedrock-runtime clients
            wrapt.wrap_function_wrapper(
                "botocore.client", "BaseClient._make_request", self._wrap_client_request
            )

            logger.info("Successfully instrumented Bedrock provider")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument Bedrock provider: {e}")
            return False

    def _wrap_client_request(self, wrapped, instance, args, kwargs):
        """Wrap boto3 client requests to capture Bedrock API calls."""

        # Check if this is a bedrock-runtime client
        if not self._is_bedrock_client(instance):
            return wrapped(*args, **kwargs)

        # Extract operation name from args
        operation_model = args[0] if args else None
        if not operation_model or not hasattr(operation_model, "name"):
            return wrapped(*args, **kwargs)

        operation_name = operation_model.name

        # Only instrument bedrock operations we care about
        if operation_name not in [
            "InvokeModel",
            "InvokeModelWithResponseStream",
            "Converse",
            "ConverseStream",
        ]:
            return wrapped(*args, **kwargs)

        return self._handle_bedrock_call(
            wrapped,
            instance,
            args,
            kwargs,
            operation=operation_name.lower()
            .replace("invokemodel", "invoke_model")
            .replace("conversestream", "converse_stream"),
        )

    def _is_bedrock_client(self, client) -> bool:
        """Check if the client is a bedrock-runtime client."""
        if not hasattr(client, "_service_model"):
            return False

        service_model = client._service_model
        if not hasattr(service_model, "service_name"):
            return False

        return service_model.service_name == "bedrock-runtime"

    def _handle_bedrock_call(self, wrapped, instance, args, kwargs, operation: str):
        """Handle instrumentation for any Bedrock API call."""

        # Check if instrumentation should be suppressed
        if AlignXTraceContext.should_suppress():
            return wrapped(*args, **kwargs)

        # Extract call data from the request
        request_dict = args[1] if len(args) > 1 else {}
        call_data = self.extract_call_data((), request_dict, operation)

        # Start span
        span = self._create_span(call_data, SpanKind.CLIENT)

        if not span:
            return wrapped(*args, **kwargs)

        start_time = time.time()

        try:
            # Make the actual API call
            response = wrapped(*args, **kwargs)

            # Handle streaming response
            if operation in ["invoke_model_with_response_stream", "converse_stream"]:
                return self._wrap_stream_response(response, span, call_data, start_time)

            # Handle regular response
            end_time = time.time()
            response_data = self.extract_response_data(response, operation)
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

        # Extract the streaming body
        if isinstance(response, dict) and "body" in response:
            streaming_body = response["body"]

            if hasattr(streaming_body, "iter_events"):
                # Wrap the event stream
                original_iter_events = streaming_body.iter_events
                collected_events = []

                def wrapped_iter_events():
                    nonlocal collected_events
                    try:
                        for event in original_iter_events():
                            collected_events.append(event)
                            yield event

                        # Stream completed, finalize span
                        end_time = time.time()
                        combined_response = (
                            self.response_parser.combine_streaming_events(
                                collected_events
                            )
                        )
                        response_data = self.extract_response_data(
                            combined_response, call_data.operation
                        )
                        metrics = self.calculate_metrics(
                            call_data, response_data, start_time, end_time
                        )
                        self._finalize_span(span, call_data, response_data, metrics)

                    except Exception as e:
                        end_time = time.time()
                        self._handle_exception(span, call_data, e, start_time, end_time)
                        raise

                # Replace the iter_events method
                streaming_body.iter_events = wrapped_iter_events

        return response

    def extract_call_data(
        self, args: tuple, kwargs: dict, operation: str = "unknown"
    ) -> AlignXProviderCallData:
        """Extract structured data from Bedrock API call arguments."""
        return self.metrics_extractor.extract_call_data(args, kwargs, operation)

    def extract_response_data(
        self, response: Any, operation: str = "unknown"
    ) -> Dict[str, Any]:
        """Extract structured data from Bedrock API response."""
        return self.response_parser.extract_response_data(response, operation)

    def calculate_metrics(
        self,
        call_data: AlignXProviderCallData,
        response_data: Dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> AlignXProviderMetrics:
        """Calculate comprehensive metrics for the Bedrock API call."""

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
            provider="bedrock",
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
                "foundation_model": self._extract_foundation_model_provider(
                    call_data.model
                ),
                **(framework_context.metadata if framework_context else {}),
            },
        )

    def _extract_foundation_model_provider(self, model_id: str) -> str:
        """Extract the foundation model provider from the model ID."""
        if not model_id:
            return "unknown"

        model_id_lower = model_id.lower()

        if "anthropic" in model_id_lower:
            return "anthropic"
        elif "amazon" in model_id_lower or "titan" in model_id_lower:
            return "amazon"
        elif "cohere" in model_id_lower:
            return "cohere"
        elif "meta" in model_id_lower or "llama" in model_id_lower:
            return "meta"
        elif "mistral" in model_id_lower:
            return "mistral"
        elif "ai21" in model_id_lower:
            return "ai21"
        elif "stability" in model_id_lower:
            return "stability"
        else:
            return "unknown"
