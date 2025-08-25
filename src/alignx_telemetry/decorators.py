"""
AlignX Telemetry decorators for tracing business logic.

This module provides convenient decorators for tracing function execution
with automatic span creation, attribute enrichment, and error handling.
"""

import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


def trace_function(
    operation: Optional[str] = None,
    attributes: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    tracer_name: str = "alignx_telemetry.decorator",
    record_exception: bool = True,
    set_status_on_exception: bool = True
) -> Callable:
    """Decorator to trace function execution with enhanced AlignX telemetry.
    
    This decorator creates an OpenTelemetry span for the decorated function,
    automatically capturing function metadata, custom attributes, and handling
    exceptions according to OpenTelemetry best practices.
    
    Args:
        operation: Custom operation name (defaults to function name with module)
        attributes: Dictionary of custom attributes to add to the span
        tracer_name: Name of the tracer to use (defaults to alignx_telemetry.decorator)
        record_exception: Whether to record exceptions in the span (default: True)
        set_status_on_exception: Whether to set span status to ERROR on exception (default: True)
        
    Returns:
        Decorated function with tracing capability
        
    Examples:
        Basic usage:
            @alignx_telemetry.trace()
            def process_data():
                return "processed"
                
        With custom operation name and attributes:
            @alignx_telemetry.trace(
                operation="data_processing",
                attributes={"version": "1.0", "environment": "production"}
            )
            def process_insurance_data(user_id: str):
                # Business logic here
                return results
                
        In a class method:
            class DataProcessor:
                @alignx_telemetry.trace(
                    operation="insurance_analysis", 
                    attributes={"processor_type": "insurance"}
                )
                def analyze_claims(self, claims_data):
                    # Business logic
                    return analysis_results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracer - this will use the global OpenTelemetry tracer provider
            tracer = trace.get_tracer(tracer_name)
            
            # Create span name - use custom operation or generate from function
            if operation:
                span_name = operation
            else:
                # Use module.function_name format for better organization
                module_name = getattr(func, '__module__', 'unknown')
                if module_name == '__main__':
                    span_name = func.__name__
                else:
                    # Shorten module name for cleaner span names
                    module_parts = module_name.split('.')
                    if len(module_parts) > 2:
                        # Keep only last 2 parts: ...package.module
                        short_module = '.'.join(module_parts[-2:])
                    else:
                        short_module = module_name
                    span_name = f"{short_module}.{func.__name__}"
            
            # Start the span
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__ or "unknown")
                    
                    # Add function signature info if available
                    if hasattr(func, '__qualname__'):
                        span.set_attribute("function.qualname", func.__qualname__)
                    
                    # Add custom attributes if provided
                    if attributes:
                        for key, value in attributes.items():
                            # Ensure attribute value is compatible with OpenTelemetry
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(f"custom.{key}", value)
                            else:
                                # Convert complex types to strings
                                span.set_attribute(f"custom.{key}", str(value))
                    
                    # Add argument count for debugging
                    span.set_attribute("function.args_count", len(args))
                    span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Mark successful execution
                    span.set_attribute("function.success", True)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Handle exception according to configuration
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error.type", type(e).__name__)
                    span.set_attribute("function.error.message", str(e))
                    
                    if record_exception:
                        # Record the full exception with stack trace
                        span.record_exception(e)
                    
                    if set_status_on_exception:
                        # Set span status to ERROR
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Re-raise the exception
                    raise
        
        # Preserve original function metadata
        wrapper.__wrapped__ = func
        return wrapper
    
    return decorator


# Create convenient aliases
trace = trace_function  # Primary alias: @alignx_telemetry.trace()
span = trace_function   # Alternative alias: @alignx_telemetry.span()


def trace_async(
    operation: Optional[str] = None,
    attributes: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    tracer_name: str = "alignx_telemetry.decorator",
    record_exception: bool = True,
    set_status_on_exception: bool = True
) -> Callable:
    """Async version of the trace decorator for async functions.
    
    This decorator provides the same functionality as trace_function but
    for async/await functions.
    
    Args:
        operation: Custom operation name (defaults to function name with module)
        attributes: Dictionary of custom attributes to add to the span
        tracer_name: Name of the tracer to use
        record_exception: Whether to record exceptions in the span
        set_status_on_exception: Whether to set span status to ERROR on exception
        
    Returns:
        Decorated async function with tracing capability
        
    Example:
        @alignx_telemetry.trace_async(
            operation="async_data_processing",
            attributes={"processing_type": "parallel"}
        )
        async def process_data_async():
            # Async business logic
            await some_async_operation()
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get tracer
            tracer = trace.get_tracer(tracer_name)
            
            # Create span name
            if operation:
                span_name = operation
            else:
                module_name = getattr(func, '__module__', 'unknown')
                if module_name == '__main__':
                    span_name = f"async.{func.__name__}"
                else:
                    module_parts = module_name.split('.')
                    if len(module_parts) > 2:
                        short_module = '.'.join(module_parts[-2:])
                    else:
                        short_module = module_name
                    span_name = f"async.{short_module}.{func.__name__}"
            
            # Start the span
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function metadata  
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__ or "unknown")
                    span.set_attribute("function.type", "async")
                    
                    if hasattr(func, '__qualname__'):
                        span.set_attribute("function.qualname", func.__qualname__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(f"custom.{key}", value)
                            else:
                                span.set_attribute(f"custom.{key}", str(value))
                    
                    # Add argument info
                    span.set_attribute("function.args_count", len(args))
                    span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    # Execute async function
                    result = await func(*args, **kwargs)
                    
                    # Mark success
                    span.set_attribute("function.success", True)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Handle exception
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error.type", type(e).__name__)
                    span.set_attribute("function.error.message", str(e))
                    
                    if record_exception:
                        span.record_exception(e)
                    
                    if set_status_on_exception:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    raise
        
        # Preserve metadata
        async_wrapper.__wrapped__ = func
        return async_wrapper
    
    return decorator


def trace_class(
    class_name_prefix: Optional[str] = None,
    method_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    tracer_name: str = "alignx_telemetry.class_decorator"
) -> Callable:
    """Class decorator to automatically trace all public methods.
    
    This decorator applies tracing to all public methods (not starting with _)
    of a class, with optional per-method configuration.
    
    Args:
        class_name_prefix: Optional prefix for span names (defaults to class name)
        method_attributes: Dictionary mapping method names to their custom attributes
        tracer_name: Name of the tracer to use
        
    Returns:
        Decorated class with all public methods traced
        
    Example:
        @alignx_telemetry.trace_class(
            class_name_prefix="insurance_processor",
            method_attributes={
                "analyze_claims": {"operation_type": "analysis"},
                "generate_report": {"operation_type": "reporting"}
            }
        )
        class InsuranceProcessor:
            def analyze_claims(self, claims):
                # This will be automatically traced
                pass
                
            def generate_report(self, analysis):
                # This will also be automatically traced  
                pass
                
            def _internal_method(self):
                # This won't be traced (starts with _)
                pass
    """
    def class_decorator(cls):
        class_name = class_name_prefix or cls.__name__.lower()
        
        # Get all methods that should be traced
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            
            # Only trace public callable methods
            if (not attr_name.startswith('_') and 
                callable(attr_value) and 
                not isinstance(attr_value, (staticmethod, classmethod))):
                
                # Get custom attributes for this method
                method_attrs = method_attributes.get(attr_name, {}) if method_attributes else {}
                
                # Create operation name
                operation_name = f"{class_name}.{attr_name}"
                
                # Apply trace decorator
                traced_method = trace_function(
                    operation=operation_name,
                    attributes=method_attrs,
                    tracer_name=tracer_name
                )(attr_value)
                
                # Replace the method
                setattr(cls, attr_name, traced_method)
        
        return cls
    
    return class_decorator