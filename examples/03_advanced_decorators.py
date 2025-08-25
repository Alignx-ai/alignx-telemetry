#!/usr/bin/env python3
"""
AlignX Telemetry SDK - Advanced Decorators Example

This example demonstrates the advanced tracing capabilities of AlignX telemetry:
1. Basic function tracing with @trace()
2. Async function tracing with @trace_async()
3. Class-level tracing with @trace_class()
4. Error handling and exception tracing
5. Custom attributes and span enrichment

Perfect for understanding all the tracing decorator options!
"""

import asyncio
import time
import alignx_telemetry


# 1. Basic function tracing
@alignx_telemetry.trace()
def simple_function():
    """Simple function with basic tracing."""
    print("Executing simple function...")
    time.sleep(0.1)
    return "simple_result"


@alignx_telemetry.trace(
    operation="complex_data_processing",
    attributes={
        "processor_type": "advanced",
        "version": "2.0",
        "environment": "development"
    }
)
def complex_function(data_size: int):
    """Function with custom operation name and attributes."""
    print(f"Processing {data_size} records...")
    
    # Simulate processing
    for i in range(data_size):
        time.sleep(0.01)  # Simulate work per record
    
    return {"processed_records": data_size, "status": "complete"}


# 2. Async function tracing
@alignx_telemetry.trace_async(
    operation="async_ai_processing",
    attributes={"model": "gpt-4", "streaming": False}
)
async def async_ai_function(prompt: str):
    """Async function demonstrating async tracing."""
    print(f"Processing async AI request: {prompt[:30]}...")
    
    # Simulate async AI call
    await asyncio.sleep(0.2)
    
    return {
        "prompt": prompt,
        "response": f"AI response to: {prompt}",
        "tokens_used": len(prompt.split()) * 3
    }


@alignx_telemetry.trace_async(
    operation="async_database_operation", 
    attributes={"database": "vector_db", "operation_type": "search"}
)
async def async_database_search(query: str):
    """Async database operation with tracing."""
    print(f"Searching database for: {query}")
    
    # Simulate database query
    await asyncio.sleep(0.1)
    
    return {
        "query": query,
        "results": ["result1", "result2", "result3"],
        "search_time_ms": 100
    }


# 3. Class-level tracing
@alignx_telemetry.trace_class(
    class_name_prefix="insurance_processor",
    method_attributes={
        "analyze_claims": {"operation_type": "analysis", "priority": "high"},
        "generate_report": {"operation_type": "reporting", "priority": "medium"},
        "send_notifications": {"operation_type": "communication", "priority": "low"}
    }
)
class InsuranceProcessor:
    """Example class with automatic method tracing."""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        print(f"Initialized InsuranceProcessor: {processor_id}")
    
    def analyze_claims(self, claims_data):
        """Analyze insurance claims - automatically traced."""
        print(f"Analyzing {len(claims_data)} claims...")
        time.sleep(0.15)  # Simulate analysis
        
        return {
            "claims_analyzed": len(claims_data),
            "total_value": sum(claim.get("amount", 0) for claim in claims_data),
            "recommendations": ["approve", "review", "deny"]
        }
    
    def generate_report(self, analysis_results):
        """Generate report - automatically traced.""" 
        print("Generating analysis report...")
        time.sleep(0.1)  # Simulate report generation
        
        return {
            "report_type": "claims_analysis",
            "summary": analysis_results,
            "generated_at": time.time()
        }
    
    def send_notifications(self, recipients):
        """Send notifications - automatically traced."""
        print(f"Sending notifications to {len(recipients)} recipients...")
        time.sleep(0.05)  # Simulate notification sending
        
        return {"notifications_sent": len(recipients), "status": "success"}
    
    def _internal_method(self):
        """Internal method - not traced (starts with _)."""
        print("This internal method is not traced")
        return "internal_result"


# 4. Error handling and exception tracing
@alignx_telemetry.trace(
    operation="error_prone_function",
    attributes={"error_simulation": True}
)
def function_with_errors(should_fail: bool = False):
    """Function demonstrating error tracing."""
    print("Executing function that might fail...")
    
    time.sleep(0.05)
    
    if should_fail:
        raise ValueError("Simulated error for tracing demonstration")
    
    return {"status": "success", "result": "no_errors"}


@alignx_telemetry.trace_async(operation="async_error_function")
async def async_function_with_errors(should_fail: bool = False):
    """Async function demonstrating async error tracing."""
    print("Executing async function that might fail...")
    
    await asyncio.sleep(0.1)
    
    if should_fail:
        raise RuntimeError("Simulated async error")
    
    return {"status": "async_success", "result": "no_async_errors"}


async def main():
    """Main demonstration function."""
    print("=" * 70)
    print("AlignX Telemetry SDK - Advanced Decorators Example")
    print("=" * 70)
    
    # Initialize telemetry
    print("\nInitializing telemetry...")
    telemetry = alignx_telemetry.init({
        "service_name": "advanced-decorators-example",
        "environment": "development", 
        "console_export": True
    })
    print("Telemetry initialized!")
    
    # 1. Basic function tracing
    print("\n" + "=" * 40)
    print("1. Basic Function Tracing")
    print("=" * 40)
    
    result1 = simple_function()
    print(f"Simple function result: {result1}")
    
    result2 = complex_function(5)
    print(f"Complex function result: {result2}")
    
    # 2. Async function tracing
    print("\n" + "=" * 40)
    print("2. Async Function Tracing")
    print("=" * 40)
    
    async_result1 = await async_ai_function("What is the meaning of life?")
    print(f"Async AI result: {async_result1}")
    
    async_result2 = await async_database_search("insurance policies")
    print(f"Async database result: {async_result2}")
    
    # 3. Class-level tracing
    print("\n" + "=" * 40)
    print("3. Class-Level Tracing")
    print("=" * 40)
    
    processor = InsuranceProcessor("PROC_001")
    
    # Sample claims data
    claims_data = [
        {"id": "claim1", "amount": 1500, "type": "auto"},
        {"id": "claim2", "amount": 3000, "type": "home"}, 
        {"id": "claim3", "amount": 800, "type": "health"}
    ]
    
    analysis = processor.analyze_claims(claims_data)
    print(f"Analysis result: {analysis}")
    
    report = processor.generate_report(analysis)
    print(f"Report result: {report}")
    
    notifications = processor.send_notifications(["admin@example.com", "user@example.com"])
    print(f"Notifications result: {notifications}")
    
    # Call internal method (not traced)
    internal_result = processor._internal_method()
    print(f"Internal method result: {internal_result}")
    
    # 4. Error handling demonstration  
    print("\n" + "=" * 40)
    print("4. Error Handling & Exception Tracing")
    print("=" * 40)
    
    # Successful case
    success_result = function_with_errors(should_fail=False)
    print(f"Success case: {success_result}")
    
    # Error case - exception will be traced
    try:
        function_with_errors(should_fail=True)
    except ValueError as e:
        print(f"Caught expected error (traced): {e}")
    
    # Async success case
    async_success = await async_function_with_errors(should_fail=False)
    print(f"Async success case: {async_success}")
    
    # Async error case
    try:
        await async_function_with_errors(should_fail=True)
    except RuntimeError as e:
        print(f"Caught expected async error (traced): {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DECORATORS DEMONSTRATED:")
    print("=" * 70)
    print("✓ @alignx_telemetry.trace() - Basic function tracing")
    print("✓ @alignx_telemetry.trace(...) - Custom operation name and attributes") 
    print("✓ @alignx_telemetry.trace_async() - Async function tracing")
    print("✓ @alignx_telemetry.trace_class() - Automatic class method tracing")
    print("✓ Exception handling - Automatic error recording in spans")
    print("✓ Custom attributes - Rich span metadata")
    
    print("\n" + "=" * 70)
    print("ADVANCED FEATURES:")
    print("=" * 70)
    print("• Automatic span naming (module.function)")
    print("• Custom operation names override defaults")
    print("• Rich attribute support (strings, numbers, booleans)")
    print("• Exception recording with stack traces")
    print("• Async/await compatibility")
    print("• Class-level instrumentation with per-method config")
    print("• Private method exclusion (methods starting with _)")
    
    print("\nADVANCED DECORATORS EXAMPLE COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())