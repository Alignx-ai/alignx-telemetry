#!/usr/bin/env python3
"""
AlignX Telemetry SDK - Basic Usage Example

This example demonstrates the core functionality of the consolidated AlignX Telemetry SDK:
1. Simple initialization 
2. AI metrics recording for business logic
3. Custom tracing with decorators
4. Manual span creation

Perfect for getting started with AlignX telemetry!
"""

import time
import alignx_telemetry


def example_business_function(user_id: str, query: str):
    """Example business function that processes user requests."""
    print(f"Processing request for user {user_id}: {query}")
    
    # Simulate some processing time
    time.sleep(0.1)
    
    # Simulate database operation
    time.sleep(0.05)
    
    return {
        "user_id": user_id,
        "query": query,
        "results": ["result1", "result2", "result3"],
        "processing_time": 0.15
    }


@alignx_telemetry.trace(
    operation="insurance_claim_analysis", 
    attributes={"processor": "claims", "version": "1.0"}
)
def analyze_insurance_claims(claims_data):
    """Example function using the @trace decorator."""
    print("Analyzing insurance claims...")
    
    # Simulate AI processing
    time.sleep(0.2)
    
    return {
        "claims_processed": len(claims_data),
        "total_amount": sum(claim.get("amount", 0) for claim in claims_data),
        "recommendations": ["approve", "review", "deny"]
    }


def main():
    print("=" * 60)
    print("AlignX Telemetry SDK - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize telemetry (simple dict config)
    print("\n1. Initializing telemetry...")
    
    telemetry = alignx_telemetry.init({
        "service_name": "basic-example-service",
        "environment": "development",
        "console_export": True  # Shows traces in console for testing
    })
    
    print("   SUCCESS: Telemetry initialized")
    
    # 2. Access AI metrics for custom recording
    print("\n2. Recording custom AI metrics...")
    
    ai_metrics = telemetry.get_ai_metrics()
    if ai_metrics:
        # Record a database operation
        ai_metrics.record_db_request(
            db_system="lancedb",
            operation="search_insurance_plans",
            success=True,
            duration_seconds=0.1
        )
        print("   SUCCESS: AI metrics recorded")
    else:
        print("   WARNING: AI metrics not available (needs OpenTelemetry deps)")
    
    # 3. Use the trace decorator
    print("\n3. Using @trace decorator...")
    
    sample_claims = [
        {"id": "claim1", "amount": 1000},
        {"id": "claim2", "amount": 2500},
        {"id": "claim3", "amount": 750}
    ]
    
    analysis_result = analyze_insurance_claims(sample_claims)
    print(f"   Analysis result: {analysis_result}")
    
    # 4. Manual span creation for custom tracing
    print("\n4. Creating custom spans...")
    
    tracer = telemetry.tracer
    if tracer:
        with tracer.start_as_current_span("business_logic_processing") as span:
            if span:
                span.set_attribute("operation_type", "user_request_processing")
                span.set_attribute("user_count", 1)
                
                # Process business logic
                result = example_business_function("user123", "find insurance plans")
                
                span.set_attribute("results_count", len(result["results"]))
                span.set_attribute("processing_success", True)
                
                print(f"   Business logic result: {result}")
                print("   SUCCESS: Custom span created")
            else:
                print("   WARNING: Span creation returned None")
    else:
        print("   WARNING: Tracer not available (needs OpenTelemetry deps)")
    
    # 5. Show the simple API summary
    print("\n" + "=" * 60)
    print("API SUMMARY - What you just used:")
    print("=" * 60)
    print("✓ alignx_telemetry.init(config)")
    print("✓ telemetry.get_ai_metrics()")  
    print("✓ ai_metrics.record_db_request(...)")
    print("✓ @alignx_telemetry.trace(...)")
    print("✓ telemetry.tracer")
    print("✓ tracer.start_as_current_span(...)")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("• Install OpenTelemetry deps: uv add opentelemetry-api opentelemetry-sdk")
    print("• Try the FastAPI example for web application integration")
    print("• Configure OTEL exporters for your observability platform")
    print("• Explore advanced decorators: @trace_async, @trace_class")
    
    print("\nBASIC USAGE EXAMPLE COMPLETE!")


if __name__ == "__main__":
    main()