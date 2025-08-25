#!/usr/bin/env python3
"""
AlignX Telemetry SDK - FastAPI Integration Example

This example shows how to integrate AlignX telemetry with a FastAPI application:
1. Initialize telemetry at startup
2. Instrument the FastAPI app  
3. Use decorators in route handlers
4. Record AI metrics for business operations
5. Proper cleanup at shutdown

Run with: uv run python 02_fastapi_integration.py
Test with: curl -X POST http://localhost:8000/process -H "Content-Type: application/json" -d '{"text": "Hello world"}'

Requirements: uv add fastapi uvicorn
"""

import time
import alignx_telemetry

# Global telemetry instance
telemetry = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    # Request/Response models
    class ProcessRequest(BaseModel):
        text: str
        model: str = "gpt-4"
        
    class ProcessResponse(BaseModel):
        original_text: str
        processed_text: str
        model_used: str
        processing_time_ms: float
        trace_id: str
        
    # Initialize telemetry
    print("Initializing AlignX telemetry...")
    telemetry = alignx_telemetry.init({
        "service_name": "fastapi-example-service", 
        "environment": "development",
        "console_export": True
    })
    print("Telemetry initialized successfully!")
    
    # Create FastAPI app
    app = FastAPI(
        title="AlignX Telemetry FastAPI Example",
        description="Example showing AlignX telemetry integration with FastAPI",
        version="1.0.0"
    )
    
    # IMPORTANT: Instrument the FastAPI app
    print("Instrumenting FastAPI application...")
    success = alignx_telemetry.instrument_library("fastapi", app=app)
    if success:
        print("SUCCESS: FastAPI instrumentation complete")
    else:
        print("WARNING: FastAPI instrumentation failed (missing dependencies?)")
    
    @alignx_telemetry.trace(
        operation="text_processing",
        attributes={"service": "nlp", "version": "1.0"}
    )
    def process_text_with_ai(text: str, model: str) -> dict:
        """Simulate AI text processing with automatic tracing."""
        print(f"Processing text with {model}: {text[:50]}...")
        
        # Simulate AI API call
        time.sleep(0.1)
        
        # Record AI metrics
        ai_metrics = telemetry.get_ai_metrics()
        if ai_metrics:
            ai_metrics.record_db_request(
                db_system="vector_db",
                operation="semantic_search",
                success=True,
                duration_seconds=0.05
            )
        
        # Simulate text transformation
        processed = text.upper() + " [PROCESSED]"
        
        return {
            "original": text,
            "processed": processed,
            "model": model,
            "tokens_used": len(text.split()) * 2
        }
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"message": "AlignX Telemetry FastAPI Example", "status": "healthy"}
    
    @app.get("/health")
    async def health_check():
        """Detailed health check with telemetry status."""
        return {
            "status": "healthy",
            "service": "fastapi-example-service",
            "telemetry_initialized": telemetry is not None,
            "ai_metrics_available": telemetry.get_ai_metrics() is not None if telemetry else False
        }
    
    @app.post("/process", response_model=ProcessResponse)
    async def process_text(request: ProcessRequest):
        """Main processing endpoint with telemetry."""
        start_time = time.time()
        
        # Use custom tracing for the overall operation
        tracer = telemetry.tracer if telemetry else None
        
        if tracer:
            with tracer.start_as_current_span("process_request") as span:
                if span:
                    span.set_attribute("request.text_length", len(request.text))
                    span.set_attribute("request.model", request.model)
                    
                    try:
                        # Call the traced business logic function
                        result = process_text_with_ai(request.text, request.model)
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        span.set_attribute("response.processing_time_ms", processing_time)
                        span.set_attribute("response.tokens_used", result["tokens_used"])
                        span.set_attribute("operation.success", True)
                        
                        # Get trace ID for response
                        trace_id = f"{span.get_span_context().trace_id:032x}" if span.get_span_context() else "unknown"
                        
                        return ProcessResponse(
                            original_text=result["original"],
                            processed_text=result["processed"], 
                            model_used=result["model"],
                            processing_time_ms=processing_time,
                            trace_id=trace_id
                        )
                        
                    except Exception as e:
                        span.set_attribute("operation.success", False)
                        span.set_attribute("error.message", str(e))
                        raise HTTPException(status_code=500, detail=str(e))
                else:
                    # Fallback without span
                    result = process_text_with_ai(request.text, request.model)
                    processing_time = (time.time() - start_time) * 1000
                    
                    return ProcessResponse(
                        original_text=result["original"],
                        processed_text=result["processed"],
                        model_used=result["model"],
                        processing_time_ms=processing_time,
                        trace_id="no-tracing"
                    )
        else:
            # Fallback without telemetry
            result = process_text_with_ai(request.text, request.model)
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessResponse(
                original_text=result["original"],
                processed_text=result["processed"],
                model_used=result["model"],
                processing_time_ms=processing_time,
                trace_id="no-telemetry"
            )
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean shutdown with telemetry cleanup."""
        print("Shutting down FastAPI application...")
        if telemetry:
            # Note: In a real app, you might want to flush telemetry data here
            print("Telemetry cleanup would happen here")
    
    def main():
        """Run the FastAPI application."""
        print("\n" + "=" * 60)
        print("AlignX Telemetry FastAPI Integration Example")
        print("=" * 60)
        print("Starting FastAPI server...")
        print("API Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print("\nTest the API with:")
        print('curl -X POST http://localhost:8000/process \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"text": "Hello AlignX telemetry!", "model": "gpt-4"}\'')
        print("=" * 60)
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("\nTo run this example, install the required packages:")
    print("uv add fastapi uvicorn")
    print("\nThen run: uv run python 02_fastapi_integration.py")
except Exception as e:
    print(f"Error: {e}")
    if telemetry:
        print("Cleaning up telemetry...")
        # telemetry.shutdown() - if needed