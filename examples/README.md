# AlignX Telemetry SDK - Examples

This directory contains simple, practical examples demonstrating the comprehensive capabilities of alignx-telemetry package.

## 🚀 **Quick Start**

The AlignX Telemetry SDK provides a simple 3-function API:

1. **`alignx_telemetry.init(config)`** - Initialize telemetry
2. **`alignx_telemetry.instrument_library("fastapi", app=app)`** - Instrument web frameworks  
3. **`@alignx_telemetry.trace()`** - Trace business functions

## 📁 **Examples**

### 🌟 [01_basic_usage.py](./01_basic_usage.py)
**Core SDK functionality - Start here!**

Demonstrates the essential AlignX telemetry features:
- ✅ Simple initialization with dict config
- ✅ AI metrics access and recording 
- ✅ @trace() decorator for automatic function tracing
- ✅ Manual span creation for custom tracing
- ✅ Proper error handling and fallbacks

```bash
# Run the basic example
uv run python 01_basic_usage.py
```

**Perfect for:** Understanding core concepts, getting started, learning the API

---

### 🌐 [02_fastapi_integration.py](./02_fastapi_integration.py)  
**Real-world FastAPI integration**

Shows how to integrate AlignX telemetry into a FastAPI web application:
- ✅ FastAPI app instrumentation with `instrument_library()`
- ✅ Route handler tracing with decorators
- ✅ Request/response tracing with custom attributes
- ✅ AI metrics recording for business operations
- ✅ Proper startup/shutdown handling

```bash  
# Install dependencies and run
uv add fastapi uvicorn
uv run python 02_fastapi_integration.py

# Test the API
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello AlignX!", "model": "gpt-4"}'
```

**Perfect for:** Web applications, API services, production deployments

---

### 🎯 [03_advanced_decorators.py](./03_advanced_decorators.py)
**Advanced tracing with decorators**

Comprehensive guide to all AlignX tracing decorators:
- ✅ `@trace()` - Basic function tracing with custom attributes
- ✅ `@trace_async()` - Async function tracing  
- ✅ `@trace_class()` - Automatic class method tracing
- ✅ Exception handling and error tracing
- ✅ Custom operation names and span enrichment

```bash
# Run the advanced decorators example
uv run python 03_advanced_decorators.py
```

**Perfect for:** Complex applications, microservices, advanced observability needs

---

## 🛠️ **Prerequisites**

### Basic Requirements
```bash
# The examples work with just the SDK (fallback mode)
# But for full functionality, add OpenTelemetry dependencies:
uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### Web Framework Requirements  
```bash
# For FastAPI example
uv add fastapi uvicorn

# For Flask (if you adapt the examples)
uv add flask
```

## 📋 **Common Patterns**

### 1. **Simple Service Setup**
```python
import alignx_telemetry

# Initialize once at startup
telemetry = alignx_telemetry.init({
    "service_name": "my-ai-service",
    "environment": "production"
})

# Use everywhere in your code
@alignx_telemetry.trace()
def my_business_function():
    pass
```

### 2. **FastAPI Integration**
```python
import alignx_telemetry
from fastapi import FastAPI

# Initialize telemetry
telemetry = alignx_telemetry.init({"service_name": "my-api"})

# Create and instrument app
app = FastAPI()
alignx_telemetry.instrument_library("fastapi", app=app)

# Use decorators in route handlers
@app.post("/process")
@alignx_telemetry.trace(operation="user_request_processing")
async def process_request(request: MyRequest):
    return {"status": "processed"}
```

### 3. **AI Metrics Recording**
```python
# Get AI metrics instance
ai_metrics = telemetry.get_ai_metrics()

# Record database operations
ai_metrics.record_db_request(
    db_system="lancedb",
    operation="search_insurance_plans", 
    success=True,
    duration_seconds=0.1
)
```

### 4. **Custom Tracing**
```python
# Get tracer for manual spans
tracer = telemetry.tracer

# Create custom spans
with tracer.start_as_current_span("business_logic") as span:
    span.set_attribute("operation_type", "data_processing")
    # Your code here
```

## 🎯 **API Summary**

The consolidated AlignX Telemetry SDK provides these key functions:

### **Primary API (80% of users)**
- `alignx_telemetry.init(config)` - Initialize telemetry
- `alignx_telemetry.instrument_library(name, app=app)` - Instrument frameworks
- `@alignx_telemetry.trace()` - Trace functions

### **Advanced API (20% of users)**
- `alignx_telemetry.create_telemetry()` - Builder pattern
- `alignx_telemetry.trace_async()` - Async tracing
- `alignx_telemetry.trace_class()` - Class tracing
- `alignx_telemetry.TelemetryManager` - Direct manager access

## 🚀 **Next Steps**

1. **Start with `01_basic_usage.py`** to understand core concepts
2. **Try `02_fastapi_integration.py`** if building web applications  
3. **Explore `03_advanced_decorators.py`** for sophisticated tracing needs
4. **Install OpenTelemetry dependencies** for full functionality
5. **Configure your observability platform** (Grafana, Jaeger, etc.)

## 📚 **Additional Resources**

- **Main README**: [../README.md](../README.md) - Complete SDK documentation
- **Configuration**: [../src/alignx_telemetry/configuration.py](../src/alignx_telemetry/configuration.py) - Config options

