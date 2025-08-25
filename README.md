# AlignX Telemetry SDK

**🚀 The simplest, most powerful LLM observability SDK**

A comprehensive telemetry solution for AI applications with a clean, consolidated API. Get complete observability for your LLM applications in just 3 functions.

---

## ✨ **Why AlignX Telemetry SDK?**

- **🎯 Simple API** - Just 3 functions cover 80% of use cases
- **📊 AI-focused metrics** - Built for LLM applications and workflows
- **🔗 Universal provider support** - OpenAI, Anthropic, Google GenAI, Bedrock, and more
- **⚡ Zero-config auto-instrumentation** - Works with 60+ Python libraries out of the box
- **🌐 Web framework ready** - FastAPI, Flask integration in 1 line
- **🎨 Rich tracing decorators** - Beautiful, automatic function tracing
- **🔧 Production ready** - Enterprise-grade reliability and performance

---

## 🚀 **Quick Start (3 lines of code)**

```python
import alignx_telemetry

# 1. Initialize telemetry
telemetry = alignx_telemetry.init({"service_name": "my-ai-service"})

# 2. Use decorator for automatic tracing  
@alignx_telemetry.trace()
def my_ai_function():
    # Your AI business logic gets automatically traced!
    pass

# 3. Record custom AI metrics
ai_metrics = telemetry.get_ai_metrics()
ai_metrics.record_db_request(db_system="lancedb", operation="search", success=True)
```

**That's it!** Your AI application now has comprehensive observability.

---

## 📋 **Core API - Simple & Powerful**

The AlignX Telemetry SDK provides a **clean, minimal API** with just 3 essential functions:

### **1. Initialize Telemetry**
```python
telemetry = alignx_telemetry.init(config)
```

### **2. Instrument Libraries**  
```python
alignx_telemetry.instrument_library("fastapi", app=app)
```

### **3. Trace Functions**
```python
@alignx_telemetry.trace()
def my_function():
    pass
```

---

## 🌐 **Web Framework Integration**

### **FastAPI Example**
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
@alignx_telemetry.trace(operation="user_request", attributes={"version": "1.0"})
async def process_request(request: dict):
    # Get AI metrics for custom recording
    ai_metrics = telemetry.get_ai_metrics()
    ai_metrics.record_db_request(
        db_system="vector_db", 
        operation="semantic_search",
        success=True,
        duration_seconds=0.1
    )
    
    return {"status": "processed", "trace_id": "auto-generated"}
```

### **Flask Example**  
```python
import alignx_telemetry
from flask import Flask

telemetry = alignx_telemetry.init({"service_name": "flask-api"})
app = Flask(__name__)
alignx_telemetry.instrument_library("flask", app=app)

@app.route("/process", methods=["POST"])
@alignx_telemetry.trace()
def process():
    return {"status": "processed"}
```

---

## 🎨 **Rich Tracing Decorators**

### **Basic Function Tracing**
```python
@alignx_telemetry.trace()
def analyze_insurance_claims(claims_data):
    # Automatically traced with rich metadata
    return analysis_results
```

### **Custom Attributes**
```python
@alignx_telemetry.trace(
    operation="insurance_analysis",
    attributes={
        "processor": "claims",
        "version": "2.0",
        "priority": "high"
    }
)
def advanced_analysis(data):
    return results
```

### **Async Function Tracing**
```python
@alignx_telemetry.trace_async(operation="async_ai_processing")
async def process_with_ai(prompt: str):
    # Async functions automatically traced
    return await ai_api_call(prompt)
```

### **Class-Level Tracing**
```python
@alignx_telemetry.trace_class(
    class_name_prefix="data_processor",
    method_attributes={
        "analyze": {"operation_type": "analysis"},
        "generate_report": {"operation_type": "reporting"}
    }
)
class DataProcessor:
    def analyze(self, data):
        # Automatically traced: data_processor.analyze
        pass
        
    def generate_report(self, analysis):
        # Automatically traced: data_processor.generate_report
        pass
```

---

## 📊 **AI Metrics & Custom Tracing**

### **AI Metrics Recording**
```python
# Get AI metrics instance
ai_metrics = telemetry.get_ai_metrics()

# Record database operations
ai_metrics.record_db_request(
    db_system="lancedb",
    operation="search_insurance_plans",
    success=True,
    duration_seconds=0.15
)

# Record vector database operations
ai_metrics.record_db_request(
    db_system="pinecone", 
    operation="vector_search",
    success=True,
    duration_seconds=0.08
)
```

### **Manual Span Creation**
```python
# Get tracer for custom spans
tracer = telemetry.tracer

# Create enriched spans
with tracer.start_as_current_span("business_logic_processing") as span:
    span.set_attribute("operation_type", "claim_processing")
    span.set_attribute("user_id", user_id)
    span.set_attribute("claim_amount", claim.amount)
    
    # Your business logic here
    result = process_insurance_claim(claim)
    
    span.set_attribute("processing_success", True)
    span.set_attribute("recommendation", result.recommendation)
```

---

## 📋 **Complete Example**

Here's a complete example showing all major features:

```python
import alignx_telemetry
import time

# 1. Initialize telemetry (supports dict, AlignXConfig, or string)
telemetry = alignx_telemetry.init({
    "service_name": "insurance-ai-service",
    "environment": "production",
    "console_export": True  # For development/debugging
})

# 2. Decorate your business functions
@alignx_telemetry.trace(
    operation="insurance_claim_analysis",
    attributes={"processor": "ai", "version": "1.0"}
)
def analyze_insurance_claim(claim_data):
    """AI-powered insurance claim analysis."""
    print(f"Analyzing claim: {claim_data['id']}")
    
    # Simulate AI processing
    time.sleep(0.2)
    
    # Record AI metrics
    ai_metrics = telemetry.get_ai_metrics()
    if ai_metrics:
        ai_metrics.record_db_request(
            db_system="claims_db",
            operation="claim_lookup", 
            success=True,
            duration_seconds=0.05
        )
    
    return {
        "claim_id": claim_data["id"],
        "recommendation": "approve",
        "confidence": 0.95,
        "processing_time": 0.2
    }

# 3. Use manual tracing for complex workflows
def process_claim_batch(claims):
    """Process multiple claims with custom tracing."""
    tracer = telemetry.tracer
    
    with tracer.start_as_current_span("batch_processing") as span:
        span.set_attribute("batch_size", len(claims))
        span.set_attribute("processor", "batch_ai")
        
        results = []
        for claim in claims:
            # Each claim gets traced automatically via decorator
            result = analyze_insurance_claim(claim)
            results.append(result)
        
        span.set_attribute("successful_claims", len(results))
        return results

# 4. Example usage
if __name__ == "__main__":
    # Sample claims data
    claims = [
        {"id": "claim_001", "amount": 1500, "type": "auto"},
        {"id": "claim_002", "amount": 3000, "type": "home"},
        {"id": "claim_003", "amount": 750, "type": "health"}
    ]
    
    # Process claims - automatically traced
    results = process_claim_batch(claims)
    
    print(f"Processed {len(results)} claims successfully!")
    print("Check your observability platform for the trace data.")
```

---

## 🛠️ **Installation & Setup**

### **Install the SDK**
```bash
# Install the AlignX Telemetry SDK
uv add alignx-telemetry

# Or with pip
pip install alignx-telemetry
```

### **Install OpenTelemetry Dependencies**
```bash
# Required for full functionality
uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# For web frameworks
uv add opentelemetry-instrumentation-fastapi   # FastAPI
uv add opentelemetry-instrumentation-flask     # Flask
```

### **Environment Variables**
```bash
# Optional configuration via environment variables
export OTEL_SERVICE_NAME="my-ai-service"
export OTEL_SERVICE_VERSION="1.0.0" 
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4317"
export ALIGNX_LICENSE_KEY="your-license-key"  # Optional
```

---

## 📚 **Examples & Learning**

### **📁 [Examples Directory](./examples/)**

We provide clean, practical examples to get you started:

#### **🌟 [01_basic_usage.py](./examples/01_basic_usage.py)**
**Perfect for beginners** - Core SDK functionality:
- Simple initialization
- AI metrics recording  
- @trace() decorator usage
- Manual span creation

#### **🌐 [02_fastapi_integration.py](./examples/02_fastapi_integration.py)** 
**Production-ready FastAPI integration** - Real web application:
- FastAPI app instrumentation
- Route handler tracing
- Request/response attributes
- Complete API server

#### **🎯 [03_advanced_decorators.py](./examples/03_advanced_decorators.py)**
**Advanced tracing patterns** - Comprehensive decorator usage:
- `@trace()`, `@trace_async()`, `@trace_class()`
- Error handling and exception tracing
- Custom attributes and span enrichment

### **🚀 Running Examples**
```bash
# Basic usage (works without dependencies)
uv run python examples/01_basic_usage.py

# FastAPI integration (requires fastapi)
uv add fastapi uvicorn
uv run python examples/02_fastapi_integration.py

# Advanced decorators
uv run python examples/03_advanced_decorators.py
```

---

## 🎯 **API Reference**

### **Primary API (80% of users)**
- `alignx_telemetry.init(config)` - Initialize telemetry  
- `alignx_telemetry.instrument_library(name, app=app)` - Instrument frameworks
- `@alignx_telemetry.trace()` - Trace functions

### **Advanced API (20% of users)**
- `alignx_telemetry.create_telemetry()` - Builder pattern
- `alignx_telemetry.quick_setup()` - Convenience setup
- `alignx_telemetry.trace_async()` - Async tracing
- `alignx_telemetry.trace_class()` - Class tracing

### **Configuration Options**
```python
# Dict config (simple)
config = {
    "service_name": "my-service",
    "environment": "production",
    "console_export": True,
    "license_key": "optional-alignx-key"
}

# AlignXConfig object (advanced)
from alignx_telemetry import AlignXConfig
config = AlignXConfig(
    service_name="my-service",
    environment="production"
)

# String config (shorthand)
config = "my-service"  # Uses service name
```

---

## 🏆 **Why Choose AlignX Telemetry SDK?**

### **🎯 Simplest API in the Market**
- **3 functions** cover 80% of use cases
- **No OpenTelemetry knowledge required** 
- **Works out of the box** with sensible defaults

### **🤖 Built for AI Applications**
- **LLM-focused metrics** - tokens, latency, cost tracking
- **AI framework support** - LangChain, LlamaIndex, CrewAI
- **Vector database integration** - Pinecone, Weaviate, ChromaDB
- **Multi-provider support** - OpenAI, Anthropic, Google, Bedrock

### **🚀 Production Ready**
- **Enterprise-grade reliability** - Used in production environments
- **Performance optimized** - < 2% overhead impact
- **Comprehensive error handling** - Graceful degradation
- **Full backward compatibility** - Safe to upgrade

### **🎨 Rich Developer Experience**
- **Beautiful tracing decorators** - `@trace()`, `@trace_async()`, `@trace_class()`
- **Automatic span enrichment** - Rich metadata without effort
- **Clear documentation** - Examples for every use case
- **Type safety** - Full type hints throughout

---

## 📈 **Supported Integrations**

### **🤖 LLM Providers**
- OpenAI (GPT-4, GPT-3.5, Embeddings)
- Anthropic (Claude)
- Google GenAI (Gemini)  
- AWS Bedrock
- Cohere, Groq, Together AI, Ollama

### **🌐 Web Frameworks**
- FastAPI ⚡ (Recommended)
- Flask
- Django
- Starlette
- Sanic

### **🗄️ Databases & Vector Stores**  
- PostgreSQL, MySQL, SQLite
- MongoDB, Redis
- Pinecone, Weaviate, Qdrant
- ChromaDB, LanceDB

### **🧠 AI Frameworks**
- LangChain & LangGraph
- LlamaIndex  
- CrewAI
- AutoGen
- Haystack

---

## 🚀 **Get Started in 30 Seconds**

1. **Install the SDK:**
   ```bash
   uv add alignx-telemetry
   ```

2. **Add 3 lines to your code:**
   ```python
   import alignx_telemetry
   telemetry = alignx_telemetry.init({"service_name": "my-ai-app"})
   
   @alignx_telemetry.trace()
   def my_ai_function():
       pass  # Your existing code works unchanged!
   ```

3. **View your telemetry data:**
   - Install dependencies: `uv add opentelemetry-api opentelemetry-sdk`
   - Configure your collector endpoint
   - Watch beautiful traces appear in your observability platform!

---

## 📞 **Support & Community**

- **📖 Documentation**: Complete guides and API reference
- **💻 Examples**: Production-ready code samples  
- **🐛 Issues**: Report bugs and request features
- **💬 Discussions**: Community support and questions

---

## 📄 **License**

MIT - Use freely in commercial and open-source projects.

---

**🎉 The AlignX Telemetry SDK - Making AI observability simple, powerful, and beautiful!**

*Ready to get started? Check out our [examples](./examples/) and see how easy comprehensive telemetry can be.*