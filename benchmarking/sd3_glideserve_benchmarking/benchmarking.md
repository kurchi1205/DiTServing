# Benchmark Suite
Comprehensive performance testing tools for the SD3 inference server to measure latency, throughput, and quality metrics.

## 📊 Available Benchmarks

### Single Inference Latency
**Description:**  
Measures the processing time of individual requests and provides statistical analysis.

**Run:**
```bash
python test_sd3_serve_with_model_latency.py
```
What it tests:
- End-to-end latency for single requests
- Performance consistency across iterations
- Statistical analysis (mean, median, percentiles)

### Batch Processing Latency
**Description:**  
Evaluates server efficiency when handling multiple concurrent requests.

**Run:**
```bash
python test_sd3_serve_with_model_latency_batched.py
```
What it tests:
- Batch processing efficiency vs sequential processing
- Individual request latencies within batches


# FID check for different caching interval
python test_custom_service_with_cache.py

