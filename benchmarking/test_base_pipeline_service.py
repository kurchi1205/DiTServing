# pytest test_base_pipeline_service.py --benchmark-json benchmark_results.json

import pytest
import requests
import json
import time

def make_request():
    url = "http://localhost:8080/predictions/dit-model/"
    prompt = "white shark"
    response = requests.post(url, data=prompt)
    response.raise_for_status()  # Ensure we raise an exception on failed requests
    return response.json()


# Pytest benchmark test case
@pytest.mark.benchmark(
    group="API-Inference",  # Group related benchmarks for better reporting
    min_rounds=10  # Ensure at least 10 rounds are run for statistical relevance
)
def test_inference_latency(benchmark):
    """Benchmark the inference latency of the FastAPI endpoint."""
    # Use benchmark fixture to measure execution time of the request
    result = benchmark(make_request)

