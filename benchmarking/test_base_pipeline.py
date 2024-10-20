import pytest
import requests
import json
import time

def make_request():
    url = "http://127.0.0.1:8000/generate/"
    payload = {
        "prompt": "white shark",
        "num_inference_steps": 50
    }
    response = requests.post(url, json=payload)
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

    # Optionally, assert something in the result
    assert "message" in result
    assert "Image saved at" in result["message"]
