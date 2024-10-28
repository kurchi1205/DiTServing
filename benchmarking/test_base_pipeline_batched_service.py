# pytest test_base_pipeline_service.py --benchmark-json benchmark_results.json

import pytest
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_requests_concurrently(prompts):
    url = "http://localhost:8080/predictions/dit-model/"
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit all requests to the thread pool
        futures = [executor.submit(requests.post, url, data=prompt) for prompt in prompts]
        # Wait for all futures to complete
        results = [future.result() for future in as_completed(futures)]
    return results

@pytest.mark.benchmark(
    group="Batched-API-Inference",
    min_rounds=10
)
def test_batched_inference_latency(benchmark):
    """Benchmark the inference latency of the FastAPI endpoint for batched requests."""
    prompts = ["white shark", "spider web"]  # Example prompts

    # Benchmark the function that sends requests concurrently
    benchmark(send_requests_concurrently, prompts)

