import asyncio
import time
import statistics
from datetime import datetime
import sys
sys.path.append("../custom_serve/")
from client import Client


async def benchmark_client():
    """
    Benchmark the client's ability to handle requests and retrieve outputs.
    """
    # Initialize the client
    client = Client()

    # Number of requests and configurations
    total_requests = 50  # Total number of requests to send
    timesteps_left = 30  # Timesteps for each request
    prompts = [f"Benchmark image {i}" for i in range(total_requests)]

    # Start the background process
    await client.start_background_process()

    # Submit all requests
    tasks = [client.add_request(prompt, timesteps_left) for prompt in prompts]
    await asyncio.gather(*tasks)

    # Poll for completed requests
    completed_requests = []
    request_times = []  # Track time taken for each request
    while len(completed_requests) < total_requests:
        outputs = await client.get_output()
        if isinstance(outputs, list):
            for output in outputs:
                if output["request_id"] not in [req["request_id"] for req in completed_requests]:
                    completed_requests.append(output)
                    # Calculate time taken for this request
                    start_time = datetime.fromisoformat(output["timestamp"])
                    end_time = datetime.fromisoformat(output["time_completed"])
                    time_taken = (end_time - start_time).total_seconds()
                    request_times.append(time_taken)
        await asyncio.sleep(client.poll_interval)

    # Calculate statistics
    min_time = min(request_times)
    max_time = max(request_times)
    median_time = statistics.median(request_times)
    std_dev = statistics.stdev(request_times)

    # Print results
    print(f"--- Benchmark Results ---")
    print(f"Minimum time taken: {min_time:.2f} seconds")
    print(f"Maximum time taken: {max_time:.2f} seconds")
    print(f"Median time taken: {median_time:.2f} seconds")
    print(f"Standard deviation: {std_dev:.2f} seconds")


if __name__ == "__main__":
    try:
        asyncio.run(benchmark_client())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
