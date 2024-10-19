import requests
import time
import json
from pathlib import Path

# JSON file to store metadata
META_FILE = Path("./api_call_metadata.json")

def log_metadata(data):
    """Append metadata to the JSON file."""
    try:
        # Load existing data if the file exists
        if META_FILE.exists():
            with META_FILE.open("r") as f:
                metadata = json.load(f)
        else:
            metadata = []

        # Add new data and save back to file
        metadata.append(data)
        with META_FILE.open("w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        print(f"Failed to log metadata: {str(e)}")

def run_inference(n=10):
    url = "http://127.0.0.1:8000/generate/"
    payload = {
        "prompt": "white shark",
        "num_inference_steps": 50
    }

    total_time = 0
    successes = 0

    for i in range(n):
        try:
            print(f"Running inference {i + 1} of {n}...")
            start_time = time.time()

            # Send the POST request
            response = requests.post(url, json=payload)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            if response.status_code == 200:
                data = response.json()
                print(f"Success: {data['message']}")

                # Log successful inference metadata
                successes += 1

            else:
                print(f"Failed: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error: {str(e)}")
            # Log failure or error metadata
            metadata = {
                "prompt": payload["prompt"],
                "num_inference_steps": payload["num_inference_steps"],
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
            log_metadata(metadata)

    # Calculate and print performance metrics
    average_latency = total_time / n
    throughput = successes / total_time if total_time > 0 else 0

    print("\n=== Performance Metrics ===")
    print(f"Total Inferences: {n}")
    print(f"Successful Inferences: {successes}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Latency: {average_latency:.4f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")

    # Log summary to JSON file
    summary = {
        "total_inferences": n,
        "successful_inferences": successes,
        "total_time_seconds": round(total_time, 4),
        "average_latency_seconds": round(average_latency, 4),
        "throughput_requests_per_second": round(throughput, 2)
    }
    log_metadata(summary)

if __name__ == "__main__":
    run_inference(n=10)  # Adjust 'n' as needed
