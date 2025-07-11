import json
from datetime import datetime
import numpy as np
from pathlib import Path

def parse_time(t):
    return datetime.fromisoformat(t)


def save_latency_throughput_metrics(
    total_latencies,
    queue_latencies,
    inference_latencies,
    all_end_times,
    time_span,
    throughput,
    output_file="latency_metrics.json"
):
    def safe_stats(data):
        if not data:
            return {"avg": None, "p95": None, "max": None}
        return {
            "avg": round(np.mean(data), 3),
            "p95": round(np.percentile(data, 95), 3),
            "max": round(np.max(data), 3)
        }

    metrics = {
        "latency": {
            "total": safe_stats(total_latencies),
            "queue": safe_stats(queue_latencies),
            "inference": safe_stats(inference_latencies)
        },
        "throughput": {
            "requests_per_second": round(throughput, 3),
            "time_span_sec": round(time_span, 2),
            "total_requests": len(all_end_times)
        }
    }
    try:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")


def analyze_requests(requests):
    total_latencies = []
    queue_latencies = []
    inference_latencies = []

    all_start_times = []
    all_end_times = []

    for req in requests:
        try:
            t_add = parse_time(req["timestamp"])
            t_start = parse_time(req["processing_time_start"])
            t_end = parse_time(req["time_completed"])

            total_latencies.append((t_end - t_add).total_seconds())
            queue_latencies.append((t_start - t_add).total_seconds())
            inference_latencies.append((t_end - t_start).total_seconds())

            all_start_times.append(t_add)
            all_end_times.append(t_end)
        except Exception as e:
            print(f"Skipping request {req.get('request_id', '?')} due to error: {e}")
            continue

    # Throughput
    if all_end_times:
        time_span = (max(all_end_times) - min(all_start_times)).total_seconds()
        throughput = len(all_end_times) / time_span if time_span > 0 else 0
    else:
        throughput = 0

    save_latency_throughput_metrics(
        total_latencies,
        queue_latencies,
        inference_latencies,
        all_end_times,
        time_span,
        throughput,
        output_file = "/home/DiTServing/outputs/throughput_metrics_4_req.json"
    )


if __name__ == "__main__":
    log_path = "/home/DiTServing/metrics/completed_requests_log_4_req.json"
    with open(log_path, "r") as f:
        data = json.load(f)
        if "completed_requests" in data:
            requests = data["completed_requests"]
        else:
            requests = data
        analyze_requests(requests)
