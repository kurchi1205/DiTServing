import json
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse
import sys


def parse_time(t):
    """Parses ISO formatted timestamp into a datetime object."""
    return datetime.fromisoformat(t)


def safe_stats(data):
    """Calculates average, p95, and max safely for a list of values."""
    if not data:
        return {"avg": None, "p95": None, "max": None}
    return {
        "avg": round(np.mean(data), 3),
        "p95": round(np.percentile(data, 95), 3),
        "max": round(np.max(data), 3)
    }


def save_latency_throughput_metrics(
    total_latencies,
    queue_latencies,
    inference_latencies,
    gpu_latencies,
    all_end_times,
    time_span,
    throughput_rps,
    throughput_per_gpu_min,
    failed_requests_count,
    output_file
):
    """Saves calculated latency and throughput metrics to a JSON file."""
    metrics = {
        "latency": {
            "total": safe_stats(total_latencies),
            "queue": safe_stats(queue_latencies),
            "inference": safe_stats(inference_latencies),
            "gpu_inference": safe_stats(gpu_latencies)
        },
        "throughput": {
            "requests_per_second": round(throughput_rps, 3),
            "gpu_minutes": round(sum(gpu_latencies) / 60, 3),
            "requests_per_gpu_minute": round(throughput_per_gpu_min, 3),
            "time_span_sec": round(time_span, 2),
            "failed_requests": failed_requests_count,
            "total_completed_requests": len(all_end_times)
        }
    }

    try:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")


def analyze_requests(requests, output_file):
    """
    Analyzes latency and throughput metrics from a list of request logs.
    """
    total_latencies, queue_latencies, inference_latencies, gpu_latencies = [], [], [], []
    all_start_times, all_end_times = [], []
    failed_requests_count = 0

    for req in requests:
        status = req.get("status", "").lower()
        if status == "failed":
            failed_requests_count += 1
            continue

        try:
            t_add = parse_time(req["timestamp"])
            t_start = parse_time(req["processing_time_start"])
            t_end = parse_time(req["time_completed"])

            total_latencies.append((t_end - t_add).total_seconds())
            queue_latencies.append((t_start - t_add).total_seconds())
            inference_latencies.append((t_end - t_start).total_seconds())

            # GPU time from request if present, fallback to inference time
            gpu_time = float(req.get("elapsed_gpu_time", (t_end - t_start).total_seconds()))
            gpu_latencies.append(gpu_time)

            all_start_times.append(t_add)
            all_end_times.append(t_end)

        except Exception as e:
            print(f"Skipping request {req.get('request_id', '?')} due to error: {e}")
            continue

    # Throughput calculations
    if all_end_times:
        time_span = (max(all_end_times) - min(all_start_times)).total_seconds()
        throughput_rps = len(all_end_times) / time_span if time_span > 0 else 0
    else:
        time_span, throughput_rps = 0, 0

    total_gpu_minutes = sum(gpu_latencies) / 60
    throughput_per_gpu_min = len(all_end_times) / total_gpu_minutes if total_gpu_minutes > 0 else 0

    # Save metrics
    save_latency_throughput_metrics(
        total_latencies,
        queue_latencies,
        inference_latencies,
        gpu_latencies,
        all_end_times,
        time_span,
        throughput_rps,
        throughput_per_gpu_min,
        failed_requests_count,
        output_file
    )



def main():
    parser = argparse.ArgumentParser(description="Analyze latency and throughput metrics from completed requests.")
    parser.add_argument("--log_file", required=True, help="Path to the completed requests JSON file.")
    parser.add_argument("--output_file", default="latency_metrics.json", help="Path to save the metrics output JSON.")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")

    try:
        with open(log_path, "r") as f:
            data = json.load(f)
            requests = data.get("completed_requests", data)
    except Exception as e:
        print(f"Failed to read log file: {e}")

    analyze_requests(requests, args.output_file)


if __name__ == "__main__":
    main()
