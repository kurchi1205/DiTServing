import psutil
import time
import json
import os
from datetime import datetime
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)


def get_gpu_usage():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_mem = info.used / (1024 ** 2)  # MB
        total_mem = info.total / (1024 ** 2)
        return {"gpu_memory_used_MB": used_mem, "gpu_memory_total_MB": total_mem}
    except Exception as e:
        return {"gpu_error": str(e)}
    finally:
        try:
            nvmlShutdown()
        except:
            pass


def collect_system_metrics(interval=2, output_file="system_metrics.json"):
    metrics = []
    print("System monitoring started. Press Ctrl+C to stop...\n")
    try:
        while True:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            gpu = get_gpu_usage()

            record = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu,
                "ram_percent": ram,
                **gpu,
            }

            print(record)
            metrics.append(record)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n Monitoring stopped by user. Saving to file...")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    collect_system_metrics(interval=2, output_file="/home/fast-dit-serving/outputs/metrics.json")
