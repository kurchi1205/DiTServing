import json
import matplotlib.pyplot as plt

def draw_bar_graph(gpu_throughput, total_time_throughput, output_path):
    x = range(1, len(gpu_throughput) + 1)
    labels = ["0.5", "1", "1.5", "2"]
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], gpu_throughput, width, label='Throughput (GPU time)', color='skyblue')
    plt.bar([i + width/2 for i in x], total_time_throughput, width, label='Throughput (Total time)', color='orange')

    plt.xlabel("Request Rate Run")
    plt.ylabel("Requests per Minute")
    plt.title("Throughput Comparison: GPU vs Total Time")
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_path}")


def get_throughput_from_jsons(json_list, output_path):
    gpu_throughput = []
    total_time_throughput = []
    
    for json_file in json_list:
        data = json.load(open(json_file))
        throughput_meta = data["throughput"]
        total_time = throughput_meta["time_span_sec"] / 60
        total_requests = throughput_meta["total_completed_requests"]
        total_time_throughput.append(total_requests / total_time)
        gpu_throughput.append(throughput_meta["requests_per_gpu_minute"])

    draw_bar_graph(gpu_throughput, total_time_throughput, output_path)

if __name__ == '__main__':
    json_list = [
        "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_0.5_sec_100_a100.json",
        "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_1_sec_100_a100.json",
        "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_1.5_sec_100_a100.json",
        "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_2_sec_100_a100.json"
    ]
    get_throughput_from_jsons(json_list, "/home/DiTServing/system_experiments/outputs/throughput_plot.jpeg")
