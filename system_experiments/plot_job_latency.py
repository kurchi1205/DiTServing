import json
import matplotlib.pyplot as plt

def draw_latency_line_plot(avg_total, avg_queue, avg_inference, output_path):
    request_rates = ["0.5", "1", "1.5", "2"]
    plt.figure(figsize=(10, 6))

    # Plot each latency metric
    plt.plot(request_rates, avg_total, marker='o', label='Total Latency (s)', color='royalblue')
    plt.plot(request_rates, avg_queue, marker='o', label='Queue Time (s)', color='darkorange')
    plt.plot(request_rates, avg_inference, marker='o', label='Inference Time (s)', color='seagreen')

    plt.xlabel("Request Rate (requests/sec)")
    plt.ylabel("Latency (seconds)")
    plt.title("Average Latency Breakdown vs Request Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_latency_from_jsons(json_list, output_path):
    inferency_latency_list = []
    queue_latency_list = []
    total_latency_list = []
    
    for json_file in json_list:
        data = json.load(open(json_file))
        latency_meta = data["latency"]
        total_latency = latency_meta["total"]["avg"]
        inferency_latency = latency_meta["inference"]["avg"]
        queue_latency = latency_meta["queue"]["avg"]

        total_latency_list.append(total_latency)
        inferency_latency_list.append(inferency_latency)
        queue_latency_list.append(queue_latency)

    draw_latency_line_plot(total_latency_list, queue_latency_list, inferency_latency_list, output_path)


if __name__ == '__main__':
    json_list = [
            "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_0.5_sec_100_a100.json",
            "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_1_sec_100_a100.json",
            "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_1.5_sec_100_a100.json",
            "/home/DiTServing/system_experiments/outputs/throughput_metrics_rr_2_sec_100_a100.json"
        ]
    get_latency_from_jsons(json_list, "/home/DiTServing/system_experiments/outputs/latency_plot.jpeg")