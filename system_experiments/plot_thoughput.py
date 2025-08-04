import json
import matplotlib.pyplot as plt
import argparse

# Define a function to draw a bar graph
def draw_bar_graph(gpu_throughput, total_time_throughput, output_path):
    """
    Draw a bar graph comparing GPU throughput and total time throughput.

    Args:
        gpu_throughput (list): List of GPU throughput values.
        total_time_throughput (list): List of total time throughput values.
        output_path (str): Path to save the graph.
    """
    # Define x-axis values and labels
    x = range(1, len(gpu_throughput) + 1)
    labels = ["0.5", "1", "1.5", "2"]
    width = 0.35

    # Create the graph
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], gpu_throughput, width, label='Throughput (GPU time)', color='skyblue')
    plt.bar([i + width/2 for i in x], total_time_throughput, width, label='Throughput (Total time)', color='orange')

    # Add labels and title
    plt.xlabel("Request Rate Run")
    plt.ylabel("Requests per Minute")
    plt.title("Throughput Comparison: GPU vs Total Time")
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the graph
    plt.savefig(f"{output_path}")

# Define a function to get throughput from JSON files
def get_throughput_from_jsons(json_list, output_path):
    """
    Get throughput values from a list of JSON files and draw a graph.

    Args:
        json_list (list): List of JSON file paths.
        output_path (str): Path to save the graph.
    """
    gpu_throughput = []
    total_time_throughput = []

    # Iterate over JSON files
    for json_file in json_list:
        # Load JSON data
        data = json.load(open(json_file))
        throughput_meta = data["throughput"]
        total_time = throughput_meta["time_span_sec"] / 60
        total_requests = throughput_meta["total_completed_requests"]
        total_time_throughput.append(total_requests / total_time)
        gpu_throughput.append(throughput_meta["requests_per_gpu_minute"])

    # Draw the graph
    draw_bar_graph(gpu_throughput, total_time_throughput, output_path)

# Define the main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot throughput comparison graph.")
    parser.add_argument("--json_list", type=str, nargs="+", help="List of JSON file paths.")
    parser.add_argument("--output_path", type=str, help="Path to save the graph.")
    args = parser.parse_args()

    # Get throughput from JSON files and draw graph
    get_throughput_from_jsons(args.json_list, args.output_path)

# Run the main function
if __name__ == "__main__":
    main()