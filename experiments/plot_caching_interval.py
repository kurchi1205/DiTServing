import matplotlib.pyplot as plt
import json

def plot_intervals(file_name, saving_path):
    data = json.load(open(file_name, "r"))
    keys = list(map(int, data.keys()))  # Convert dictionary keys to integers for sorting and plotting
    values = [data[str(k)] for k in keys]  # Fetch values in the order of sorted keys

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, marker='o', linestyle='-', color='b')  # 'o' for circular markers
    plt.xlabel('Layer')
    plt.ylabel('Optimal Caching Interval')
    plt.title('Optimal Caching Interval per Layer')
    plt.grid(True)  # Enable grid for better readability
    plt.xticks(range(min(keys), max(keys)+1))  # Set x-ticks to show every layer

    # Optional: Annotate points with their values
    for i, txt in enumerate(values):
        plt.annotate(txt, (keys[i], values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(saving_path, format="png")

if __name__=="__main__":
    file_name="../results/attention/optimal_interval.json"
    save_path="../results/attention/optimal_interval_plot.png"
    plot_intervals(file_name=file_name, saving_path=save_path)