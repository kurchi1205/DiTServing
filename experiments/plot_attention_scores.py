import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_attn_scores(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Determine the number of timesteps and the grid size for subplots
    num_timesteps = len(data)
    num_layers = len(next(iter(data.values())))  # Number of layers from the first timestep entry
    subplot_grid = int(np.ceil(np.sqrt(num_timesteps)))  # Creating a square grid that fits all timesteps
    
    fig, axes = plt.subplots(subplot_grid, subplot_grid, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    layer_name = ""
    for index, (key, layers) in enumerate(data.items()):
        timestep = key
        # We assume all layers have the same shape and can be plotted in a single heatmap
        for layer_id, values in layers.items():
            # Convert the list of values into a numpy array and reshape appropriately if necessary
            matrix = np.array(values)
            matrix = matrix[0]
            if matrix.ndim == 1:
                # Assuming the data should be square rootable into a matrix
                size = int(np.sqrt(len(values)))
                matrix = matrix.reshape((size, size))
            
            # Plotting on the corresponding subplot
            ax = axes[index]
            cax = ax.matshow(np.log(matrix + 1e-6), cmap='viridis')
            ax.set_title(f"Timestep {timestep}, Layer {layer_id}")
            ax.axis('off')  # Turn off axis numbering and ticks
            layer_name = str(layer_id)


        # Only add colorbar to the last plot to avoid repetition and clutter
        if index == len(data) - 1:
            fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"../results/attention/attn_output_{layer_name}.jpeg")  # Save the figure to a file
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with attention scores from a specific layer.')
    parser.add_argument('--attention_scores_layer', type=int, default=2, help='Specifies the transformer layer from which to pull attention scores.')
    return parser.parse_args()

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    plot_attn_scores(f"../results/attention/attention_scores_{args.attention_scores_layer}.json")
