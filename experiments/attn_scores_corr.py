import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def get_correlation(attention_scores):
    flattened_scores = {timestep: [scores for layer, scores in layer_scores.items()]
                        for timestep, layer_scores in attention_scores.items()}
    all_scores = [np.log(np.array(val)[0].flatten() + 1e-6) for key, val in flattened_scores.items()]
    correlation_matrix = np.corrcoef(all_scores)
    return correlation_matrix


def plot_correlation_matrix(json_file, layer_name):
    with open(json_file, "r") as f:
        data = json.load(f)
    correlation_matrix = get_correlation(data)
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Matrix of Attention Scores Across Timesteps')
    plt.xlabel('Attention Scores Index')
    plt.ylabel('Attention Scores Index')
    plt.savefig(f"attention_results/attn_corr_{layer_name}.jpeg")  # Save the figure to a file
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with attention scores from a specific layer.')
    parser.add_argument('--attention_scores_layer', type=int, default=2, help='Specifies the transformer layer from which to pull attention scores.')
    return parser.parse_args()

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    plot_correlation_matrix(f"attention_results/attention_scores_{args.attention_scores_layer}.json", args.attention_scores_layer)