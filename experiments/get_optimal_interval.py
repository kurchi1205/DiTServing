import numpy as np
import json
import argparse
import os

def get_correlation(file_name):
    attention_scores = json.load(open(file_name, "r"))
    # Flatten attention scores per timestep
    flattened_scores = {timestep: [scores for layer, scores in layer_scores.items()]
                        for timestep, layer_scores in attention_scores.items()}
    all_scores = [np.log(np.array(val)[0].flatten() + 1e-6) for key, val in flattened_scores.items()]
    correlation_matrix = np.corrcoef(all_scores)
    return correlation_matrix


def find_optimal_caching_interval(correlation_matrix, threshold=0.9):
    # print(correlation_matrix)
    num_timesteps = correlation_matrix.shape[0]
    min_trials = 1
    for interval in range(1, num_timesteps):
        trial = 0
        # print(interval)
        # Check correlations at intervals of `interval`
        high_corr = True
        for i in range(0, num_timesteps - interval):
            if correlation_matrix[i, i + interval] < threshold:
                trial += 1
                if trial > min_trials:
                    high_corr = False
                
        
        # If all correlations at this interval are above threshold, return interval
        if not high_corr:
            return interval - 1

    # Return -1 if no suitable interval found
    return -1


def update_json_file(file_path, layer, optimal_interval):
    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Initialize data dictionary
    data = {}
    
    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Open and try to read the JSON data
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print("JSON data in {} is corrupted. Starting fresh.".format(file_path))
                data = {}
    
    # Update the data dictionary with the new layer information
    data[layer] = optimal_interval

    # Write the updated dictionary back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with attention scores from a specific layer.')
    parser.add_argument('--layer', type=int, default=2, help='Specifies the transformer layer from which to pull attention scores.')
    parser.add_argument('--thresh', type=float, default=0.7)
    return parser.parse_args()

if __name__ =="__main__":
    args = parse_args()
    file_name = f"../results/attention/attention_scores_{args.layer}.json"
    correlation_matrix = get_correlation(file_name)
    optimal_interval = find_optimal_caching_interval(correlation_matrix, threshold=args.thresh)
    update_json_file("../results/attention/optimal_interval.json", args.layer, optimal_interval)
    if optimal_interval != -1:
        print(f"The optimal caching interval is {optimal_interval} timesteps.")
    else:
        print("No suitable caching interval found with the given threshold.")
