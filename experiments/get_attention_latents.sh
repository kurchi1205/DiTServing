#!/bin/bash

# Create the results directory if it doesn't already exist
mkdir -p ../results/attention

# Loop from 3 to 27
for i in {3..27}
do
    echo "Running inference and plotting for attention_scores_layer: $i"
    
    # Navigate to the source inference directory
    cd ../src_infer
    
    # Run the inference script with the current layer number
    python infer_pipeline_with_att_scores.py --attention_scores_layer $i
    
    # Navigate to the experiments directory
    cd ../experiments
    
    # Run the plotting script with the current layer number
    python plot_attention_scores.py --attention_scores_layer $i
    
    # Print completion of current task
    echo "Completed layer $i"
    
    # Navigate back to the starting directory if needed, change this path as necessary
    cd -  # Or specify the path directly if the directory layout is complex
done

echo "All processes completed."
