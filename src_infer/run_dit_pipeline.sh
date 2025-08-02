#!/bin/bash

# Set the prompt, number of inference steps, and output file name
PROMPT="white shark"
NUM_INFERENCE_STEPS=50
OUTPUT_FILE="dit_infer_image.jpg"

# Run the Python script
python dit_pipeline.py --prompt "$PROMPT" --num_inference_steps $NUM_INFERENCE_STEPS --output_file $OUTPUT_FILE