#!/bin/bash

# Set the prompt, number of inference steps, and output file name
PROMPT="white shark"
NUM_INFERENCE_STEPS=50
OUTPUT_FILE="dit_infer_image_with_latents.jpg"

# Run the Python script
python infer_dit_pipeline_with_latents.py --prompt "$PROMPT" --num_inference_steps $NUM_INFERENCE_STEPS --output_file $OUTPUT_FILE