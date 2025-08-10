#!/bin/bash

# Set the model name, device, prompt, and output file
MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
DEVICE="cuda"
PROMPT="a squirrel driving a toy car"
OUTPUT_FILE="diffusers_sd3_4.png"

# Run the script
python infer_sd3_pipeline.py --model_name $MODEL_NAME --device $DEVICE --prompt "$PROMPT" --output_file $OUTPUT_FILE

# Run the script with compilation
python infer_sd3_pipeline.py --model_name $MODEL_NAME --device $DEVICE --prompt "$PROMPT" --compile --output_file $OUTPUT_FILE