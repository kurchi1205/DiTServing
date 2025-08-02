#!/bin/bash

# Set the model name, device, prompt, and output file
MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
DEVICE="cuda"
PROMPT="A scenic view of mountains during sunset"
OUTPUT_FILE="generated_image.png"

# Run the script
python stable_diffusion_3_inference.py --model_name $MODEL_NAME --device $DEVICE --prompt "$PROMPT" --output_file $OUTPUT_FILE

# Run the script with compilation
python stable_diffusion_3_inference.py --model_name $MODEL_NAME --device $DEVICE --prompt "$PROMPT" --compile --output_file $OUTPUT_FILE