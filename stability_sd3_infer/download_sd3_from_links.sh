#!/bin/bash

# Set the access token
ACCESS_TOKEN="your_access_token_here"

# Set the links and output paths
LINKS=(
  "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/clip_g.safetensors model/clip_g.safetensors"
  "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/clip_l.safetensors model/clip_l.safetensors"
  "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/t5xxl_fp16.safetensors model/t5xxl.safetensors"
  "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors model/sd3_medium.safetensors"
)

# Run the Python script for each link
for link in "${LINKS[@]}"; do
  url=$(echo "$link" | cut -d' ' -f1)
  output_path=$(echo "$link" | cut -d' ' -f2-)
  python download_sd3.py -t $ACCESS_TOKEN -u $url -o $output_path
done