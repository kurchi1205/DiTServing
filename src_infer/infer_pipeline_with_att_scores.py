# Import necessary libraries
import torch
import os
import argparse
import json
from typing import Dict, List, Optional, Tuple, Union
from pipelines.pipeline_dit_with_attn_scores import DitPipelineAttnScores
from diffusers import DPMSolverMultistepScheduler

# Define the pipeline configuration
pipe_config = {
    "pipeline_type": "base"
}

# Define a function to load the pipeline
def load_pipe(pipe_config_arg=None):
    """
    Load the pipeline based on the provided configuration.
    
    Args:
    - pipe_config_arg (dict): The pipeline configuration. Defaults to None.
    
    Returns:
    - pipe (DitPipelineAttnScores): The loaded pipeline.
    """
    global pipe_config
    if pipe_config_arg is not None:
        pipe_config = pipe_config_arg
    pipe_type = pipe_config["pipeline_type"]
    
    # Load the pipeline based on the pipeline type
    if pipe_type == "base":
        # Load the DitPipelineAttnScores from the facebook/DiT-XL-2-512 model
        pipe = DitPipelineAttnScores.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float32)
    
    # Update the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move the pipeline to the CUDA device
    pipe = pipe.to("cuda")
    
    return pipe

# Define a function to generate an image
def generate(pipe, prompt, num_inference_steps, attention_scores_layer):
    """
    Generate an image based on the provided prompt and pipeline.
    
    Args:
    - pipe (DitPipelineAttnScores): The pipeline to use.
    - prompt (str): The prompt to use for generation.
    - num_inference_steps (int): The number of inference steps to perform.
    - attention_scores_layer (int): The transformer layer from which to pull attention scores.
    
    Returns:
    - image (PIL Image): The generated image.
    """
    # Get the class IDs for the prompt
    class_ids = pipe.get_label_ids([prompt])
    
    # Generate the image
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, attention_scores_layer=attention_scores_layer)
    
    # Get the generated image
    image = output.images[0]
    
    return image

# Define a function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
    - args (argparse.Namespace): The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Generate images with attention scores from a specific layer.')
    parser.add_argument('--attention_scores_layer', type=int, default=2, help='Specifies the transformer layer from which to pull attention scores.')
    parser.add_argument('--prompt', type=str, help='Prompt for generating the image.', default="white shark")
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps.')
    parser.add_argument('--output_folder', type=str, default="../results", help='Folder to save the generated image and attention scores.')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Load the pipeline
    pipe = load_pipe()
    
    attention_scores_folder = os.path.join(args.output_folder, "outputs/attention_scores")
    # Generate the image
    image = generate(pipe, args.prompt, num_inference_steps=args.num_inference_steps, attention_scores_layer=args.attention_scores_layer, attention_score_folder=attention_scores_folder)
    
    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Save the generated image
    image.save(os.path.join(args.output_folder, "base_infer_image.jpg"))
    
    