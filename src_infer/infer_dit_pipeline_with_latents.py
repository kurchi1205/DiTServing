# Import necessary libraries
import torch
import os
import numpy as np
from PIL import Image
import argparse

# Try to import DitPipeline from pipelines.base_pipeline_dit
try:
    from pipelines.base_pipeline_dit import DitPipeline
except ImportError:
    # If the import fails, add the parent directory to the system path and try again
    import sys
    sys.path.insert(0, "../")
    from pipelines.base_pipeline_dit import DitPipeline

# Import DPMSolverMultistepScheduler from diffusers
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
    - pipe (DitPipeline): The loaded pipeline.
    """
    global pipe_config
    if pipe_config_arg is not None:
        pipe_config = pipe_config_arg
    pipe_type = pipe_config["pipeline_type"]
    
    # Load the pipeline based on the pipeline type
    if pipe_type == "base":
        # Load the DitPipeline from the facebook/DiT-XL-2-512 model
        pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)
    
    # Update the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move the pipeline to the CUDA device
    pipe = pipe.to("cuda")
    
    return pipe


# Define a function to generate an image
def generate(pipe, prompt, num_inference_steps, get_intermediate_latents=True):
    """
    Generate an image based on the provided prompt and pipeline.
    
    Args:
    - pipe (DitPipeline): The pipeline to use.
    - prompt (str): The prompt to use for generation.
    - num_inference_steps (int): The number of inference steps to perform.
    - get_intermediate_latents (bool): Whether to return intermediate latents. Defaults to True.
    
    Returns:
    - image (PIL Image): The generated image.
    """
    # Get the class IDs for the prompt
    class_ids = pipe.get_label_ids([prompt])
    
    # Generate the image
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, get_intermediate_latents=get_intermediate_latents)
    
    # If intermediate latents are requested, process them
    if get_intermediate_latents:
        # Get the intermediate latents
        intermediate_images = output.latents
        
        # Concatenate the intermediate images horizontally
        num_images = len(intermediate_images)
        single_image = np.concatenate(intermediate_images, axis=1)
        
        # Convert the image to a PIL Image
        final_image = Image.fromarray((single_image * 255).astype(np.uint8))
        
        return final_image
    
    # If intermediate latents are not requested, return the final image
    return output.images[0]


# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DitPipeline Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--get_intermediate_latents", action="store_true", help="Get intermediate latents")
    parser.add_argument("--output_file", type=str, default="../results/base_infer_image_all_steps.jpg", help="Output file name")
    args = parser.parse_args()

    # Create the outputs folder if it doesn't exist
    outputs_folder = "outputs"
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    # Load the pipeline
    pipe = load_pipe()

    # Generate the image
    image = generate(pipe, args.prompt, args.num_inference_steps, args.get_intermediate_latents)

    # Save the generated image
    image.save(args.output_file)