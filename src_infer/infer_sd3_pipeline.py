# Import necessary libraries
import os
import torch
from diffusers import StableDiffusion3Pipeline
import argparse

# Define a function to load the Stable Diffusion 3 pipeline
def load_pipe(model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "cuda") -> StableDiffusion3Pipeline:
    """
    Load the Stable Diffusion 3 pipeline from the specified model name and move it to the specified device.
    
    Args:
    - model_name (str): The name of the model to load. Defaults to "stabilityai/stable-diffusion-3-medium-diffusers".
    - device (str): The device to move the model to. Defaults to "cuda".
    
    Returns:
    - pipe (StableDiffusion3Pipeline): The loaded pipeline.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe


# Define a function to compile the pipeline for better performance
def compile_pipe(pipe: StableDiffusion3Pipeline) -> StableDiffusion3Pipeline:
    """
    Compile the pipeline for better performance.
    
    Args:
    - pipe (StableDiffusion3Pipeline): The pipeline to compile.
    
    Returns:
    - pipe (StableDiffusion3Pipeline): The compiled pipeline.
    """
    # Compile the VAE with full graph and reduce-overhead mode
    pipe.vae = torch.compile(pipe.vae, fullgraph=True, mode="reduce-overhead")
    # Compile the transformer with non-full graph and reduce-overhead mode
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=False, mode="reduce-overhead")
    return pipe


# Define a function to perform inference with the pipeline
def infer(pipe: StableDiffusion3Pipeline, prompt: str, num_inference_steps: int) -> torch.Tensor:
    """
    Perform inference with the pipeline.
    
    Args:
    - pipe (StableDiffusion3Pipeline): The pipeline to use.
    - prompt (str): The prompt to use for inference.
    - num_inference_steps (int): The number of inference steps to perform.
    
    Returns:
    - image (torch.Tensor): The generated image.
    """
    # Perform inference with the pipeline
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        height=512,
        width=512,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(50)
    ).images[0]
    return image


# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stable Diffusion 3 Inference")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for inference")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline")
    parser.add_argument("--output_file", type=str, default="generated_image.png", help="Output file name")
    args = parser.parse_args()

    # Create the outputs folder if it doesn't exist
    outputs_folder = "outputs"
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    # Load the pipeline
    pipe = load_pipe(args.model_name, args.device)

    # Compile the pipeline if desired
    if args.compile:
        pipe = compile_pipe(pipe)

    # Perform inference with the pipeline
    image = infer(pipe, args.prompt, args.num_inference_steps)

    # Save the generated image
    output_file = os.path.join(outputs_folder, args.output_file)
    image.save(output_file)
