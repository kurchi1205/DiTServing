import os
import time
from PIL import Image
from pipeline import SD3Inferencer
import numpy as np  # For statistical calculations

def benchmark_gen_image(prompt, iterations=10, model_path="models/sd3_medium.safetensors", model_folder="models/", output_dir="outputs", width=1024, height=1024, steps=30, cfg_scale=5.0, seed=42, sampler="dpmpp_2m"):
    """
    Benchmark the gen_image method of SD3Inferencer.

    Args:
        prompt (str): Text prompt for image generation.
        iterations (int): Number of iterations to run the benchmark.
        model_path (str), model_folder (str), output_dir (str),
        width (int), height (int), steps (int), cfg_scale (float),
        seed (int), sampler (str): Parameters for the SD3Inferencer and image generation.
    """
    # Initialize the inferencer
    inferencer = SD3Inferencer()
    inferencer.load(
        model=model_path,
        model_folder=model_folder,
        text_encoder_device="cpu",
        verbose=True
    )

    times = []  # List to store time taken for each iteration

    for i in range(iterations):
        print(f"Running iteration {i + 1}/{iterations}...")
        start_time = time.time()
        
        # Generate image
        images = inferencer.gen_image(
            prompts=[prompt],
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            seed_type="fixed",
            sampler=sampler
        )

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        print(f"Iteration {i + 1} completed in {elapsed_time:.2f} seconds.")

        # Optionally save each generated image
        if images:
            image_path = os.path.join(output_dir, f"generated_image_{i}.png")
            images[0].save(image_path)
            print(f"Image saved to {image_path}")

    # Calculate and print benchmark statistics
    average_time = np.mean(times)
    std_dev = np.std(times)
    max_time = np.max(times)
    min_time = np.min(times)
    median_time = np.median(times)

    print("\nBenchmarking Results:")
    print(f"Average Time: {average_time:.2f} seconds")
    print(f"Standard Deviation: {std_dev:.2f} seconds")
    print(f"Median Time: {median_time:.2f} seconds")
    print(f"Max Time: {max_time:.2f} seconds")
    print(f"Min Time: {min_time:.2f} seconds")

# Example of how to call the benchmark function
if __name__ == "__main__":
    benchmark_gen_image(
        prompt="A landscape painting in the style of Van Gogh",
        iterations=10,  # Change iterations as needed
    )
