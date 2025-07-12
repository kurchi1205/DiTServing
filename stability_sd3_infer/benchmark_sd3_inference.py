import os
import fire
import time
import statistics
from PIL import Image
from pipeline import SD3Inferencer

def run_benchmark(
    prompt="A scenic view of mountains during sunset",
    model_path="models/sd3_medium.safetensors",
    model_folder="models/",
    output_dir="outputs",
    width=1024,
    height=1024,
    steps=10,
    cfg_scale=5.0,
    seed=42,
    sampler="dpmpp_2m",
    runs=5
):
    """
    Benchmark SD3 image generation time over multiple runs.

    Args:
        prompt (str): Text prompt.
        model_path (str): Path to the model.
        model_folder (str): Folder for text encoder and others.
        output_dir (str): Where to save output images.
        width (int): Image width.
        height (int): Image height.
        steps (int): Denoising steps.
        cfg_scale (float): Classifier-free guidance scale.
        seed (int): Random seed.
        sampler (str): Sampling method.
        runs (int): Number of times to repeat inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    inferencer = SD3Inferencer()
    print("Loading model...")
    inferencer.load(
        model=model_path,
        model_folder=model_folder,
        text_encoder_device="cuda",
        verbose=False
    )

    print(f"\n🚀 Starting benchmark for prompt: \"{prompt}\"")
    times = []

    for i in range(runs):
        print(f"\nRun {i+1}/{runs}")
        start = time.time()
        images = inferencer.gen_image(
            prompts=[prompt],
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed + i,  # vary seed slightly
            seed_type="fixed",
            sampler=sampler
        )
        end = time.time()
        duration = end - start
        times.append(duration)
        print(f"Run {i+1} completed in {duration:.2f} seconds.")

        # Optionally save each output
        if images:
            out_path = os.path.join(output_dir, f"generated_image_run_{i+1}.png")
            images[0].save(out_path)

    # Summary
    print("\n === Benchmark Summary ===")
    print(f"Prompt: {prompt}")
    print(f"Runs: {runs}")
    print(f"Avg time: {statistics.mean(times):.2f} sec")
    print(f"Min time: {min(times):.2f} sec")
    print(f"Max time: {max(times):.2f} sec")
    print(f"Std dev : {statistics.stdev(times):.2f} sec")

    return times

if __name__ == "__main__":
    fire.Fire(run_benchmark)
