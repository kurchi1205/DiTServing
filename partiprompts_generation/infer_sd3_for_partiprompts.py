import torch
import json
import time
from pathlib import Path
from diffusers import StableDiffusion3Pipeline


def load_pipe():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe

def compile_pipe(pipe):
    pipe.vae = torch.compile(pipe.vae, fullgraph=True, mode="reduce-overhead")
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=False, mode="reduce-overhead")
    return pipe

def infer(pipe, prompt, num_inference_steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        generator=generator
    ).images[0]
    return image

def run_batch_inference(prompt_json_path, output_dir, num_inference_steps=28, compile=True, seed=42):
    torch.manual_seed(seed)

    # Load prompts
    with open(prompt_json_path, "r") as f:
        prompts = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and compile pipeline
    pipe = load_pipe()
    if compile:
        pipe = compile_pipe(pipe)

    print(f"Running inference on {len(prompts)} prompts...")

    for key, prompt in prompts.items():
        image = infer(pipe, prompt, num_inference_steps, seed)
        image.save(output_dir / f"{key}.jpeg")


if __name__ == "__main__":
    import fire
    fire.Fire(run_batch_inference)
