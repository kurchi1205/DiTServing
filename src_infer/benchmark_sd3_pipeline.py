import torch
from diffusers import StableDiffusion3Pipeline
import time
import statistics


PROMPT = "a photo of a cat holding a sign that says hello world"
NUM_INFERENCE_STEPS = 50
HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 5.0


def load_pipe():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    return pipe


def compile_pipe(pipe):
    pipe.vae = torch.compile(pipe.vae, fullgraph=True, mode="reduce-overhead")
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=False, mode="reduce-overhead")
    return pipe


def benchmark(pipe, prompt, batch_size=1, steps=NUM_INFERENCE_STEPS, rounds=5, prefix=""):
    prompts = [prompt] * batch_size
    latencies = []

    for i in range(rounds):
        torch.cuda.synchronize()
        start = time.time()

        images = pipe(
            prompt=prompts,
            negative_prompt="",
            num_inference_steps=steps,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=GUIDANCE_SCALE,
        ).images

        torch.cuda.synchronize()
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"[{prefix}] Round {i+1}/{rounds} - Latency: {elapsed:.3f}s")


    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if rounds > 1 else 0.0
    print(f"[{prefix}] Batch size {batch_size} - Avg latency: {avg_latency:.3f}s ± {std_latency:.3f}s over {rounds} rounds\n")
    return avg_latency, std_latency


if __name__ == "__main__":
    rounds = 5

    print("Loading uncompiled pipeline...")
    pipe_uncompiled = load_pipe()
    print("Benchmarking uncompiled (1 request)...")
    benchmark(pipe_uncompiled, PROMPT, batch_size=1, rounds=rounds, prefix="uncompiled_1")

    print("Benchmarking uncompiled (batch of 6)...")
    benchmark(pipe_uncompiled, PROMPT, batch_size=6, rounds=rounds, prefix="uncompiled_6")

    del pipe_uncompiled
    torch.cuda.empty_cache()

    print("Loading and compiling pipeline")
    pipe_compiled = load_pipe()
    pipe_compiled = compile_pipe(pipe_compiled)

    print("Benchmarking compiled (1 request)")
    benchmark(pipe_compiled, PROMPT, batch_size=1, rounds=rounds, prefix="compiled_1")

    print("Benchmarking compiled (batch of 6)")
    benchmark(pipe_compiled, PROMPT, batch_size=6, rounds=rounds, prefix="compiled_6")
