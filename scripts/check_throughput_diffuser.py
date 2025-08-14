#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SD3 on first 100 prompts loaded from a JSON dictionary (or a {"prompts": [...]} list).
Runs sequentially and reports throughput.
"""

import json
import time
import argparse
from typing import List, Dict, Any

import torch
from diffusers import StableDiffusion3Pipeline


# ----------------------------
# Prompt loading
# ----------------------------
def load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "prompts" in data and isinstance(data["prompts"], list):
            for item in data["prompts"]:
                if isinstance(item, str):
                    normalized.append({"prompt": item})
                elif isinstance(item, dict) and "prompt" in item:
                    normalized.append({"prompt": item["prompt"], "timesteps": item.get("timesteps")})
        else:
            for _, value in data.items():
                if isinstance(value, str):
                    normalized.append({"prompt": value})
                elif isinstance(value, dict) and "prompt" in value:
                    normalized.append({"prompt": value["prompt"], "timesteps": value.get("timesteps")})

    if not normalized:
        raise ValueError("No prompts found in JSON.")
    return normalized[:10]


# ----------------------------
# Pipeline setup / inference
# ----------------------------
def load_pipe(model_name: str, device: str) -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    return pipe.to(device)


def compile_pipe(pipe: StableDiffusion3Pipeline) -> StableDiffusion3Pipeline:
    pipe.vae = torch.compile(pipe.vae, fullgraph=True, mode="reduce-overhead")
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=False, mode="reduce-overhead")
    return pipe


@torch.inference_mode()
def infer(pipe: StableDiffusion3Pipeline, prompt: str, num_inference_steps: int, seed: int = 50):
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        generator=generator,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run SD3 on first 100 prompts from JSON without saving images")
    parser.add_argument("--json", required=True, help="Path to JSON dict or {'prompts':[...]} file.")
    parser.add_argument("--model_name", default="stabilityai/stable-diffusion-3-medium-diffusers", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Steps per prompt")
    parser.add_argument("--compile", action="store_true", help="Compile VAE/transformer for speed")
    args = parser.parse_args()

    prompts_data = load_prompts(args.json)
    total = len(prompts_data)
    print(f"Loaded {total} prompts")

    pipe = load_pipe(args.model_name, args.device)
    if args.compile:
        pipe = compile_pipe(pipe)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize()

    successes = 0
    failures = 0
    per_prompt_times: List[float] = []

    batch_start = time.time()

    for i, item in enumerate(prompts_data, start=1):
        prompt = item["prompt"].strip()
        steps = int(item.get("timesteps", args.num_inference_steps))
        seed = 50 + i

        try:
            infer(pipe, prompt, steps, seed=seed)
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.synchronize()
            successes += 1
        except Exception as e:
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.synchronize()
            failures += 1

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize()
    total_wall = time.time() - batch_start

    throughput_succ = successes / total_wall if total_wall > 0 else 0
    throughput_total = total / total_wall if total_wall > 0 else 0

    print("\n========== Summary ==========")
    print(f"Prompts attempted : {total}")
    print(f"Successes / Fail. : {successes} / {failures}")
    print(f"Total wall time   : {total_wall:.2f}s")
    print(f"Throughput (succ) : {throughput_succ:.3f} img/s")
    print(f"Throughput (total): {throughput_total:.3f} img/s")
    print("=============================")


if __name__ == "__main__":
    main()
