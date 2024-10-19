import torch
import sys
import os

sys.path.insert(0, "../")
from pipelines.base_pipeline_dit import DitPipeline
from diffusers import DPMSolverMultistepScheduler


def load_pipe():
    pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe


def generate(pipe, prompt, num_inference_steps):
    class_ids = pipe.get_label_ids([prompt])
    generator = torch.manual_seed(33)
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, generator=generator)
    image = output.images[0]  
    return image


if __name__ == "__main__":
    pipe = load_pipe()
    image = generate(pipe, "white shark", num_inference_steps=25)
    os.makedirs("../results", exist_ok=True)
    image.save("../results/base_infer_image.jpg")