import torch
import os
try:
    from pipelines.base_pipeline_dit import DitPipeline
except:
    from ..pipelines.base_pipeline_dit import DitPipeline

from diffusers import DPMSolverMultistepScheduler

pipe_config = {
    "pipeline_type": "base"
}

def load_pipe(pipe_config_arg=None):
    global pipe_config
    if pipe_config_arg is not None:
        pipe_config = pipe_config_arg
    pipe_type = pipe_config["pipeline_type"]
    if pipe_type == "base":
        pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe


def generate(pipe, prompt, num_inference_steps):
    class_ids = pipe.get_label_ids([prompt])
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps)
    image = output.images[0]  
    return image


if __name__ == "__main__":
    pipe = load_pipe()
    image = generate(pipe, "white shark", num_inference_steps=25)
    os.makedirs("../results", exist_ok=True)
    image.save("../results/base_infer_image.jpg")