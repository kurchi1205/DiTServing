import torch
import os
import numpy as np
from PIL import Image

try:
    from pipelines.base_pipeline_dit import DitPipeline
except ImportError:
    import sys
    sys.path.insert(0, "../")
    from pipelines.base_pipeline_dit import DitPipeline

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

def generate(pipe, prompt, num_inference_steps, get_intermediate_latents=True):
    class_ids = pipe.get_label_ids([prompt])
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, get_intermediate_latents=get_intermediate_latents)
    images = output.images  # This should contain final and potentially intermediate images.
    if get_intermediate_latents:
        # Assuming intermediate latents are returned as images in the output
        intermediate_images = output.latents
        # Save each intermediate image to a single combined image
        num_images = len(intermediate_images)
        single_image = np.concatenate(intermediate_images, axis=1)  # Concatenate images horizontally
        final_image = Image.fromarray((single_image * 255).astype(np.uint8))
        return final_image
    return images[0]

if __name__ == "__main__":
    pipe = load_pipe()
    image = generate(pipe, "white shark", num_inference_steps=25)
    os.makedirs("../results", exist_ok=True)
    image.save("../results/base_infer_image_all_steps.jpg")
