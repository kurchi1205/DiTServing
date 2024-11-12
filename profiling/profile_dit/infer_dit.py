import torch
import os

from nvtx import annotate
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

@annotate(message="load", color="yellow")
def load_pipe():   
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

@annotate(message="Gen", color="blue")
def generate(pipe, prompt, num_inference_steps):
    class_ids = pipe.get_label_ids([prompt])
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps)
    image = output.images[0]  
    return image


if __name__ == "__main__":
    pipe = load_pipe()
    image = generate(pipe, "white shark", num_inference_steps=25)
    
# sm__throughput.avg.pct_of_peak_sustained_elapsed# ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python infer_dit.py

# export PATH=$PATH:/usr/local/cuda/NVIDIA-Nsight-Compute-2024.3/target/linux-desktop-glibc_2_11_3-x64
# nsys profile python infer_dit.py
# nsys stats report1.nsys-rep -o model_profile_summary
# nsys profile --trace=nvtx,cuda,osrt -o pipeline_profile python infer_dit.py

