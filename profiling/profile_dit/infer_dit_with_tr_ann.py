import torch
import os

from nvtx import annotate
from diffusers import DPMSolverMultistepScheduler
from base_pipeline import DitPipeline

@annotate(message="load", color="yellow")
def load_pipe():   
    pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

@annotate(message="Gen", color="red")
def generate(pipe, prompt, num_inference_steps):
    class_ids = pipe.get_label_ids([prompt, prompt])
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
# nsys profile --trace=nvtx,cuda,osrt -o transformer_profile python infer_dit_tr_ann.py
# nsys profile --trace=nvtx,cuda,osrt -o reports/batched_norm_profile python infer_dit_with_tr_ann.py
# nsys profile --trace=nvtx,cuda,osrt -o reports/attn_norm_profile python infer_dit_with_tr_ann.py

# ncu  --nvtx --kernel-name-base function --kernel-name "regex:sm80_xmma" -o profile_output python infer_dit_with_tr_ann.py
#  ncu --nvtx --nvtx-include "Domain A@Range A"