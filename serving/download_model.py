import torch
import os
try:
    from pipelines.base_pipeline_dit import DitPipeline
except:
    import sys
    sys.path.insert(0, "../")
    from pipelines.base_pipeline_dit import DitPipeline

from diffusers import DPMSolverMultistepScheduler

pipe_config = {
    "pipeline_type": "base"
}

def download_pipe(pipe_config_arg=None):
    global pipe_config
    if pipe_config_arg is not None:
        pipe_config = pipe_config_arg
    pipe_type = pipe_config["pipeline_type"]
    if pipe_type == "base":
        pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained("./DitModel")

if __name__=="__main__":
    download_pipe()