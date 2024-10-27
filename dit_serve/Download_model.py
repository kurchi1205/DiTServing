import torch
from diffusers import DiffusionPipeline, DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.save_pretrained("./Dit_model")
