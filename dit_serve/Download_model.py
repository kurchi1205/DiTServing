import torch
import json
from base_pipeline_dit import DitPipeline

pipe = DitPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.save_pretrained("./Dit_model")
config = json.load(open("./Dit_model/model_index.json", "r"))
config["id2label"] = pipe.id2label
with open("./Dit_model/model_index.json", "w") as f:
    json.dump(config, f)


