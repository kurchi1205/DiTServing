from ast import List
import torch
import os
import zipfile
import logging
import numpy as np
import diffusers
from diffusers import DPMSolverMultistepScheduler
from pydantic import BaseModel

try:
    from pipelines.base_pipeline_dit import DitPipeline
except:
    import sys
    sys.path.insert(0, "../")
    from pipelines.base_pipeline_dit import DitPipeline
from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class PromptModel(BaseModel):
    prompt: str | List[str]
    num_inference_steps: int

class DitHandler(BaseHandler):
    def __init__(self):
        super(DitHandler, self).__init__()
        self.initialized = False

    def load_pipe(self, model_dir):
        self.pipe = DitPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def initialize(self, ctx):   
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        self.load_pipe(model_dir + "/model")
        self.pipe.to(self.device)
        logger.info("Dit model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        req = []
        for _, data in enumerate(requests):
            input_req = data.get("data")
            req.append(input_req)
            logger.info("Received text: '%s'", req)
        return req
    
    def inference(self, inputs):
        inferences = self.pipe(
            class_labels=inputs, guidance_scale=7.5, num_inference_steps=50
        ).images

        logger.info("Generated image: '%s'", inferences)
        return inferences
    
    def postprocess(self, inference_output):
        images = []
        for image in inference_output:
            images.append(np.array(image).tolist())
        return images