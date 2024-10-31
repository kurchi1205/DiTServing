import logging
import zipfile
from abc import ABC

import diffusers
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from base_pipeline_dit import DitPipeline

from ts.torch_handler.base_handler import BaseHandler
from dit_handler import DitHandler

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class DitCompiledBatchHandler(DitHandler):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized=False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        # self.pipe = DiffusionPipeline.from_pretrained(model_dir + "/model")
        self.pipe = DitPipeline.from_pretrained(model_dir + "/model")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=False, mode="reduce-overhead")
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        class_ids = self.pipe.get_label_ids(["white shark", "white shark"])
        warmup_inf = self.pipe(
            class_labels=class_ids, guidance_scale=7.5, num_inference_steps=2
        ).images
        self.initialized = True
