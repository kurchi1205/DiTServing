import logging
import zipfile
from abc import ABC

import diffusers
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class SD3Handler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False

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
        with zipfile.ZipFile(model_dir + "/sd3_model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/sd3_model")

        # self.pipe = DiffusionPipeline.from_pretrained(model_dir + "/model")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_dir + "/sd3_model", torch_dtype=torch.float16)
        self.pipe.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        inputs = []
        for _, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info("Received text: '%s'", input_text)
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        logger.info("Inputs: %s", inputs)
        inferences = self.pipe(
            prompt=inputs,
            negative_prompt="",
            num_inference_steps=50,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        ).images

        logger.info("Generated image: '%s'", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """
        images = []
        for image in inference_output:
            images.append(np.array(image).tolist())
        return images
