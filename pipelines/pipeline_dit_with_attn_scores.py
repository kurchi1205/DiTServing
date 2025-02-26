import copy
import torch
import json
from typing import Dict, List, Optional, Tuple, Union
from pipelines.base_pipeline_dit import DitPipeline, ImagePipelineOutputWithLatents
from diffusers import AutoencoderKL, DDIMScheduler, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

try:
    from models.dit import Transformer2DModelWithAttScores
except:
    import sys
    sys.path.insert(0, "../")
    from models.dit import Transformer2DModelWithAttScores

class DitPipelineAttnScores(DitPipeline):
    def __init__(
            self, 
            transformer: Transformer2DModelWithAttScores, 
            vae: AutoencoderKL, 
            scheduler: DDIMScheduler, 
            id2label: Dict[int, str] | None = None
        ):
        new_transformer = copy.deepcopy(transformer)
        new_transformer.__class__ = Transformer2DModelWithAttScores
        new_transformer.__init__(**transformer.config)
        new_transformer.load_state_dict(transformer.state_dict())
        super().__init__(new_transformer, vae, scheduler, id2label)
        
    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        get_intermediate_latents: bool = False,
        attention_scores_layer: int = 0
        
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels
        intermediate_latents = [] if get_intermediate_latents else None

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        self.attn_scores_across_timesteps = {}
        for n, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            trans_output = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input, attention_scores_layer=attention_scores_layer
            )
            noise_pred = trans_output.sample
            attn_scores = copy.deepcopy(trans_output.scores)

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample
            if get_intermediate_latents:
                if guidance_scale > 1:
                    intermediate_latents.append(latent_model_input.chunk(2, dim=0)[0])
                else:
                    intermediate_latents.append(latent_model_input)
            self.attn_scores_across_timesteps[n] = attn_scores

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        if intermediate_latents is not None:
            intermediate_latents = torch.cat(intermediate_latents, dim=0)
            intermediate_samples = self.decode_latents(intermediate_latents, output_type)
        

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)
        with open(f"../results/attention/attention_scores_{attention_scores_layer}.json", "w") as f:
            json.dump(self.attn_scores_across_timesteps, f)

        # Offload all models
        self.maybe_free_model_hooks()
        if get_intermediate_latents:
            if not return_dict:
                return (samples, intermediate_samples)

            return ImagePipelineOutputWithLatents(images=samples, latents=intermediate_samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)