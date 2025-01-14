import torch
from safetensors import safe_open
from utils import load_into
from mmditx import MMDiTX

class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(
        self,
        shift=1.0,
        device=None,
        dtype=torch.float32,
        file=None,
        prefix="",
        control_model_ckpt=None,
        verbose=False,
    ):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape

        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )
        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )

        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }
        self.diffusion_model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, *args):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SkipLayerCFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, steps, skip_layer_config):
        super().__init__()
        self.model = model
        self.steps = steps
        self.slg = skip_layer_config["scale"]
        self.skip_start = skip_layer_config["start"]
        self.skip_end = skip_layer_config["end"]
        self.skip_layers = skip_layer_config["layers"]
        self.step = 0

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        # Then run with skip layer
        if (
            self.slg > 0
            and self.step > (self.skip_start * self.steps)
            and self.step < (self.skip_end * self.steps)
        ):
            skip_layer_out = self.model.apply_model(
                x,
                timestep,
                c_crossattn=cond["c_crossattn"],
                y=cond["y"],
                skip_layers=self.skip_layers,
            )
            # Then scale acc to skip layer guidance
            scaled = scaled + (pos_out - skip_layer_out) * self.slg
        self.step += 1
        return scaled

    
class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor(
            [
                [-0.0645, 0.0177, 0.1052],
                [0.0028, 0.0312, 0.0650],
                [0.1848, 0.0762, 0.0360],
                [0.0944, 0.0360, 0.0889],
                [0.0897, 0.0506, -0.0364],
                [-0.0020, 0.1203, 0.0284],
                [0.0855, 0.0118, 0.0283],
                [-0.0539, 0.0658, 0.1047],
                [-0.0057, 0.0116, 0.0700],
                [-0.0412, 0.0281, -0.0039],
                [0.1106, 0.1171, 0.1220],
                [-0.0248, 0.0682, -0.0481],
                [0.0815, 0.0846, 0.1207],
                [-0.0120, -0.0055, -0.0867],
                [-0.0749, -0.0634, -0.0456],
                [-0.1418, -0.1457, -0.1259],
            ],
            device="cpu",
        )
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (
            ((latent_image + 1) / 2)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
            .byte()
        ).cpu()

        return Image.fromarray(latents_ubyte.numpy())


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu"
    ):
        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cuda", torch.float16)
        


