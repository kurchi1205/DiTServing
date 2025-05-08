import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from safetensors import safe_open
import sys
sys.path.insert(0, "../")
from custom_serve.pipeline.vae import SDVAE
from custom_serve.pipeline.sd3 import ModelSamplingDiscreteFlow


@torch.no_grad()
def predict_denoised_image(image_path, vae, model_sampling, device="cuda", image_size=1024, dtype=torch.float16):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),               # [0, 1]
        transforms.Normalize([0.5], [0.5])    # [-1, 1]
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)

    # Add noise
    # t = torch.randint(30, 50, (1,))
    # sigma = model_sampling.sigma(t).view(1, 1, 1, 1).to(device)
    # noise = torch.randn_like(img_tensor)
    # noisy_tensor = sigma * noise + (1.0 - sigma) * img_tensor

    # Denoise with VAE
    vae = vae.to(device).eval()
    z, _, _ = vae.encode(img_tensor)
    recon = vae.decode(z)

    # Convert output to PIL
    recon = recon.squeeze(0).clamp(-1, 1)
    recon = (recon + 1) / 2  # [-1,1] → [0,1]
    recon_np = (recon.cpu().numpy() * 255).astype(np.uint8)
    recon_np = np.moveaxis(recon_np, 0, 2)  # CHW → HWC

    return Image.fromarray(recon_np)


def load_vae_model(model_path, dtype=torch.float16, device="cpu"):
    """
    Load a VAE model (SDVAE) from .safetensors or .pt/.pth file.
    """
    model = SDVAE(dtype=dtype, device=device).eval().to(device)

    if model_path.endswith(".safetensors"):
        with safe_open(model_path, framework="pt", device=device) as f:
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            for key in f.keys():
                param_name = key[len(prefix):] if key.startswith(prefix) else key
                if param_name in dict(model.named_parameters()):
                    param = dict(model.named_parameters())[param_name]
                    param.data.copy_(f.get_tensor(key).to(dtype))
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        state_dict = torch.load(model_path, map_location=device)
        if any(k.startswith("first_stage_model.") for k in state_dict.keys()):
            state_dict = {k.replace("first_stage_model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    return model


def main():
    image_path = "/home/DiTServing/custom_serve/output_images/request_164409f4-2980-4b58-a57b-61f297fad19c.png"                      # Change to your image
    model_path = "/home/DiTServing/vae_finetune/outputs/models_v4_fp16/250-finetuned.pth"        # Change to your model
    save_path = "output_denoised.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading model...")
    vae = load_vae_model(model_path, dtype=dtype, device=device)
    model_sampling = ModelSamplingDiscreteFlow()

    print("Running prediction...")
    output_image = predict_denoised_image(image_path, vae, model_sampling, device=device, dtype=dtype)

    output_image.save(save_path)
    print(f"✅ Denoised image saved to {save_path}")


if __name__ == "__main__":
    main()
