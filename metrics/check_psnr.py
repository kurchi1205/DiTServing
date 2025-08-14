import os
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio

def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val=1.0):
    mse = F.mse_loss(x, y)
    return float('inf') if mse == 0 else 20 * torch.log10(max_val / torch.sqrt(mse))


def compute_psnr_torchmetrics(psnr_metric, x: torch.Tensor, y: torch.Tensor):
    # Ensure tensors are 4D (B, C, H, W)
    if x.ndim == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    return psnr_metric(x, y)


def get_request_ids_from_dir(directory):
    request_ids = set()
    pattern = re.compile(r"latent_request(.*?)_step\d+\.pt")
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            request_ids.add(match.group(1))
    return sorted(request_ids)

def load_latents_by_request_id(latent_dir, request_id):
    pattern = re.compile(rf"latent_request{re.escape(request_id)}_step(\d+)\.pt")
    files = sorted(
        [f for f in os.listdir(latent_dir) if pattern.match(f)],
        key=lambda f: int(pattern.match(f).group(1))
    )
    return [torch.load(os.path.join(latent_dir, f)) for f in files]

def get_psnr_list(src_dir, target_dir, src_id, target_id, psnr_metric):
    latents_src = load_latents_by_request_id(src_dir, src_id)
    latents_target = load_latents_by_request_id(target_dir, target_id)
    num_steps = min(len(latents_src), len(latents_target))
    psnrs = []
    for i in range(num_steps):
        a, b = latents_src[i], latents_target[i]
        psnrs.append(compute_psnr(a, b))
    return list(range(num_steps)), psnrs


if __name__=="__main__":
    # ---- Define paths ----
    src_path = "/home/fast-dit-serving/saved_latents_0"
    target_paths = [
        # "/home/fast-dit-serving/saved_latents_0",
        "/home/fast-dit-serving/saved_latents_1",
        # "/home/fast-dit-serving/saved_latents_2",
        # "/home/fast-dit-serving/saved_latents_4",
        # "/home/fast-dit-serving/saved_latents_3",
    ]
    psnr_metric = PeakSignalNoiseRatio(data_range=12.0)
    # ---- Select source request ID ----
    src_ids = get_request_ids_from_dir(src_path)
    if not src_ids:
        raise ValueError(f"No request IDs found in {src_path}")
    src_id = src_ids[0]
    print(f"Using source ID: {src_id}")

    # ---- PSNR comparisons ----
    plt.figure(figsize=(10, 6))

    for target_path in target_paths:
        target_ids = get_request_ids_from_dir(target_path)
        if not target_ids:
            print(f"⚠️ No request IDs in {target_path}, skipping.")
            continue
        if len(target_ids) > 1:
            target_id = target_ids[1]
        else:
            target_id = target_ids[0]
        try:
            steps, psnrs = get_psnr_list(src_path, target_path, src_id, target_id, psnr_metric)
            label = os.path.basename(target_path)
            plt.plot(steps, psnrs, marker='o', label=f"{label} vs src")
        except Exception as e:
            print(f"Error comparing with {target_path}: {e}")

    # ---- Plotting ----
    plt.title(f"PSNR per Denoising Step vs Source ID '{src_id}'")
    plt.xlabel("Timestep")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.locator_params(axis='y', nbins=40)
    plt.savefig("psnr_comparisons_shift_4.jpeg")
