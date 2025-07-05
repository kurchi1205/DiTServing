import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import lpips
from torchvision import transforms


def compute_lpips(img1_path, img2_path, loss_fn, transform):
    img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to("cuda")
    img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to("cuda")

    with torch.no_grad():
        dist = loss_fn(img1, img2)
    return dist.item()


def compute_lpips_for_folders(source_root, target_root, cache_intervals):
    lpips_scores = {}

    # Load LPIPS model once
    loss_fn = lpips.LPIPS(net='alex').to("cuda")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for interval in cache_intervals:
        print(f"\n=== Cache Interval: {interval} ===")
        lpips_scores[interval] = {}

        for prompt_folder in Path(source_root).iterdir():
            if not prompt_folder.is_dir():
                continue

            prompt_name = prompt_folder.name
            src_images = sorted(prompt_folder.glob("generated_image*.png"))
            tgt_folder = Path(target_root) / prompt_name
            tgt_images = sorted(tgt_folder.glob(f"cache_{interval}*.png"))

            if not src_images or not tgt_images:
                print(f"❌ Missing images for {prompt_name} at interval {interval}")
                continue

            scores = []
            for src_img, tgt_img in zip(src_images, tgt_images):
                try:
                    score = compute_lpips(src_img, tgt_img, loss_fn, transform)
                    scores.append(score)
                except Exception as e:
                    print(f"⚠️ Error comparing {src_img.name} and {tgt_img.name}: {e}")

            if scores:
                avg_score = float(np.mean(scores))
                lpips_scores[interval][prompt_name] = avg_score
                print(f"✅ {prompt_name} [cache_{interval}] → LPIPS: {avg_score:.4f}")

    # Print summary
    print("\n=== Final LPIPS Summary ===")
    for interval, results in lpips_scores.items():
        for prompt, score in results.items():
            print(f"[cache_{interval}] {prompt}: LPIPS = {score:.4f}")


if __name__ == "__main__":
    source_dir = "/home/DiTServing/assets"
    target_dir = "/home/DiTServing/assets/our_outputs"
    cache_steps = [0, 1, 2, 3, 4, 5, 6]  # include baseline

    compute_lpips_for_folders(source_dir, target_dir, cache_steps)
