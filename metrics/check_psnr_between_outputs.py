import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def compute_psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert("RGB")).astype(np.float32)
    img2 = np.array(Image.open(img2_path).convert("RGB")).astype(np.float32)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_psnr_for_folders(source_root, target_root, cache_intervals):
    psnr_scores = {}

    for interval in cache_intervals:
        print(f"\n=== Cache Interval: {interval} ===")
        psnr_scores[interval] = {}

        for prompt_folder in Path(source_root).iterdir():
            if not prompt_folder.is_dir():
                continue

            prompt_name = prompt_folder.name
            src_matches = list(prompt_folder.glob("generated_image*.png"))
            tgt_folder = Path(target_root) / prompt_folder.name
            tgt_matches = list(tgt_folder.glob(f"cache_{interval}*.png"))

            if not src_matches:
                print(f"No source images for {prompt_name}")
                continue
            if not tgt_matches:
                print(f"No target images for {prompt_name} cache_{interval}")
                continue

            # Create temp folders
            temp_src = Path(f"temp_src_{prompt_name}_{interval}")
            temp_tgt = Path(f"temp_tgt_{prompt_name}_{interval}")
            temp_src.mkdir(exist_ok=True)
            temp_tgt.mkdir(exist_ok=True)

            # Copy images into temp folders
            shutil.copy(src_matches[0], temp_src / f"image.png")
            shutil.copy(tgt_matches[0], temp_tgt / f"image.png")

            try:
                psnr_values = []
                for src_img, tgt_img in zip(sorted(temp_src.glob("*.png")), sorted(temp_tgt.glob("*.png"))):
                    psnr_val = compute_psnr(src_img, tgt_img)
                    psnr_values.append(psnr_val)

                avg_psnr = float(np.mean(psnr_values))
                psnr_scores[interval][prompt_name] = avg_psnr
                print(f"{prompt_name} [cache_{interval}] → PSNR: {avg_psnr:.2f} dB")

            except Exception as e:
                print(f"Error computing PSNR for {prompt_name} cache_{interval}: {e}")
                psnr_scores[interval][prompt_name] = None

            # Clean up temp folders
            shutil.rmtree(temp_src)
            shutil.rmtree(temp_tgt)

    # Print summary
    print("\n=== Final PSNR Summary ===")
    for interval, results in psnr_scores.items():
        for prompt, score in results.items():
            if score is None:
                print(f"[cache_{interval}] {prompt}: ERROR")
            else:
                print(f"[cache_{interval}] {prompt}: PSNR = {score:.2f} dB")


if __name__ == "__main__":
    source_dir = "/home/DiTServing/assets"
    target_dir = "/home/DiTServing/assets/our_outputs"
    cache_steps = [1, 2, 3, 4, 5, 6]  # Update as needed

    compute_psnr_for_folders(source_dir, target_dir, cache_steps)
