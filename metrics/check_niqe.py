import os
from pathlib import Path
import numpy as np
from PIL import Image
from niqe import niqe  # Assuming you have niqe() defined elsewhere

def load_image_for_niqe(path):
    """Load grayscale image for NIQE (L channel only)."""
    img = np.array(Image.open(path).convert('LA'))[:, :, 0]
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img

def compute_niqe(image_path):
    img = load_image_for_niqe(image_path)
    return niqe(img)

def evaluate_niqe_by_cache(src_root, tgt_root, cache_intervals):
    src_root = Path(src_root)
    tgt_root = Path(tgt_root)
    niqe_scores = {}

    for interval in cache_intervals:
        print(f"\n=== Cache Interval: {interval} ===")
        niqe_scores[interval] = {}

        for prompt_folder in sorted(src_root.iterdir()):
            if not prompt_folder.is_dir():
                continue

            prompt_name = prompt_folder.name
            tgt_folder = tgt_root / prompt_name
            tgt_images = sorted(tgt_folder.glob(f"cache_{interval}*.png"))

            if not tgt_images:
                print(f"❌ No target images found for {prompt_name} (cache_{interval})")
                continue

            try:
                scores = []
                for img_path in tgt_images:
                    score = compute_niqe(img_path)
                    scores.append(score)

                avg_score = float(np.mean(scores))
                niqe_scores[interval][prompt_name] = avg_score
                print(f"✅ {prompt_name} (cache_{interval}): NIQE = {avg_score:.2f}")

            except Exception as e:
                print(f"⚠️ Error computing NIQE for {prompt_name} (cache_{interval}): {e}")
                niqe_scores[interval][prompt_name] = None

    # Summary
    print("\n=== NIQE Summary ===")
    for interval, results in niqe_scores.items():
        for prompt, score in results.items():
            if score is not None:
                print(f"[cache_{interval}] {prompt}: NIQE = {score:.2f}")
            else:
                print(f"[cache_{interval}] {prompt}: ERROR")

    return niqe_scores

# --- Entry Point ---
if __name__ == "__main__":
    src_root = "/home/DiTServing/assets"
    tgt_root = "/home/DiTServing/assets/our_outputs"
    cache_intervals = [5, 6]

    evaluate_niqe_by_cache(src_root, tgt_root, cache_intervals)
