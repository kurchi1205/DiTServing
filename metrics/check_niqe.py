import os
from pathlib import Path
import numpy as np
from PIL import Image
from niqe import niqe  # Assuming you have niqe() defined elsewhere
import json

def load_image_for_niqe(path):
    """Load grayscale image for NIQE (L channel only)."""
    img = np.array(Image.open(path).convert('LA'))[:, :, 0]
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img

def compute_niqe(image_path):
    img = load_image_for_niqe(image_path)
    return niqe(img)

def evaluate_niqe_by_prompt(src_root, tgt_root, cache_intervals, output_path="niqe_scores_by_prompt.json"):
    src_root = Path(src_root)
    tgt_root = Path(tgt_root)
    niqe_scores = {}

    for prompt_folder in sorted(src_root.iterdir()):
        if not prompt_folder.is_dir():
            continue

        prompt_name = prompt_folder.name
        niqe_scores[prompt_name] = {}

        for interval in cache_intervals:
            tgt_folder = tgt_root / prompt_name
            tgt_images = sorted(tgt_folder.glob(f"cache_{interval}*.png"))

            if not tgt_images:
                print(f"No target images for {prompt_name} at cache_{interval}")
                continue

            try:
                scores = [compute_niqe(img_path) for img_path in tgt_images]
                avg_score = float(np.mean(scores))
                niqe_scores[prompt_name][str(interval)] = avg_score
                print(f"{prompt_name} [cache_{interval}]: NIQE = {avg_score:.2f}")
            except Exception as e:
                print(f"Error processing {prompt_name} [cache_{interval}]: {e}")
                niqe_scores[prompt_name][str(interval)] = None

    # Summary
    with open(output_path, "w") as f:
        json.dump(niqe_scores, f, indent=2)

    return niqe_scores

# --- Entry Point ---
if __name__ == "__main__":
    src_root = "/home/fast-dit-serving/assets"
    tgt_root = "/home/fast-dit-serving/assets/our_outputs"
    output_path = "/home/fast-dit-serving/outputs/niq_scores_by_prompt.json"
    cache_intervals = [1, 2, 3, 4, 5, 6]

    evaluate_niqe_by_prompt(src_root, tgt_root, cache_intervals, output_path)
