import os
import json
import shutil
from pathlib import Path
# from cleanfid import fid
import torch
from PIL import Image
from torch_fidelity import calculate_metrics
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance



source_root = Path("/home/fast-dit-serving/assets")
target_root = Path("/home/fast-dit-serving/assets/our_outputs")
cache_intervals = [0, 1, 2, 3, 4, 5, 6]  # Your desired intervals

transform = transforms.Compose([
    transforms.ToTensor(),  # Scales to [0, 1]
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Output results dictionary
fid_scores = {}

for interval in cache_intervals:
    print(f"\n=== Cache Interval: {interval} ===")
    fid_scores[interval] = {}

    for prompt_folder in source_root.iterdir():
        if not prompt_folder.is_dir():
            continue

        prompt_name = prompt_folder.name
        src_images = sorted(prompt_folder.glob("generated_image*.png"))
        tgt_images = sorted((target_root / prompt_name).glob(f"cache_{interval}*.png"))
        if not src_images:
            # print(f"No source image for {prompt_name}")
            continue
        if not tgt_images:
            # print(f"No target image for {prompt_name} cache_{interval}")
            continue
        print(src_images, tgt_images)

        # Ensure minimum 2 images for FID calculation
        if len(src_images) == 1:
            src_images = src_images * 2
        if len(tgt_images) == 1:
            tgt_images = tgt_images * 2



        try:
            fid_metric = FrechetInceptionDistance(feature=64, input_img_size=(3, 1024, 1024)).to(device)

            # Update with real (source) images
            for img_path in src_images:
                img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                img = (img * 255).byte()  # Convert back to uint8 as expected by FID
                fid_metric.update(img, real=True)

            # Update with fake (target) images
            for img_path in tgt_images:
                img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                img = (img * 255).byte()
                fid_metric.update(img, real=False)

            fid_score = fid_metric.compute().item()
            fid_scores[interval][prompt_name] = fid_score
            print(f"FID for {prompt_name} (cache_{interval}): {fid_score:.2f}")

        except Exception as e:
            print(f"Error computing FID for {prompt_name} cache_{interval}: {e}")
            fid_scores[interval][prompt_name] = None

        # Optional: remove temp folders (or keep for inspection)
        # shutil.rmtree(temp_src)
        # shutil.rmtree(temp_tgt)

final_scores = {}
for interval, results in fid_scores.items():
    for prompt, score in results.items():
        if prompt not in final_scores:
            final_scores[prompt] = {}
        final_scores[prompt][str(interval)] = score  # Use str for JSON keys

# Save to file
output_path = "/home/fast-dit-serving/outputs/fid_scores_by_prompt.json"
with open(output_path, "w") as f:
    json.dump(final_scores, f, indent=2)

print(f"\nSaved FID results to {output_path}")