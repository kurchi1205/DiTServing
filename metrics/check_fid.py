import os
import json
import shutil
from pathlib import Path
# from cleanfid import fid
from torch_fidelity import calculate_metrics


source_root = Path("/home/DiTServing/assets")
target_root = Path("/home/DiTServing/assets/our_outputs")
cache_intervals = [1, 2, 3, 4, 5, 6]  # Your desired intervals


# Output results dictionary
fid_scores = {}

for interval in cache_intervals:
    print(f"\n=== Cache Interval: {interval} ===")
    fid_scores[interval] = {}

    for prompt_folder in source_root.iterdir():
        if not prompt_folder.is_dir():
            continue

        prompt_name = prompt_folder.name
        src_matches = list(prompt_folder.glob("generated_image*.png"))
        tgt_matches = list((target_root / prompt_folder.name).glob(f"cache_{interval}*.png"))
        # tgt_img_path = target_root / f"{prompt_name}_cache_{interval}.png"
        if not src_matches:
            print(f"❌ No source image for {prompt_name}")
            continue
        if not tgt_matches:
            print(f"❌ No target image for {prompt_name} cache_{interval}")
            continue
        if len(src_matches) == 1:
            src_matches = src_matches * 2
        if len(tgt_matches) == 1:
            tgt_matches = tgt_matches * 2
        # Create temp folders
        temp_src = Path(f"temp_src_{prompt_name}_{interval}")
        temp_tgt = Path(f"temp_tgt_{prompt_name}_{interval}")
        temp_src.mkdir(exist_ok=True)
        temp_tgt.mkdir(exist_ok=True)

        # Copy images into temp folders
        for i, img_path in enumerate(src_matches):
            shutil.copy(img_path, temp_src / f"image_{i}.png")
        for i, img_path in enumerate(tgt_matches):
            shutil.copy(img_path, temp_tgt / f"image_{i}.png")
        # shutil.copy(tgt_img_path, temp_tgt / "image.png")

        try:
            result = calculate_metrics(
                input1=str(temp_src),
                input2=str(temp_tgt),
                fid=True,
                verbose=False
            )
            fid_score = result['frechet_inception_distance']
            fid_scores[interval][prompt_name] = fid_score
            print(f"FID for {prompt_name} (cache_{interval}): {fid_score:.2f}")
        except Exception as e:
            print(f"Error computing FID for {prompt_name} cache_{interval}: {e}")
            fid_scores[interval][prompt_name] = None

        # Optional: remove temp folders (or keep for inspection)
        shutil.rmtree(temp_src)
        shutil.rmtree(temp_tgt)

final_scores = {}
for interval, results in fid_scores.items():
    for prompt, score in results.items():
        if prompt not in final_scores:
            final_scores[prompt] = {}
        final_scores[prompt][str(interval)] = score  # Use str for JSON keys

# Save to file
output_path = "/home/DiTServing/outputs/fid_scores_by_prompt.json"
with open(output_path, "w") as f:
    json.dump(final_scores, f, indent=2)

print(f"\nSaved FID results to {output_path}")