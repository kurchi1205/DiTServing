import os
import shutil
from pathlib import Path
from cleanfid import fid
from torch_fidelity import calculate_metrics


source_root = Path("/home/DiTServing/assets")
target_root = Path("/home/DiTServing/assets/our_outputs")
cache_intervals = [5, 6]  # Your desired intervals


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
            print(f"✅ FID for {prompt_name} (cache_{interval}): {fid_score:.2f}")
        except Exception as e:
            print(f"⚠️ Error computing FID for {prompt_name} cache_{interval}: {e}")
            fid_scores[interval][prompt_name] = None

        # Optional: remove temp folders (or keep for inspection)
        shutil.rmtree(temp_src)
        shutil.rmtree(temp_tgt)

# ✅ Final summary
print("\n=== FID Scores Summary ===")
for interval, results in fid_scores.items():
    for prompt, score in results.items():
        print(f"[cache_{interval}] {prompt}: FID = {score:.2f}" if score is not None else f"[cache_{interval}] {prompt}: ERROR")