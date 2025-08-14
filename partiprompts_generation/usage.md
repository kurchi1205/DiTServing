## Pipeline Overview

| Step | Task                                    | Command(s)                                                                 | Notes |
|------|-----------------------------------------|-----------------------------------------------------------------------------|-------|
| 1    | **Prompt Curation**                     | `pip install -r requirements.txt`<br>`python extract_prompts_from_hf_to_json.py` | Extracts 20 prompts per challenge level from Google PartiPrompts (P2) dataset |
| 2    | **Image Generation (sd3_serve)**       | `cd generation`<br>`./run_custom_sd3.sh`                                    | Set `cache_interval` before running |
| 3    | **Image Generation (Diffusers)**        | `./run_diffuser_sd3.sh`                                                     | Alternative to sd3_serve generation |
| 4    | **Evaluation – CLIP Score**             | `cd evaluation`<br>`./evaluate_clip.sh`                                     | If `cache_interval` is None, evaluates all images; otherwise only those with matching interval |
| 5    | **Evaluation – FID Score**              | `./evaluate_fid.sh`                                                         | Measures image realism vs. reference set |
| 6    | **Evaluation – SSIM**                   | `./evaluate_ssim.sh`                                                        | Measures structural similarity between images |
