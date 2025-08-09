IMG_DIR="/home/DiTServing/assets/partiprompts_sd3_custom"
PROMPTS_JSON="/home/DiTServing/partiprompts_generation/parti_prompts.json"
OUTPUT_JSON="/home/DiTServing/partiprompts_generation/outputs/clip_scores_by_prompt_custom.json"



python compute_clip_scores.py --image_dir "$IMG_DIR" --prompts "$PROMPTS_JSON" --output "$OUTPUT_JSON"
python clip_stats_summary.py --json_path "$OUTPUT_JSON"