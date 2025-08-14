IMG_DIR="/home/fast-dit-serving/assets/partiprompts_sd3_custom"
PROMPTS_JSON="/home/fast-dit-serving/partiprompts_generation/parti_prompts.json"
OUTPUT_JSON="/home/fast-dit-serving/partiprompts_generation/outputs/clip_scores_by_prompt_custom_1.json"
CACHE_INTERVAL=1

python compute_clip_scores.py --image_dir "$IMG_DIR" --prompts "$PROMPTS_JSON" --output "$OUTPUT_JSON" --cache_interval $CACHE_INTERVAL
python stats_summary_clip.py --json_path "$OUTPUT_JSON"