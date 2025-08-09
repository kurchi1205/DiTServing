SRC_DIR="/home/DiTServing/assets/partiprompts_sd3_og"
TGT_DIR="/home/DiTServing/assets/partiprompts_sd3_custom"
OUTPUT_JSON="/home/DiTServing/partiprompts_generation/outputs/fid_scores_by_prompt.json"
INTERVALS="5"

# Run the Python script
python compute_fid_scores.py \
  --src_dir "$SRC_DIR" \
  --tgt_dir "$TGT_DIR" \
  --output "$OUTPUT_JSON" \
  --intervals $INTERVALS

python fid_stats_summary.py --json_path "/home/DiTServing/partiprompts_generation/outputs/fid_scores_by_prompt.json"