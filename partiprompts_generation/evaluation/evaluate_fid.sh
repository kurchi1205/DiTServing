SRC_DIR="/home/fast-dit-serving/assets/partiprompts_sd3_og"
TGT_DIR="/home/fast-dit-serving/assets/partiprompts_sd3_custom"
OUTPUT_JSON="/home/fast-dit-serving/partiprompts_generation/outputs/fid_scores_by_prompt.json"
INTERVALS="1 2 3 4 5 6"

# Run the Python script
python compute_fid_scores.py \
  --src_dir "$SRC_DIR" \
  --tgt_dir "$TGT_DIR" \
  --output "$OUTPUT_JSON" \
  --intervals $INTERVALS

python stats_summary_fid.py --json_path "/home/fast-dit-serving/partiprompts_generation/outputs/fid_scores_by_prompt.json"