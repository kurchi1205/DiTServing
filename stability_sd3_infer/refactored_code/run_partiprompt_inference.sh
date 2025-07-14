PROMPT_JSON="/home/DiTServing/partiprompts_generation/parti_prompts.json"
CHALLENGE_JSON="/home/DiTServing/partiprompts_generation/parti_challenges.json"
MODEL_PATH="../models/sd3_medium.safetensors"
MODEL_FOLDER="../models/"
OUTPUT_DIR="/home/DiTServing/assets/partiprompts_sd3_og"
WIDTH=1024
HEIGHT=1024
STEPS=50
SEED=50


echo "🚀 Starting batch inference..."

python inference.py \
    --prompt_json_path "$PROMPT_JSON" \
    --challenge_json_path "$CHALLENGE_JSON"\
    --model_path "$MODEL_PATH" \
    --model_folder "$MODEL_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --steps "$STEPS" \
    --seed "$SEED" \

echo "All prompts processed. Images saved in $OUTPUT_DIR."