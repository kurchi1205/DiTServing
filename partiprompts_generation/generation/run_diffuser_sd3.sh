python infer_diffuser_sd3.py \
    --prompt_json_path ../parti_prompts.json \
    --challenge_json_path ../parti_challenges.json \
    --output_dir /home/fast-dit-serving/assets/partiprompts_sd3_diffuser \
    --num_inference_steps 50 \
    --compile False \
    --seed 50
