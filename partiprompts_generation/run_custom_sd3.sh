#!/bin/bash

# Define the arguments
INTERVAL_LIST=(5)
PROMPT_PATH="parti_prompts.json"
CHALLENGE_PATH="parti_challenges.json"

# Install requirements and run server
cd ../custom_serve
pip install -r requirements.txt
python server.py &

# Run background process and client
cd partiprompts_generation
python start_custom_sd3_bck_proc.py &
python client_for_custom_sd3.py --interval_list "${INTERVAL_LIST[@]}" --prompt_path "$PROMPT_PATH" --challenge_path "$CHALLENGE_PATH"