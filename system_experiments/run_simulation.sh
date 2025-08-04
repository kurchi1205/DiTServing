#!/bin/bash

PROMPT_FILE="/home/DiTServing/partiprompts_generation/parti_prompts.json"
COMPLETED_LOG="/home/DiTServing/system_experiments/completed_requests_rr_2_sec_100.json"
HOST="http://localhost:8000"
SAVE_INTERVAL=5
RATE=2              # requests per second
DURATION=100        # total duration in seconds
MODE="constant"     # "constant" or "burst"

python simulate_requests.py \
  --prompt_file "$PROMPT_FILE" \
  --completed_log "$COMPLETED_LOG" \
  --host "$HOST" \
  --save_interval $SAVE_INTERVAL \
  --rate $RATE \
  --duration $DURATION \
  --mode $MODE
