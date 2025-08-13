python find_failure_cases.py \
--clip_path ../outputs/clip_scores_by_prompt_custom.json \
--fid_path ../outputs/fid_scores_by_prompt.json \
--ssim_path ../outputs/ssim_scores_by_prompt.json \
--fid_interval_key 5 --ssim_interval_key 5 \
--w_clip 3 --w_fid 1 --w_ssim 1 \
--gamma 1 --quantile 0.10 \
--topN 10 --out_json ../failure_cases/failures_cache_5.json

