mkdir -p ../results/attention
cd ../src_infer
python infer_pipeline_with_att_scores.py --attention_scores_layer 10
cd ../experiments
python plot_attention_scores.py --attention_scores_layer 10