import torch
import os
import argparse

try:
    from pipelines.pipeline_dit_with_attn_scores import DitPipelineAttnScores
except:
    import sys
    sys.path.insert(0, "../")
    from pipelines.pipeline_dit_with_attn_scores import DitPipelineAttnScores

from diffusers import DPMSolverMultistepScheduler

pipe_config = {
    "pipeline_type": "base"
}

def load_pipe(pipe_config_arg=None):
    global pipe_config
    if pipe_config_arg is not None:
        pipe_config = pipe_config_arg
    pipe_type = pipe_config["pipeline_type"]
    if pipe_type == "base":
        pipe = DitPipelineAttnScores.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float32)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe


def generate(pipe, prompt, num_inference_steps, attention_scores_layer):
    class_ids = pipe.get_label_ids([prompt])
    output = pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, attention_scores_layer=attention_scores_layer)
    image = output.images[0]  
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with attention scores from a specific layer.')
    parser.add_argument('--attention_scores_layer', type=int, default=2, help='Specifies the transformer layer from which to pull attention scores.')
    parser.add_argument('--prompt', type=str, help='Prompt for generating the image.', default="white shark")
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipe = load_pipe()
    image = generate(pipe, args.prompt, num_inference_steps=args.num_inference_steps, attention_scores_layer=args.attention_scores_layer)
    os.makedirs("../results", exist_ok=True)
    image.save("../results/base_infer_image.jpg")