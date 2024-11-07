import torch
from diffusers import StableDiffusion3Pipeline


def load_pipe():
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)

    pipe = pipe.to("cuda")
    return pipe


def infer(pipe, prompt, num_inference_steps):
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        height=512,
        width=512,
        guidance_scale=7.0,
    ).images[0]
    return image


if __name__=="__main__":
    pipe = load_pipe()
    prompt = "a photo of a cat holding a sign that says hello world"
    num_inference_steps = 28
    infer(pipe, prompt, num_inference_steps)