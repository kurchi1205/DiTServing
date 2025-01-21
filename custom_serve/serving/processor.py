import time
from pipeline.sd3 import SD3LatentFormat

def process_each_timestep(handler, request_id, request_pool, compute_attention=True):
    denoiser = handler.denoiser
    model = handler.sd3.model
    request = request_pool.requests[request_id]
    prompt = request["prompt"]
    timesteps_left = request["timesteps_left"]
    cfg_scale = request["cfg_scale"]
    empty_latent = request_pool.empty_latent
    neg_cond = request_pool.neg_cond
    old_denoised = None
    width = 1024
    height = 1024
    if request["current_timestep"] == 0:
        noise_scaled, sigmas, conditioning, neg_cond, seed_num = handler.prepare_for_first_timestep(empty_latent, prompt, neg_cond, timesteps_left, seed_type="rand")
        request["noise_scaled"] = noise_scaled
        request["sigmas"] = sigmas
        request["conditioning"] = conditioning
        request["neg_cond"] = neg_cond
        request["seed_num"] = seed_num
        request["old_denoised"] = old_denoised

    else:
        noise_scaled = request["noise_scaled"]
        sigmas = request["sigmas"]
        conditioning = request["conditioning"]
        neg_cond = request["neg_cond"]
        seed_num = request["seed_num"]
        old_denoised = request["old_denoised"]

    current_timestep = request["current_timestep"]
    extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": None,
            "request": request,
            "compute_attention": compute_attention
        }
    denoiser_model = denoiser(model)
    latent, old_denoised = handler.denoise_each_step(denoiser_model, noise_scaled, sigmas[current_timestep], sigmas[current_timestep - 1], sigmas[current_timestep + 1], old_denoised, extra_args)
    request["noise_scaled"] = latent
    request["old_denoised"] = old_denoised
    request_pool.requests[request_id] = request


