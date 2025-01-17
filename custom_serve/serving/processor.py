from pipeline.sd3 import SD3LatentFormat

def process_each_timestep(handler, request_id, request_pool):
    denoiser = handler.denoiser
    model = handler.sd3.model
    prompt = request_pool.requests[request_id]["prompt"]
    timesteps_left = request_pool.requests[request_id]["timesteps_left"]
    cfg_scale = request_pool.requests[request_id]["cfg_scale"]
    width = 1024
    height = 1024
    if request_pool.requests[request_id]["current_timestep"] == 0:
        noise_scaled, sigmas, conditioning, neg_cond, seed_num = handler.prepare_for_first_timestep(width, height, prompt, timesteps_left, seed_type="rand")
        request_pool.requests[request_id]["noise_scaled"] = noise_scaled
        request_pool.requests[request_id]["sigmas"] = sigmas
        request_pool.requests[request_id]["conditioning"] = conditioning
        request_pool.requests[request_id]["neg_cond"] = neg_cond
        request_pool.requests[request_id]["seed_num"] = seed_num

    else:
        noise_scaled = request_pool.requests[request_id]["noise_scaled"]
        sigmas = request_pool.requests[request_id]["sigmas"]
        conditioning = request_pool.requests[request_id]["conditioning"]
        neg_cond = request_pool.requests[request_id]["neg_cond"]
        seed_num = request_pool.requests[request_id]["seed_num"]

    current_timestep = request_pool.requests[request_id]["current_timestep"]
    extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": None,
        }
    denoiser_model = denoiser(model)
    latent = handler.denoise_each_step(denoiser_model, noise_scaled, sigmas[current_timestep], sigmas[current_timestep - 1], sigmas[current_timestep + 1], None, extra_args)
    request_pool.requests[request_id]["noise_scaled"] = latent


