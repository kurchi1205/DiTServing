import torch
from pipeline.sd3 import SD3LatentFormat

def process_each_timestep(handler, request_id, request_pool, compute_attention=True):
    print("Processing for attention")
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
        # request["noise_scaled"] = torch.cat([noise_scaled, noise_scaled], dim=0)
        # noise_scaled = request["noise_scaled"]
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
    # noise_batch = torch.cat([noise_scaled, noise_scaled], dim=0)
    sigma_current = torch.stack([sigmas[current_timestep]], dim=0)
    sigma_prev = torch.stack([sigmas[current_timestep - 1]], dim=0)
    sigma_next = torch.stack([sigmas[current_timestep + 1]], dim=0)

    # old_denoised = torch.stack([old_denoised, old_denoised], dim=0)
    # print("Noise shape: ", noise_batch.size())
    # latent, old_denoised = handler.denoise_each_step(denoiser_model, noise_scaled, sigmas[current_timestep], sigmas[current_timestep - 1], sigmas[current_timestep + 1], old_denoised, extra_args)

    latent, old_denoised = handler.denoise_each_step(denoiser_model, noise_scaled, sigma_current, sigma_prev, sigma_next, old_denoised, extra_args)
    request["noise_scaled"] = latent
    request["old_denoised"] = old_denoised
    request_pool.requests[request_id] = request




def process_each_timestep_batched(handler, request_ids, request_pool, compute_attention=False):
    print("Processing without attention")
    denoiser = handler.denoiser
    model = handler.sd3.model
    requests = []
    # Prepare batched inputs
    noise_scaled_batch = []
    sigma_current_batch = []
    sigma_prev_batch = []
    sigma_next_batch = []
    old_denoised_batch = []
    conditioning_batch = {
        "c_crossattn": [],
        "y": []
    }
    neg_cond_batch = {
        "c_crossattn": [],
        "y": []
    }
    context_latent_batch = {}
    x_latent_batch = {}


    for request_id in request_ids:
        request = request_pool.requests[request_id]
        noise_scaled_batch.append(request.pop("noise_scaled"))
        current_timestep = request["current_timestep"]
        sigma_current_batch.append(request["sigmas"][current_timestep])
        sigma_prev_batch.append(request["sigmas"][current_timestep - 1])
        sigma_next_batch.append(request["sigmas"][current_timestep + 1])
        old_denoised_batch.append(request.pop("old_denoised"))
        conditioning_batch["c_crossattn"].append(request["conditioning"]["c_crossattn"])
        conditioning_batch["y"].append(request["conditioning"]["y"])
        neg_cond_batch["c_crossattn"].append(request_pool.neg_cond["c_crossattn"])
        neg_cond_batch["y"].append(request_pool.neg_cond["y"])
        context_latent = request.pop("context_latent")
        x_latent = request.pop("x_latent")
        for key, value in context_latent.items():
            if key not in context_latent_batch:
                context_latent_batch[key] = []
            context_latent_batch[key].append(value)
        for key, value in x_latent.items():
            if key not in x_latent_batch:
                x_latent_batch[key] = []
            x_latent_batch[key].append(value)
        # del attention


    # Convert lists to tensors
    noise_scaled_batch = torch.cat(noise_scaled_batch)
    sigma_current_batch = torch.stack(sigma_current_batch)
    sigma_prev_batch = torch.stack(sigma_prev_batch)
    sigma_next_batch = torch.stack(sigma_next_batch)
    old_denoised_batch = torch.cat(old_denoised_batch)
    conditioning_batch["c_crossattn"] = torch.cat(conditioning_batch["c_crossattn"])
    conditioning_batch["y"] = torch.cat(conditioning_batch["y"])
    neg_cond_batch["c_crossattn"] = torch.cat(neg_cond_batch["c_crossattn"])
    neg_cond_batch["y"] = torch.cat(neg_cond_batch["y"])
    # neg_cond_batch = torch.stack(neg_cond_batch)
    for key in context_latent_batch:
        if context_latent_batch[key][0] is not None:
            context_latent_batch[key] = torch.cat(context_latent_batch[key])
        else:
            context_latent_batch[key] = None
    for key in x_latent_batch:
        x_latent_batch[key] = torch.cat(x_latent_batch[key])
        # print("Attention size: ", attention_batch[key].size())

    extra_args = {
        "cond": conditioning_batch,
        "uncond": neg_cond_batch,
        "cond_scale": 7.5,  # Assuming cfg_scale is consistent across all requests
        "controlnet_cond": None,
        "compute_attention": compute_attention,
        "context_latent": context_latent_batch,
        "x_latent": x_latent_batch
    }
    denoiser_model = denoiser(model)
    latent_batch, new_old_denoised_batch = handler.denoise_each_step(
        denoiser_model,
        noise_scaled_batch,
        sigma_current_batch,
        sigma_prev_batch,
        sigma_next_batch,
        old_denoised_batch,
        extra_args
    )

    # Update each request in the pool
    for idx, request_id in enumerate(request_ids):
        request = request_pool.requests[request_id]
        if len(request_ids) > 1:
            try:
                assert torch.equal(
                    latent_batch[1 * 0: 1 * 0 + 1],  # First slice
                    latent_batch[1 * 1: 1 * 1 + 1]   # Second slice
                ), "Slices of latent_batch are not equal!"
            except:
                print("latents are not equal")

        request["noise_scaled"] = latent_batch[1 * idx: 1 * idx + 1]
        request["old_denoised"] = new_old_denoised_batch[1 * idx: 1 * idx + 1]
        request["context_latent"] = {}
        for key in context_latent_batch:
            if context_latent_batch[key] is not None:
                request["context_latent"][key] = context_latent_batch[key][2 * idx: 2 * idx + 2]
            else:
                request["context_latent"][key] = None
        request["x_latent"] = {key: x_latent_batch[key][2 * idx: 2 * idx + 2] for key in x_latent_batch}
        # request["noise_scaled"] = latent_batch
        # request["old_denoised"] = new_old_denoised_batch
        # request["attention"] = {key: attention_batch[key] for key in attention_batch}
        requests.append(request)
    return requests
