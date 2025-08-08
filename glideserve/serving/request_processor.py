import torch
import os

PROFILE_GPU = os.getenv("PROFILE_GPU", "false").lower() == "true"

def process_each_timestep(handler, request_id, request, neg_cond, cache_interval, compute_attention=True, save_latents=False):
    denoiser = handler.denoiser
    model = handler.sd3.model
    cfg_scale = request["cfg_scale"]
    old_denoised = None
    noise_scaled = request["noise_scaled"]
    sigmas = request["sigmas"]
    conditioning = request["conditioning"]
    neg_cond = request["neg_cond"]
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

    sigma_current = torch.stack([sigmas[current_timestep]], dim=0)
    sigma_prev = torch.stack([sigmas[current_timestep - 1]], dim=0)
    sigma_next = torch.stack([sigmas[current_timestep + 1]], dim=0)

    if PROFILE_GPU:
        latent, old_denoised, elapsed_gpu_time = handler.denoise_each_step(denoiser_model, noise_scaled, sigma_current, sigma_prev, sigma_next, old_denoised, extra_args)
        request["elapsed_gpu_time"] += elapsed_gpu_time
    else:
        latent, old_denoised = handler.denoise_each_step(denoiser_model, noise_scaled, sigma_current, sigma_prev, sigma_next, old_denoised, extra_args)
    
    request["noise_scaled"] = latent
    request["old_denoised"] = old_denoised

    if save_latents:
        save_dir = f"../saved_latents_{cache_interval}"
        os.makedirs(save_dir, exist_ok=True)
        latent_path = os.path.join(save_dir, f"latent_request_{request_id}_step{current_timestep:04d}.pt")
        torch.save(request["noise_scaled"].detach().cpu(), latent_path)
    return request



def process_each_timestep_batched(handler, request_ids, requests, neg_cond, cache_interval, compute_attention=False, save_latents=False):
    # st = time.time()
    denoiser = handler.denoiser
    model = handler.sd3.model
    num_requests = len(request_ids)
    
    first_request = requests[0]
    noise_shape = first_request["noise_scaled"].shape
    cond_cross_shape = first_request["conditioning"]["c_crossattn"].shape
    cond_y_shape = first_request["conditioning"]["y"].shape
    
    # Pre-allocate tensors
    noise_scaled_batch = torch.empty((num_requests, *noise_shape[1:]), device=first_request["noise_scaled"].device, dtype=first_request["noise_scaled"].dtype)
    old_denoised_batch = torch.empty_like(noise_scaled_batch)
    sigma_current_batch = torch.empty(num_requests, device=first_request["sigmas"].device, dtype=first_request["sigmas"].dtype)
    sigma_prev_batch = torch.empty_like(sigma_current_batch)
    sigma_next_batch = torch.empty_like(sigma_current_batch)
    
    conditioning_cross = torch.empty((num_requests, *cond_cross_shape[1:]), device=first_request["conditioning"]["c_crossattn"].device, dtype=first_request["conditioning"]["c_crossattn"].dtype)
    conditioning_y = torch.empty((num_requests, *cond_y_shape[1:]), device=first_request["conditioning"]["y"].device, dtype=first_request["conditioning"]["y"].dtype)
    neg_cond_cross = torch.empty_like(conditioning_cross)
    neg_cond_y = torch.empty_like(conditioning_y)
    
    x_latent_batch = {}
    
    for idx, request in enumerate(requests):
        noise_scaled_batch[idx] = request["noise_scaled"]
        if request["old_denoised"] is not None:
            old_denoised_batch[idx] = request["old_denoised"]
        
        current_timestep = request["current_timestep"]
        sigma_current_batch[idx] = request["sigmas"][current_timestep]
        sigma_prev_batch[idx] = request["sigmas"][current_timestep - 1]
        sigma_next_batch[idx] = request["sigmas"][current_timestep + 1]
        
        conditioning_cross[idx] = request["conditioning"]["c_crossattn"]
        conditioning_y[idx] = request["conditioning"]["y"]
        neg_cond_cross[idx] = neg_cond["c_crossattn"]
        neg_cond_y[idx] = neg_cond["y"]
        
        
        for key, value in request["x_latent"].items():
            if key not in x_latent_batch:
                x_latent_batch[key] = []
            x_latent_batch[key].append(value.clone())

    
    for key in x_latent_batch:
        x_latent_batch[key] = torch.stack([x.clone() for x in x_latent_batch[key]], dim=1).reshape(-1, *x_latent_batch[key][0].shape[1:])

    conditioning_batch = {
        "c_crossattn": conditioning_cross,
        "y": conditioning_y
    }
    neg_cond_batch = {
        "c_crossattn": neg_cond_cross,
        "y": neg_cond_y
    }

    # Prepare denoising arguments
    extra_args = {
        "cond": conditioning_batch,
        "uncond": neg_cond_batch,
        "cond_scale": 5,
        "controlnet_cond": None,
        "compute_attention": compute_attention,
        "context_latent": None,
        "x_latent": x_latent_batch
    }

    # Run denoising
    denoiser_model = denoiser(model)
    elapsed_gpu_time = 0
    if PROFILE_GPU:
        latent_batch, new_old_denoised_batch, elapsed_gpu_time = handler.denoise_each_step(
            denoiser_model,
            noise_scaled_batch,
            sigma_current_batch,
            sigma_prev_batch,
            sigma_next_batch,
            old_denoised_batch,
            extra_args
        )
    else:
        latent_batch, new_old_denoised_batch = handler.denoise_each_step(
            denoiser_model,
            noise_scaled_batch,
            sigma_current_batch,
            sigma_prev_batch,
            sigma_next_batch,
            old_denoised_batch,
            extra_args
        )

    # Update requests directly
    for idx, request in enumerate(requests):
        request["noise_scaled"] = latent_batch[idx:idx+1]
        request["old_denoised"] = new_old_denoised_batch[idx:idx+1]
        request["elapsed_gpu_time"] += elapsed_gpu_time
        request_id = request["request_id"]

        if save_latents:
            save_dir = f"../saved_latents_{cache_interval}"
            os.makedirs(save_dir, exist_ok=True)
            latent_path = os.path.join(save_dir, f"latent_request_{request_id}_step{current_timestep:04d}.pt")
            torch.save(request["noise_scaled"].detach().cpu(), latent_path)
   
    return requests
