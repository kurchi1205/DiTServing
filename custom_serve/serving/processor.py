import time
import torch
from pipeline.sd3 import SD3LatentFormat

def process_each_timestep(handler, request_id, request_pool, compute_attention=True):
    # st = time.time()
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
    # if request["current_timestep"] == 0:
    #     noise_scaled, sigmas, conditioning, neg_cond, seed_num = handler.prepare_for_first_timestep(empty_latent, prompt, neg_cond, timesteps_left, seed_type="rand")
    #     # request["noise_scaled"] = torch.cat([noise_scaled, noise_scaled], dim=0)
    #     # noise_scaled = request["noise_scaled"]
    #     request["noise_scaled"] = noise_scaled
    #     request["sigmas"] = sigmas
    #     request["conditioning"] = conditioning
    #     request["neg_cond"] = neg_cond
    #     request["seed_num"] = seed_num
    #     request["old_denoised"] = old_denoised

    # else:
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
    # st_2 = time.time()
    latent, old_denoised = handler.denoise_each_step(denoiser_model, noise_scaled, sigma_current, sigma_prev, sigma_next, old_denoised, extra_args)
    request["noise_scaled"] = latent
    request["old_denoised"] = old_denoised
    request_pool.requests[request_id] = request
    # print("Time taken with attention", time.time() - st)
    # print("Time taken with attention computation", time.time() - st_2)





def process_each_timestep_batched_1(handler, request_ids, request_pool, compute_attention=False):
    st = time.time()
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
    # if "Generate image of a fish" in request_pool.requests:
        # print("Generate image of a fish", request_pool.requests["Generate image of a fish"]["x_latent"]["0"])
        # print("Generate image of a fish", request_pool.requests["Generate image of a fish"]["noise_scaled"])
        # image = handler.vae_decode(request_pool.requests["Generate image of a fish"]["noise_scaled"])
        # image.save(f"Generate image of a fish_{request_pool.requests["Generate image of a fish"]["current_timestep"]}.png")
    # if "Generate image of a dog" in request_pool.requests:
        # print("Generate image of a dog", request_pool.requests["Generate image of a dog"]["x_latent"]["0"])
        # print("Generate image of a dog", request_pool.requests["Generate image of a dog"]["noise_scaled"])
        # image = handler.vae_decode(request_pool.requests["Generate image of a dog"]["noise_scaled"])
        # image.save(f"Generate image of a dog_{request_pool.requests["Generate image of a dog"]["current_timestep"]}.png")
    for request_id in request_ids:
        request = request_pool.requests[request_id]
        noise_scaled_batch.append(request["noise_scaled"])
        current_timestep = request["current_timestep"]
        sigma_current_batch.append(request["sigmas"][current_timestep])
        sigma_prev_batch.append(request["sigmas"][current_timestep - 1])
        sigma_next_batch.append(request["sigmas"][current_timestep + 1])
        old_denoised_batch.append(request["old_denoised"])
        conditioning_batch["c_crossattn"].append(request["conditioning"]["c_crossattn"])
        conditioning_batch["y"].append(request["conditioning"]["y"])
        neg_cond_batch["c_crossattn"].append(request_pool.neg_cond["c_crossattn"])
        neg_cond_batch["y"].append(request_pool.neg_cond["y"])
        x_latent = request["x_latent"].copy()
        # print("stored attn: ", x_latent["0"])
        for key, value in x_latent.items():
            if key not in x_latent_batch:
                x_latent_batch[key] = []
            x_latent_batch[key].append(value.clone())
        # del attention


    # Convert lists to tensors
    noise_scaled_batch = torch.cat(noise_scaled_batch)
    # print("Size of noise scaled: ", noise_scaled_batch.size())
    # print("Size of cross attn: ",  conditioning_batch["c_crossattn"][0].size())
    # print("Size of y: ",  conditioning_batch["y"][0].size())
    # sigma_current_batch = sigma_current_batch[0]
    # sigma_prev_batch = sigma_prev_batch[0]
    # sigma_next_batch = sigma_next_batch[0]
    sigma_current_batch = torch.stack(sigma_current_batch)
    sigma_prev_batch = torch.stack(sigma_prev_batch)
    sigma_next_batch = torch.stack(sigma_next_batch)
    old_denoised_batch = torch.cat(old_denoised_batch)
    conditioning_batch["c_crossattn"] = torch.cat(conditioning_batch["c_crossattn"])
    conditioning_batch["y"] = torch.cat(conditioning_batch["y"])
    neg_cond_batch["c_crossattn"] = torch.cat(neg_cond_batch["c_crossattn"])
    neg_cond_batch["y"] = torch.cat(neg_cond_batch["y"])
    for key in x_latent_batch:
        x_latent_batch[key] = torch.stack([x.clone() for x in x_latent_batch[key]], dim=1).reshape(-1, *x_latent_batch[key][0].shape[1:])

    extra_args = {
        "cond": conditioning_batch,
        "uncond": neg_cond_batch,
        "cond_scale": 5,  # Assuming cfg_scale is consistent across all requests
        "controlnet_cond": None,
        "compute_attention": compute_attention,
        "context_latent": None,
        "x_latent": x_latent_batch
    }
    denoiser_model = denoiser(model)
    st_2 = time.time()
    latent_batch, new_old_denoised_batch = handler.denoise_each_step(
        denoiser_model,
        noise_scaled_batch,
        sigma_current_batch,
        sigma_prev_batch,
        sigma_next_batch,
        old_denoised_batch,
        extra_args
    )
    num_requests = len(request_ids)

    # Chunk latent tensors into equal parts
    latent_chunks = torch.chunk(latent_batch, num_requests, dim=0)
    old_denoised_chunks = torch.chunk(new_old_denoised_batch, num_requests, dim=0)
    # Assign each chunk to its respective request
    st_3 = time.time()
    for idx, request_id in enumerate(request_ids):
        request = request_pool.requests[request_id]
        request["noise_scaled"] = latent_chunks[idx]
        request["old_denoised"] = old_denoised_chunks[idx]
        requests.append(request)
    print(f"Time taken without attention of {num_requests}: {time.time() - st}")
    print(f"Time taken without attention computation of {num_requests}: {time.time() - st_2}")
    print(f"Time taken unbatching {num_requests}: {time.time() - st_3}")
    return requests


def process_each_timestep_batched(handler, request_ids, request_pool, compute_attention=False):
    # st = time.time()
    denoiser = handler.denoiser
    model = handler.sd3.model
    num_requests = len(request_ids)
    # print("Num requests: ", len(request_ids))
    # Get first request to determine tensor shapes
    first_request = request_pool.requests[request_ids[0]]
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
    
    # Pre-allocate x_latent tensors
    # x_latent_shapes = {k: v.shape for k, v in first_request["x_latent"].items()}
    # x_latent_batch = {
    #     k: torch.empty((num_requests, *shape[1:]), device=v.device)
    #     for k, shape in x_latent_shapes.items()
    # }
    # x_latent_batch = {
    #     k: torch.empty((v.shape[0], num_requests * v.shape[1], v.shape[2]), device=v.device, dtype=v.dtype)
    #     for k, v in first_request["x_latent"].items()
    # }
    x_latent_batch = {}
    # Fill tensors directly
    for idx, request_id in enumerate(request_ids):
        request = request_pool.requests[request_id]
        
        # Direct tensor assignments
        noise_scaled_batch[idx] = request["noise_scaled"]
        old_denoised_batch[idx] = request["old_denoised"]
        
        current_timestep = request["current_timestep"]
        sigma_current_batch[idx] = request["sigmas"][current_timestep]
        sigma_prev_batch[idx] = request["sigmas"][current_timestep - 1]
        sigma_next_batch[idx] = request["sigmas"][current_timestep + 1]
        
        conditioning_cross[idx] = request["conditioning"]["c_crossattn"]
        conditioning_y[idx] = request["conditioning"]["y"]
        neg_cond_cross[idx] = request_pool.neg_cond["c_crossattn"]
        neg_cond_y[idx] = request_pool.neg_cond["y"]
        
        # Fill x_latent tensors
        # for key, tensor in request["x_latent"].items():
        #     # x_latent_batch[key][2 * idx: 2 * idx+2] = tensor
        #     slice_size = tensor.shape[1]
        #     x_latent_batch[key][:, idx*slice_size:(idx+1)*slice_size, :] = tensor
        for key, value in request["x_latent"].items():
            if key not in x_latent_batch:
                x_latent_batch[key] = []
            x_latent_batch[key].append(value.clone())
    
    for key in x_latent_batch:
        x_latent_batch[key] = torch.stack([x.clone() for x in x_latent_batch[key]], dim=1).reshape(-1, *x_latent_batch[key][0].shape[1:])

    # for key in x_latent_batch:
    #     x_latent_batch[key] = x_latent_batch[key].reshape(-1, *x_latent_batch[key].shape[2:])
    # Prepare conditioning batches
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
    # st_2 = time.time()
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

    # Update requests directly
    requests = []
    # st_3 = time.time()
    for idx, request_id in enumerate(request_ids):
        request = request_pool.requests[request_id]
        request["noise_scaled"] = latent_batch[idx:idx+1]
        request["old_denoised"] = new_old_denoised_batch[idx:idx+1]
        requests.append(request)

    # print(f"Time taken without attention of {num_requests}: {time.time() - st}")
    # print(f"Time taken without attention computation of {num_requests}: {time.time() - st_2}")
    # print(f"Time taken unbatching {num_requests}: {time.time() - st_3}")
    return requests
