import argparse
import logging
import math
import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "5000"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1000"

import wandb

import numpy as np
import torch
import torchvision
from datasets import load_dataset
import torch.nn.functional as F
from torchvision.transforms import v2  # from vision transforms
from torchvision import transforms


import lpips
from PIL import Image
from pytorch_optimizer import CAME
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


from arg_parser import parse_basic_args
import sys
sys.path.insert(0, "../")
from custom_serve.pipeline.vae import VAE
from custom_serve.pipeline.sd3 import ModelSamplingDiscreteFlow
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def get_dtype(str_type=None):
    # return torch dtype
    torch_type = torch.float32
    if str_type == "fp16":
        torch_type = torch.float16
    elif str_type == "bf16":
        torch_type = torch.bfloat16
    return torch_type


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a VAE training script.")
    parser = parse_basic_args(parser)
    args = parser.parse_args()

    # default setting I'm using:
    args.pretrained_model_name_or_path = r"pretrained_model/sdxl_vae"
    args.vae_model = "/home/DiTServing/stability_sd3_infer/models/sd3_medium.safetensors"
    # args.revision
    # args.dataset_name = "ideepankarsharma2003/MidJourney-generated-images"
    args.dataset_name = "ehristoforu/midjourney-images"
    args.dataset_config_name = "default"
    
    args.image_column = "image"
    args.output_dir = r"outputs/models_v4_fp16"
    #args.huggingface_repo
    #cache_dir =
    args.seed = 420
    args.resolution = 1024
    args.train_batch_size = 2 # batch 2 was the best for a single rtx3090
    args.num_train_epochs = 1
    args.gradient_accumulation_steps = 1
    args.gradient_checkpointing = False
    args.learning_rate = 1e-04
    args.scale_lr = True
    args.lr_scheduler = "constant"
    args.max_data_loader_n_workers = 1
    args.lr_warmup_steps = 0
    args.logging_dir = r"outputs/vae_log"
    args.mixed_precision = 'fp16'
    
    
    #args.checkpoints_total_limit
    #args.resume_from_checkpoint
    args.test_samples = 20
    args.validation_epochs = 1
    args.validation_steps = 1
    args.tracker_project_name = "vae-fine-tune"
    args.use_8bit_adam = False
    # args.use_ema = True # this will drastically slow your training, check speed vs performance
    args.use_ema = False # changed to False
    #args.kl_scale
    args.push_to_hub = False
    #hub_token
    args.lpips_scale = 5e-8
    args.mse_scale = 0.000005
    args.kl_scale = 1e-6 # this is not relevant with patched loss
    args.lpips_start = 50001 # this doesn't do anything?
    
    #args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/sample"
    # args.train_data_dir = r"/home/wasabi/Documents/Projects/data/vae/train"
    # args.test_data_dir = r"/home/wasabi/Documents/Projects/data/vae/test"
    args.checkpointing_steps = 5000 # return to 5000
    args.model_saving_steps = 50
    args.report_to = 'wandb'

    #following are new parameters
    args.use_came = False
    args.diffusers_xformers = False
    args.save_for_SD = False # need to check tjis
    args.save_precision = "fp16"
    args.train_only_decoder = True
    # args.comment = "VAE finetune by Wasabi, test model using patched MSE"
    
    args.patch_loss = False
    args.patch_size = 64
    args.patch_stride = 32
    
    

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

def acc_unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (e.g., DDP, FSDP, etc.).

    Args:
        model (torch.nn.Module): The model to unwrap.

    Returns:
        torch.nn.Module: The unwrapped model.
    """
    # If the model is wrapped by torch's DDP (DistributedDataParallel), return the original model.
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    # If the model is wrapped by torch's FSDP (FullyShardedDataParallel), return the original model.
    elif hasattr(model, "_fsdp_wrapped_module"):
        return unwrap_model(model._fsdp_wrapped_module)
    # If the model is wrapped by torch's AMP (Automatic Mixed Precision) or other wrappers, return the original model.
    else:
        return model

def extract_patches(image, patch_size, stride):
    # Unfold the image into patches
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # Reshape to get a batch of patches
    patches = patches.contiguous().view(image.size(0), image.size(1), -1, patch_size, patch_size)
    return patches

def patch_based_mse_loss(real_images, recon_images, patch_size=32, stride=16):
    real_patches = extract_patches(real_images, patch_size, stride)
    recon_patches = extract_patches(recon_images, patch_size, stride)
    mse_loss = F.mse_loss(real_patches, recon_patches)
    return mse_loss

def patch_based_lpips_loss(lpips_model, real_images, recon_images, patch_size=32, stride=16):
    with torch.no_grad():
        real_patches = extract_patches(real_images, patch_size, stride)
        recon_patches = extract_patches(recon_images, patch_size, stride)
        
        lpips_loss = 0
        # Iterate over each patch and accumulate LPIPS loss
        for i in range(real_patches.size(2)):  # Loop over number of patches
            real_patch = real_patches[:, :, i, :, :].contiguous()
            recon_patch = recon_patches[:, :, i, :, :].contiguous()
            patch_lpips_loss = lpips_model(real_patch, recon_patch).mean()
            
            # Handle non-finite values
            if not torch.isfinite(patch_lpips_loss):
                patch_lpips_loss = torch.tensor(0, device=real_patch.device)
            
            lpips_loss += patch_lpips_loss

    return lpips_loss / real_patches.size(2)  # Normalize by the number of patches


def log_validation(test_dataloader, vae, accelerator, curr_step = 0, max_validation_sample=4):
    # vae_model = acc_unwrap_model(vae.model)
    images = []
    vae_model_unwrapped = accelerator.unwrap_model(vae.model).eval()
    from safetensors.torch import save_file
    state_dict = {
        k: v.detach().cpu().to(torch.float16) for k, v in vae_model_unwrapped.state_dict().items()
    }
    # torch.save(vae_model_unwrapped.state_dict(), "vae_temp.pth")
    save_file(state_dict, "vae_temp.safetensors")
    # === Reload the model ===
    # Create new model instance and load weights
    vae_reload = VAE("vae_temp.safetensors", dtype=torch.float16)
    # vae_reload = SDVAE(dtype=torch.float16, device="cpu").eval().to(accelerator.device)
    # vae_reload.load_state_dict(torch.load("vae_temp.pth", map_location=accelerator.device), strict=True)
    vae_reload.model = vae_reload.model.to('cuda')
    vae_reload.model = vae_reload.model.eval()
    for i, sample in enumerate(test_dataloader):
        if i < max_validation_sample:
            x = sample["src_pixel_values"]
            # encoded = vae_model.encode(x)
            # reconstructions = vae_model.decode(encoded)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                # Reconstructed using reloaded model
                z1, _, _ = vae_reload.model.encode(x)
                recon1 = vae_reload.model.decode(z1)

                # Reconstructed using in-memory model
                z2, _, _ = vae_model_unwrapped.encode(x)
                recon2 = vae_model_unwrapped.decode(z2)
            images.append(
                torch.cat([sample["src_pixel_values"].cpu(), sample["tgt_pixel_values"].cpu(), recon1.cpu(), recon2.cpu()], axis=0)
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left), Target (center), Reconstruction_saved, Reconstruction": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                },
                step=curr_step
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.gen_images}")
    del vae_model_unwrapped
    del vae_reload
    vae.model.train()
    torch.cuda.empty_cache()

def compute_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

# def add_noise_to_image(pil_image, model_sampling):
#     image_tensor = transforms.ToTensor()(pil_image)
#     t = torch.randint(30, 50, (1,))
#     sigma = model_sampling.sigma(t).view(1, 1, 1, 1).to(image_tensor.device)
#     noise = torch.randn_like(image_tensor)
#     noisy_image = sigma * noise + (1.0 - sigma) * image_tensor
#     noised = noisy_image.squeeze().clamp(0, 1)
#     noised_img = transforms.ToPILImage()(noised.cpu())
#     return noised_img

def add_noise_to_image(image_tensor, model_sampling, noise_std=0.2):
    """
    Add noise to an image tensor in [-1, 1] using a scaled blend.
    """
    noise = torch.randn_like(image_tensor) * noise_std
    noisy = image_tensor + noise

    return noisy.clamp(-1, 1)

def main():
    # clear any chache
    torch.cuda.empty_cache()
    debug = False # need to find the use
    args = parse_args()

    

    # if not os.path.exists(os.path.join(args.dataset_name, "train\\metadata.jsonl")):
    #    fnames = get_all_images_in_folder(args.dataset_name)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)



    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )

    # Make one log on every process with the configuration for debugging.
    
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)    

    # weight_dtype = get_dtype(accelerator.mixed_precision)
    print("num process", accelerator.num_processes)
    # print("working with", weight_dtype)

        
    vae = VAE(args.vae_model, dtype=torch.float32)
    vae.model.requires_grad_(True)

    

    # # Load ema vae
    # if args.use_ema:
    #     try:
    #         ema_vae = AutoencoderKL.from_pretrained(
    #             args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float32)
    #     except:
    #         ema_vae = AutoencoderKL.from_pretrained(
    #             args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32)
    #     ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)
    #     # no need to do special loading

    # vae.model.to(accelerator.device)#, dtype=weight_dtype)
    # vae.model.encoder.to(accelerator.device)#, dtype=weight_dtype)
    # vae.model.decoder.to(accelerator.device)#, dtype=weight_dtype)

    if args.gradient_checkpointing:
        vae.model.enable_gradient_checkpointing()
    
    if args.train_only_decoder:
        # freeze the encoder weights
        for param in vae.model.encoder.parameters():
            param.requires_grad = False
        # set encoder to eval mode
        vae.model.encoder.eval()

    # if True:

    #     for param in vae.model.parameters():
    #         if param.requires_grad:
    #             param.data.copy_(param.data.to(torch.float32))
    # else:
    #     vae.model = vae.model.half()

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes for 8-bit Adam. Run `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows")

        optimizer_cls = bnb.optim.AdamW8bit
        optimizer = optimizer_cls(
            vae.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    elif args.use_came:
        optimizer_cls = CAME
        optimizer = optimizer_cls(
            vae.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999),
            weight_decay=0.01
        )
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            vae.model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            num_proc=1,
            token=""
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
        
    train_transforms = v2.Compose(
        [
            v2.Resize(
                args.resolution, interpolation=v2.InterpolationMode.BILINEAR
            ),
            v2.CenterCrop(args.resolution),
            v2.ToTensor(), # this is apparently going to be depreciated in the future, replacing with the following 2 lines
            # v2.ToImage(), 
            # v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
            #v2.ToDtype(weight_dtype)
        ]
    )

    model_sampling = ModelSamplingDiscreteFlow()

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_column]]

        src_pixel_values = []
        tgt_pixel_values = []

        for img in images:
            # Add noise using your function
            # noisy_pil = add_noise_to_image(img, model_sampling)

            # Apply transforms
            tgt = train_transforms(img)  # clean image in [-1, 1]
            src = add_noise_to_image(tgt.clone(), model_sampling)  # noisy version

            src_pixel_values.append(src)
            tgt_pixel_values.append(tgt)

        examples["src_pixel_values"] = src_pixel_values
        examples["tgt_pixel_values"] = tgt_pixel_values
        return examples
    

    def collate_fn(examples):
        src_pixel_values = torch.stack([example["src_pixel_values"] for example in examples])
        tgt_pixel_values = torch.stack([example["tgt_pixel_values"] for example in examples])

        src_pixel_values = src_pixel_values.to(memory_format=torch.contiguous_format)
        tgt_pixel_values = tgt_pixel_values.to(memory_format=torch.contiguous_format)

        return {
            "src_pixel_values": src_pixel_values,
            "tgt_pixel_values": tgt_pixel_values
        }
    

    with accelerator.main_process_first():
        # Load test data from test_data_dir
        if (args.test_data_dir is not None and args.train_data_dir is not None):
            logger.info(f"load test data from {args.test_data_dir}")
            test_dir = os.path.join(args.test_data_dir, "**")
            test_dataset = load_dataset(
                "imagefolder",
                data_files=test_dir,
                cache_dir=args.cache_dir,
            )
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = test_dataset["train"].with_transform(preprocess)
        # Load train/test data from train_data_dir
        elif "test" in dataset.keys():
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)
        # Split into train/test
        else:
            dataset = dataset["train"].train_test_split(test_size=args.test_samples)
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)


    # persistent_workers=args.persistent_data_loader_workers,
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=n_workers  # args.train_batch_size*accelerator.num_processes,
    )

    # we use a batch size of 1 bc we want to see samples side by side, which is made by the validation sample function
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=1, num_workers=1,  # args.train_batch_size*accelerator.num_processes,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )
    
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"args.max_train_steps {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            # dirs = os.listdir(args.output_dir)
            dirs = os.listdir(args.resume_from_checkpoint)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(os.path.join(path))  # kiml
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    if args.patch_loss:
        patch_size = args.patch_size
        stride = args.patch_stride

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)
    lpips_loss_fn.requires_grad_(False)
    lpips_loss_fn.eval()  # added
    

    (
        vae.model, vae.model.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    ) = accelerator.prepare(
        vae.model, vae.model.decoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    lpips_loss_fn = accelerator.prepare(lpips_loss_fn)
    #if not args.train_only_decoder:
    # vae.model.encoder = accelerator.prepare(vae.model.encoder)


    for epoch in range(first_epoch, args.num_train_epochs):
        # with torch.amp.autocast("cuda", dtype=torch.float32):#accelerator.autocast():
        vae.model.train()
        accelerator.wait_for_everyone()
        train_loss = 0.0
        logger.info(f"{epoch = }")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(vae.model.decoder):
                src = batch["src_pixel_values"]
                target = batch["tgt_pixel_values"]#.to(accelerator.device, dtype=weight_dtype)
                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py

                if accelerator.num_processes > 1:
                    posterior = vae.model.encode(src).latent_dist
                    z = posterior.sample()#.to(weight_dtype)
                    pred = vae.model.decode(z).sample#.to(weight_dtype)
                else:
                    z, mu, logvar = vae.model.encode(src)#.to(weight_dtype)
                    # z = mean                      if posterior.mode()
                    # z = mean + variable*epsilon   if posterior.sample()
                    z = z #.to(weight_dtype) # Not mode()
                    pred = vae.model.decode(z) #.to(weight_dtype)

                # pred = pred#.to(dtype=weight_dtype)
                kl_loss = compute_kl(mu, logvar) #.to(weight_dtype)
                # kl_loss = posterior.kl().mean().to(weight_dtype)

                # mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                
                if args.patch_loss:
                    # patched loss
                    mse_loss = patch_based_mse_loss(target, pred, patch_size, stride) #.to(weight_dtype)
                    lpips_loss = patch_based_lpips_loss(lpips_loss_fn, target, pred, patch_size, stride) #.to(weight_dtype)

                else:
                    # default loss
                    mse_loss = F.mse_loss(pred, target, reduction="mean") #.to(weight_dtype)
                    with torch.no_grad():
                        lpips_loss = lpips_loss_fn(pred, target).mean() #.to(weight_dtype)
                        if not torch.isfinite(lpips_loss):
                            lpips_loss = torch.tensor(0)
                            
                if args.train_only_decoder:
                    # remove kl term from loss, bc when we only train the decoder, the latent is untouched
                    # and the kl loss describes the distribution of the latent
                    loss = (args.mse_scale * mse_loss 
                            + args.lpips_scale*lpips_loss)  # .to(weight_dtype)
                else:
                    loss = (mse_loss 
                            + args.lpips_scale*lpips_loss 
                            + args.kl_scale*kl_loss)  # .to(weight_dtype)
                if args.gradient_accumulation_steps and args.gradient_accumulation_steps>1:
                    loss = loss/args.gradient_accumulation_steps 

                if not torch.isfinite(loss):
                    pred_mean = pred.mean()
                    target_mean = target.mean()
                    logger.info("\nWARNING: non-finite loss, ending training ")
                
                if debug and step < 1:
                    print(f"loss dtype: {loss.dtype}")
                    print(f"pred dtype: {pred.dtype}")
                    print(f"target dtype: {target.dtype}")
                    print(f"z dtype: {z.dtype}")
                    print(f"kl_loss dtype: {kl_loss.dtype}")
                    print(f"mse_loss dtype: {mse_loss.dtype}")
                    print(f"lpips_loss dtype: {lpips_loss.dtype}")
                    print(f"vae parameters dtype: {[param.dtype for param in vae.parameters()]}")


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes

            # Gather the losses across all processes for logging (if we use distributed training).
            if loss is not None:
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps
            else:
                logger.warning("Loss not defined, skipping gathering.")
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": args.mse_scale * mse_loss.detach().item(),
                "lpips": args.lpips_scale * lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    logger.info("Validating model....")
                    with torch.no_grad():
                        log_validation(test_dataloader, vae, accelerator, global_step)
                if global_step % args.model_saving_steps == 0:
                    vae.model = accelerator.unwrap_model(vae.model)
                    vae.save(os.path.join(args.output_dir, f"{global_step}-finetuned.pth"))


        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                with torch.no_grad():
                    log_validation(test_dataloader, vae, accelerator, global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae.model = accelerator.unwrap_model(vae.model)
        # if args.use_ema:
        #     ema_vae.copy_to(vae.parameters())
        vae.save(os.path.join(args.output_dir, "finetuned.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    main()