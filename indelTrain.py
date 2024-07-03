#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)

# schedulers
# pipeline.scheduler.compatibles for compatible schedulers
# PNDMScheduler: default for stable diffusion
# DPMSolverMultistepScheduler: more performant

# Notes
# We strongly suggest always running your pipelines in float16, and so far, we’ve rarely seen any degradation in output quality (https://huggingface.co/docs/diffusers/stable_diffusion).

from tqdm.auto import tqdm
from accelerate import Accelerator
import os
import torch
from huggingface_hub import create_repo, upload_folder
from CRISPRdiffuser.load_data import train_dataloader, valid_dataloader
from CRISPRdiffuser.diffuser_model import diffuser_model
from CRISPRdiffuser.diffuser_schedular import diffuser_scheduler
from CRISPRdiffuser.optimizer import optimizer
from CRISPRdiffuser.learn_schedular import learn_scheduler
from CRISPRdiffuser.loss_function import negative_likelihood
from CRISPRdiffuser.config import args, reflen
from CRISPRdiffuser.visualize import sample_images

# Initialize accelerator and tensorboard logging
output_dir = args.data_file.parent / "output"
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(output_dir, "logs")
)

if accelerator.is_main_process:
    os.makedirs(output_dir, exist_ok=True)
    if args.push_to_hub:
        repo_id = create_repo(
            repo_id="ljw20180420/CRISPRdiffuser", exist_ok=True
        ).repo_id
    accelerator.init_trackers(args.data_file.as_posix())

# Prepare everything
# There is no specific order to remember, you just need to unpack the
# objects in the same order you gave them to the prepare method.
diffuser_model, optimizer, train_dataloader,learn_scheduler = accelerator.prepare(
    diffuser_model, optimizer, train_dataloader, learn_scheduler
)

global_step = 0
# Train the model
for epoch in range(args.num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        # generate batch of clean images from ref2_start, ref1_end, count
        clean_images=torch.stack([
            torch.sparse_coo_tensor(
                indices=torch.stack([
                    batch['ref2_start'][i],
                    batch['ref1_end'][i]
                ]),
                values=batch['count'][i], 
                size=(reflen, reflen)
            ).to_dense()
            for i in range(len(batch['condition']))
        ])

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        batch_size = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, diffuser_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device,
            dtype=torch.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = diffuser_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(diffuser_model):
            # Predict the noise residual
            # TODO: use both condition and noisy_images to predict noise
            noise_pred = diffuser_model(noisy_images, timesteps, return_dict=False)[0]
            loss = negative_likelihood(noise_pred, noise)
            accelerator.backward(loss)

            # Clip gradient so that its norm equals max_norm. This prevents gradient from exploding
            accelerator.clip_grad_norm_(diffuser_model.parameters(), args.max_norm)
            optimizer.step()
            learn_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": learn_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

    # After each epoch you optionally sample some demo images with evaluate() and save the model
    if accelerator.is_main_process:
        # TODO: generate several images based on random conditions
        # diffuser_model needs to be unwrapped by accelerator.unwrap_model(diffuser_model)
        if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # generate images here
            # images = ...
            

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)
            if config.push_to_hub:

                upload_folder(

                    repo_id=repo_id,

                    folder_path=config.output_dir,

                    commit_message=f"Epoch {epoch}",

                    ignore_patterns=["step_*", "epoch_*"],

                )

            else:

                
    

# from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
