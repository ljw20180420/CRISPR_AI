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
from CRISPRdiffuser.load_data import valid_dataloader, epoch_training_steps
from CRISPRdiffuser.train_schedular import diffuser_scheduler
from CRISPRdiffuser.loss_function import negative_likelihood
from CRISPRdiffuser.config import ref1len, ref2len, args
# from CRISPRdiffuser.visualize import sample_images
from CRISPRdiffuser.save import save_model, push_model
from CRISPRdiffuser.accelerator import accelerator, diffuser_model, optimizer, train_dataloader, learn_scheduler
from .inference import inferenc_edit

global_step = 0
# Train the model
for epoch in range(args.num_epochs):
    progress_bar = tqdm(total=epoch_training_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):


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

    if accelerator.is_main_process:
        # TODO: After each epoch
    
        # TODO: Calculate the validation loss

        # TODO: Optionally save the model
        if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            save_model()
            # TODO: optionally push the model to hugginface hub
            if args.push_to_hub:
                push_model(epoch)

        # TODO: Optionally sample some editing results
        if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # TODO: construct example
            inferenc_edit(example, 99999)

