#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)

# schedulers
# pipeline.scheduler.compatibles for compatible schedulers
# PNDMScheduler: default for stable diffusion
# DPMSolverMultistepScheduler: more performant

# Notes
# We strongly suggest always running your pipelines in float16, and so far, we’ve rarely seen any degradation in output quality (https://huggingface.co/docs/diffusers/stable_diffusion).

from datetime import datetime
import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from accelerate import Accelerator
import subprocess
from config import ref1len, ref2len, args, device
from noise_scheduler import noise_scheduler
from model import stationary_sampler1, stationary_sampler2
from loss_function import continuous_time_loss_function
from load_data import valid_dataloader
from inference import inference_function
from save import save_model, push_model, save_heatmap

def train_loop():
    ymd = f"{datetime.now():%Y-%m-%d}"
    hms = f"{datetime.now():%H:%M:%S}"
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=(args.data_file.parent / "output" / "tensorboard" / ymd).as_posix()
    )
    if accelerator.is_main_process:
        # accelerator.init_trackers(args.data_file.parent.name, config={
        #     "mixed precision": args.mixed_precision,
        #     "gradient accumulation steps": args.gradient_accumulation_steps,
        #     "total epoch": args.epoch_num,
        #     "noise time steps": args.noise_timesteps,
        #     "maximal norm": args.max_norm
        # })
        accelerator.init_trackers(args.data_file.parent.name)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    from model import model
    from optimizer import optimizer
    from learn_schedular import learn_scheduler
    from load_data import train_dataloader
    model, optimizer, train_dataloader, learn_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, learn_scheduler
    )

    global_step = 0
    # Train the model
    for epoch in range(args.epoch_num):
        progress_bar = tqdm.auto.tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # recalculate the batch size because the last batch may be smaller than args.batch_size
            batch_size = batch['condition'].shape[0]
            # sample ref2_start and ref1_end
            x_cross0 = Categorical(probs=batch['observation'].view(batch_size, -1)).sample()
            x20 = x_cross0 // (ref1len + 1)
            x10 = x_cross0 % (ref1len + 1)
            # sample time and forward diffusion
            t = noise_scheduler(torch.rand(batch_size, device=device) * args.noise_timesteps)
            alpha_t = torch.e ** (-t)
            mask = torch.rand(batch_size, device=device) < alpha_t
            x1t = x10 * mask + stationary_sampler1.sample(torch.Size([batch_size])) * ~mask
            mask = torch.rand(batch_size, device=device) < alpha_t
            x2t = x20 * mask + stationary_sampler2.sample(torch.Size([batch_size])) * ~mask
            with accelerator.accumulate(model):
                # inference and apply softmax to the result
                values = F.one_hot(x1t, num_classes=ref1len + 1).view(batch_size, 1, -1) * F.one_hot(x2t, num_classes=ref2len + 1).view(batch_size, -1, 1)
                p_theta_0_logit = model(
                    torch.cat((
                        values[:, None, :, :],
                        batch['condition']
                    ), dim = 1),
                    t
                )
                p_theta_0 = F.softmax(
                    p_theta_0_logit.view(batch_size, -1),
                    dim = 1
                ).view(batch_size, ref2len + 1, ref1len + 1)
                # calculate loss
                loss = continuous_time_loss_function(alpha_t, x1t, x2t, p_theta_0, batch)
                assert not torch.isnan(loss), "the loss function is nan"
                accelerator.backward(loss)
                # Clip gradient so that its norm equals max_norm. This prevents gradient from exploding
                accelerator.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                learn_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            cuda_memory = subprocess.run("nvidia-smi | grep 8192MiB | sed -r 's/^.*[^0-9]([0-9]+MiB) \\/  8192MiB.*$/\\1/'", shell=True, capture_output=True).stdout.decode()
            cuda_memory = int(cuda_memory[:-4])
            logs = {"loss": loss.detach().item(), "lr": learn_scheduler.get_last_lr()[0], "cuda memory": cuda_memory, "step": step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # after each epoch
        if accelerator.is_main_process:
            # calculate the validation loss
            with torch.no_grad():
                valid_cross_entropy = 0
                valid_cross_entropy_count = 0
                valid_global_step = 0
                for valid_epoch in range(args.valid_epoch_num):
                    valid_progress_bar = tqdm.auto.tqdm(total=len(valid_dataloader), disable=not accelerator.is_local_main_process)
                    valid_progress_bar.set_description(f"Valid_Epoch {valid_epoch}")
                    for step, batch in enumerate(valid_dataloader):
                        # optionally save the inference process as images
                        do_save = ((epoch + 1) % args.save_image_epochs == 0 or epoch == args.epoch_num - 1) and (valid_epoch % args.save_image_valid_epochs == 0) and (step % args.save_image_valid_batchs == 0)
                        if do_save:
                            for in_batch in range(0, batch['observation'].shape[0], args.save_image_valid_in_batchs):
                                save_heatmap(batch['observation'][in_batch] / batch['observation'][in_batch].sum(), args.data_file.parent / "output" / "images" / ymd / f"epoch{epoch}" / f"valid_epoch{valid_epoch}" / f"valid_batch{step}" / f"ob{in_batch}.png")
                        save_dir = args.data_file.parent / "output" / "images" if do_save else None
                        # cross entropy
                        this_valid_cross_entropy = inference_function(accelerator.unwrap_model(model), step, batch, valid_epoch, epoch, ymd=ymd, save_dir=save_dir)
                        assert not torch.isnan(this_valid_cross_entropy), "the validation cross entropy is nan"
                        valid_cross_entropy += this_valid_cross_entropy
                        valid_cross_entropy_count += batch['condition'].shape[0]

                        valid_progress_bar.update(1)
                        cuda_memory = subprocess.run("nvidia-smi | grep 8192MiB | sed -r 's/^.*[^0-9]([0-9]+MiB) \\/  8192MiB.*$/\\1/'", shell=True, capture_output=True).stdout.decode()
                        cuda_memory = int(cuda_memory[:-4])
                        logs = {"cross entropy": this_valid_cross_entropy.item(), "cuda memory": cuda_memory, "step": step}
                        valid_progress_bar.set_postfix(**logs)
                        valid_global_step += 1
                
                accelerator.log(
                    {
                        "mean_valid_cross_entropy": valid_cross_entropy / valid_cross_entropy_count
                    }, 
                    step=global_step
                )

            # optionally save the model
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epoch_num - 1:
                save_model(accelerator.unwrap_model(model), ymd, hms, epoch)
                # optionally push the model to hugginface hub
                if args.push_to_hub:
                    push_model(ymd, hms, epoch)
    
    accelerator.end_training()

        

