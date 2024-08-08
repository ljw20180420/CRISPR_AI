#!/usr/bin/env python

from datetime import datetime
import tqdm
from accelerate import Accelerator
import torch
from loss_function import kl_divergence
from config import args
from load_data import valid_dataloader
from save import save_model, push_model
import subprocess

def train_loop():
    ymd = f"{datetime.now():%Y-%m-%d}"
    hms = f"{datetime.now():%H:%M:%S}"
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=(args.data_file.parent / "FOREcasT" /"output" / "tensorboard" / ymd).as_posix()
    )
    if accelerator.is_main_process:
        # accelerator.init_trackers(args.data_file.parent.name, config={
        #     "mixed precision": args.mixed_precision,
        #     "gradient accumulation steps": args.gradient_accumulation_steps,
        #     "total epoch": args.epoch_num,
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
            with accelerator.accumulate(model):
                logits = model(batch['data'])
                loss = kl_divergence(logits, batch['count'], model.reg_coff, model.theta)
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
                valid_loss = 0
                valid_loss_count = 0
                valid_progress_bar = tqdm.auto.tqdm(total=len(valid_dataloader), disable=not accelerator.is_local_main_process)
                valid_progress_bar.set_description("Validation")
                for step, batch in enumerate(valid_dataloader):
                    unwrap_model = accelerator.unwrap_model(model)
                    this_valid_loss = kl_divergence(unwrap_model(batch['data']), batch['count'], unwrap_model.reg_coff, unwrap_model.theta)
                    assert not torch.isnan(this_valid_loss), "the validation loss is nan"
                    valid_loss += this_valid_loss
                    valid_loss_count += batch['data'].shape[0]

                    valid_progress_bar.update(1)
                    cuda_memory = subprocess.run("nvidia-smi | grep 8192MiB | sed -r 's/^.*[^0-9]([0-9]+MiB) \\/  8192MiB.*$/\\1/'", shell=True, capture_output=True).stdout.decode()
                    cuda_memory = int(cuda_memory[:-4])
                    logs = {"loss": this_valid_loss.item(), "cuda memory": cuda_memory, "step": step}
                    valid_progress_bar.set_postfix(**logs)
            
                accelerator.log(
                    {
                        "mean validation loss": valid_loss / valid_loss_count
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







