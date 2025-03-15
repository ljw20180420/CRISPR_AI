#!/usr/bin/env python

from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .load_data import data_collector, outputs_test
from .inference import data_collector_inference
from ..config import get_config
from .scheduler import scheduler
from .pipeline import DiffusionPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = ""

args = get_config("config_CRISPR_diffuser.ini")


@torch.no_grad()
def to_numpy(xts):
    if not isinstance(xts[0], torch.Tensor):
        return xts
    return [xt.cpu().numpy() for xt in xts]


@torch.no_grad()
def draw(x1ts, x2ts, filename, interval=120, pad=5):
    x1ts, x2ts = to_numpy(x1ts), to_numpy(x2ts)
    fig, ax = plt.subplots()
    scat = ax.scatter(x1ts[0], x2ts[0], c="b", s=5)
    ax.set(xlim=[0, args.ref1len], ylim=[0, args.ref2len], xlabel="ref1", ylabel="ref2")

    def update(frame):
        idx = min(frame, len(x1ts) - 1)
        scat.set_offsets(np.stack([x1ts[idx], x2ts[idx]]).T)
        return scat

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(x1ts) + pad, interval=interval
    )
    ani.save(filename=f"{filename}.gif", writer="pillow")
    fig.savefig(f"{filename}.png")
    plt.close()


@torch.no_grad()
def forward(
    x10: torch.Tensor, x20: torch.Tensor, step: torch.Tensor, noise_scheduler, pipe
) -> Tuple[torch.Tensor, torch.Tensor]:
    # add noise to x10, x20 at step 0, return x1t, x2t at step step (step is not time)
    return noise_scheduler.add_noise(
        x10,
        x20,
        noise_scheduler.step_to_time(step),
        pipe.stationary_sampler1,
        pipe.stationary_sampler2,
    )


@torch.no_grad()
def reverse(
    x1t: torch.Tensor,
    x2t: torch.Tensor,
    step: torch.Tensor,
    ref: str,
    cut: int,
    noise_scheduler,
    pipe,
) -> torch.Tensor:
    batch_size = len(step)
    batch = data_collector_inference(
        [{"ref": ref, "cut": cut}],
        noise_scheduler,
        pipe.stationary_sampler1,
        pipe.stationary_sampler2,
    )
    p_theta_0_logit = pipe.unet(
        {
            "x1t": x1t.to(pipe.unet.device),
            "x2t": x2t.to(pipe.unet.device),
            "t": noise_scheduler.step_to_time(step).to(pipe.unet.device),
        },
        batch["condition"].to(pipe.unet.device).expand(batch_size, -1, -1, -1),
    )["p_theta_0_logit"]

    return (
        F.softmax(p_theta_0_logit.view(batch_size, -1), dim=1)
        .mean(axis=0)
        .view(args.ref2len + 1, args.ref1len + 1)
        ** args.display_scale_factor
    )


@torch.no_grad()
def dynamics(
    data_name, text_idx=0, batch_size: int = 100, epoch: int = 1
) -> List[torch.Tensor]:
    noise_scheduler = scheduler(
        noise_scheduler=args.noise_scheduler,
        noise_timesteps=args.noise_timesteps,
        cosine_factor=args.cosine_factor,
        exp_scale=args.exp_scale,
        exp_base=args.exp_base,
        uniform_scale=args.uniform_scale,
    )

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_CRISPR_diffuser",
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    pipe = DiffusionPipeline.from_pretrained(
        f"{args.owner}/{data_name}_CRISPR_diffuser",
        trust_remote_code=True,
        custom_pipeline=f"{args.owner}/{data_name}_CRISPR_diffuser",
    )
    pipe.unet.to(args.device)

    batch = data_collector(
        [ds["test"][text_idx]],
        noise_scheduler,
        pipe.stationary_sampler1,
        pipe.stationary_sampler2,
        outputs_test,
    )

    x1ts_all, x2ts_all, x1ts_perfect_all, x2ts_perfect_all = None, None, None, None
    for ep in range(epoch):
        # generate reverse diffusion from model
        x1ts, x2ts, _ = pipe(batch, batch_size=batch_size, record_path=True)
        if not x1ts_all:
            x1ts_all = x1ts
            x2ts_all = x2ts
        else:
            x1ts_all = [
                torch.cat([x1t_all, x1t]) for x1t_all, x1t in zip(x1ts_all, x1ts)
            ]
            x2ts_all = [
                torch.cat([x2t_all, x2t]) for x2t_all, x2t in zip(x2ts_all, x2ts)
            ]

        # generate reverse diffusion from observation
        x1t, x2t = x1ts[0], x2ts[0]
        t = noise_scheduler.step_to_time(
            torch.tensor([noise_scheduler.config.num_train_timesteps])
        )
        x1ts_perfect, x2ts_perfect = [x1t], [x2t]
        for timestep in tqdm(noise_scheduler.timesteps):
            if timestep >= t:
                continue
            # the scheduler automatically set t = timestep
            x1t, x2t, t = noise_scheduler.step(
                batch["observation"].log().expand(batch_size, -1, -1),
                x1t,
                x2t,
                t,
                pipe.stationary_sampler1,
                pipe.stationary_sampler2,
            )
            x1ts_perfect.append(x1t)
            x2ts_perfect.append(x2t)

        if not x1ts_perfect_all:
            x1ts_perfect_all = x1ts_perfect
            x2ts_perfect_all = x2ts_perfect
        else:
            x1ts_perfect_all = [
                torch.cat([x1t_perfect_all, x1t_perfect])
                for x1t_perfect_all, x1t_perfect in zip(x1ts_perfect_all, x1ts_perfect)
            ]
            x2ts_perfect_all = [
                torch.cat([x2t_perfect_all, x2t_perfect])
                for x2t_perfect_all, x2t_perfect in zip(x2ts_perfect_all, x2ts_perfect)
            ]

    return (
        torch.stack(x1ts_all),
        torch.stack(x2ts_all),
        torch.stack(x1ts_perfect_all),
        torch.stack(x2ts_perfect_all),
    )
