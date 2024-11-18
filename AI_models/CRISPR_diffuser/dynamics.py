#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .inference import data_collector_inference
from ..config import get_config
from .scheduler import CRISPRDiffuserLinearScheduler, CRISPRDiffuserCosineScheduler, CRISPRDiffuserExpScheduler, CRISPRDiffuserUniformScheduler
from .pipeline import CRISPRDiffuserPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = ""

args = get_config("config_CRISPR_diffuser.ini")

@torch.no_grad()
def to_numpy(xts):
    if not isinstance(xts[0], torch.Tensor):
        return xts
    return [xt.cpu().numpy() for xt in xts]

@torch.no_grad()
def draw(x1ts: torch.Tensor, x2ts: torch.Tensor, filename, interval=120, pad=5):
    x1ts, x2ts = to_numpy(x1ts), to_numpy(x2ts)
    fig, ax = plt.subplots()
    scat = ax.scatter(x1ts[0], x2ts[0], c="b", s=5)
    ax.set(xlim=[0, args.ref1len], ylim=[0, args.ref2len], xlabel='ref1', ylabel='ref2')

    def update(frame):
        idx = min(frame, len(x1ts) - 1)
        scat.set_offsets(
            np.stack([x1ts[idx], x2ts[idx]]).T
        )
        return scat
 
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(x1ts) + pad, interval=interval)
    ani.save(filename=filename, writer="pillow")
    plt.close()

@torch.no_grad()
def forward(x10: torch.Tensor, x20: torch.Tensor, step: torch.Tensor, noise_scheduler: CRISPRDiffuserLinearScheduler | CRISPRDiffuserCosineScheduler | CRISPRDiffuserExpScheduler | CRISPRDiffuserUniformScheduler, pipe: CRISPRDiffuserPipeline) -> Tuple[torch.Tensor, torch.Tensor]:
    # add noise to x10, x20 at step 0, return x1t, x2t at step step (step is not time)
    return noise_scheduler.add_noise(
        x10,
        x20,
        noise_scheduler.step_to_time(step),
        pipe.stationary_sampler1,
        pipe.stationary_sampler2
    )

@torch.no_grad()
def reverse(x1t: torch.Tensor, x2t: torch.Tensor, step: torch.Tensor, ref: str, cut: int, noise_scheduler: CRISPRDiffuserLinearScheduler | CRISPRDiffuserCosineScheduler | CRISPRDiffuserExpScheduler | CRISPRDiffuserUniformScheduler, pipe: CRISPRDiffuserPipeline) -> torch.Tensor:
    batch_size = len(step)
    batch = data_collector_inference(
        [{
            "ref": ref,
            "cut": cut
        }],
        noise_scheduler,
        pipe.stationary_sampler1,
        pipe.stationary_sampler2
    )
    p_theta_0_logit = pipe.unet(
        {
            "x1t": x1t.to(pipe.unet.device),
            "x2t": x2t.to(pipe.unet.device),
            "t": noise_scheduler.step_to_time(step).to(pipe.unet.device)
        },
        batch["condition"].to(pipe.unet.device).expand(batch_size, -1, -1, -1)
    )["p_theta_0_logit"]
    
    return F.softmax(
        p_theta_0_logit.view(batch_size, -1),
        dim=1
    ).mean(axis=0).view(args.ref2len + 1, args.ref1len + 1) ** args.display_scale_factor

@torch.no_grad()
def dynamics(batch_size: int, epoch: int, ref: str, cut: int, noise_scheduler: CRISPRDiffuserLinearScheduler | CRISPRDiffuserCosineScheduler | CRISPRDiffuserExpScheduler | CRISPRDiffuserUniformScheduler, pipe: CRISPRDiffuserPipeline) -> List[torch.Tensor]:
    batch = data_collector_inference(
        [{
            "ref": ref,
            "cut": cut
        }],
        noise_scheduler,
        pipe.stationary_sampler1,
        pipe.stationary_sampler2
    )
    x1ts_all, x2ts_all = None, None
    for ep in range(epoch):
        x1ts, x2ts, _ = pipe(batch, batch_size=batch_size, record_path=True)
        if not x1ts_all:
            x1ts_all = x1ts
            x2ts_all = x2ts
            continue
        x1ts_all = [torch.cat([x1t_all, x1t]) for x1t_all, x1t in zip(x1ts_all, x1ts)]
        x2ts_all = [torch.cat([x2t_all, x2t]) for x2t_all, x2t in zip(x2ts_all, x2ts)]
    
    return x1ts_all, x2ts_all
    
