#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .inference import data_collector_inference
from ..config import get_config, get_logger
from .scheduler import scheduler
from .model import CRISPRDiffuserConfig, CRISPRDiffuserModel
from .perfect_model import PerfectModel
from .pipeline import CRISPRDiffuserPipeline

args = get_config("config_CRISPR_diffuser.ini")
logger = get_logger(args)

logger.info("get scheduler")
noise_scheduler = scheduler()

logger.info("load model")
# CRISPR_diffuser_model = CRISPRDiffuserModel.from_pretrained(args.output_dir / CRISPRDiffuserConfig.model_type / f"{args.data_name}_{CRISPRDiffuserConfig.model_type}")
perfect_model = PerfectModel(CRISPRDiffuserConfig())

logger.info("setup pipeline")
pipe = CRISPRDiffuserPipeline(
    unet=perfect_model,
    scheduler=noise_scheduler
)
pipe.unet.to(args.device)

@torch.no_grad()
def to_numpy(xts):
    if not isinstance(xts[0], torch.Tensor):
        return xts
    return [xt.cpu().numpy() for xt in xts]

@torch.no_grad()
def draw(x1ts, x2ts, filename):
    x1ts, x2ts = to_numpy(x1ts), to_numpy(x2ts)
    fig, ax = plt.subplots()
    scat = ax.scatter(x1ts[0], x2ts[0], c="b", s=5)
    ax.set(xlim=[0, args.ref1len], ylim=[0, args.ref2len], xlabel='ref1', ylabel='ref2')

    def update(frame):
        scat.set_offsets(
            np.stack([x1ts[frame], x2ts[frame]]).T
        )
        return scat
 
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(x1ts), interval=240)
    ani.save(filename=filename, writer="pillow")
    plt.close()

@torch.no_grad()
def forward(step, x10=100, x20=27, batch_size=args.batch_size):
    x1ts, x2ts = [torch.tensor([x10] * batch_size)], [torch.tensor([x20] * batch_size)]
    ts = [
        noise_scheduler.step_to_time(torch.tensor([0] * batch_size)),
        noise_scheduler.step_to_time(torch.tensor([step] * batch_size))
    ]
    x1t, x2t = noise_scheduler.add_noise(
        x1ts[0],
        x2ts[0],
        ts[-1],
        pipe.stationary_sampler1,
        pipe.stationary_sampler2
    )
    x1ts.append(x1t)
    x2ts.append(x2t)
    return x1ts, x2ts, ts

@torch.no_grad()
def reverse(x1t, x2t, t, ref, cut, step):
    batch_size = len(t)
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
            "t": t.to(pipe.unet.device)
        },
        batch["condition"].to(pipe.unet.device).expand(batch_size, -1, -1, -1)
    )["p_theta_0_logit"]
    fig, ax = plt.subplots()
    ax.imshow(
        F.softmax(
            p_theta_0_logit.view(batch_size, -1),
            dim=1
        ).mean(axis=0).view(args.ref2len + 1, args.ref1len + 1).cpu().numpy() ** args.display_scale_factor
    )
    fig.savefig(f"zshit/reverse_{step}.png")
    plt.close()

@torch.no_grad()
def dynamics():
    # logger.info("calculate")
    # ref="AAAAAAAAAAAAAAAAAAAAAAAAGACGGCAGCCTTTTGACCTCCCAACCCCCCTATAGTCAGATAGTCAAGAAGGGCATTATCTGGCTTACCTGAATCGTCCCAAGAATTTTCTTCGGTGAGCATTTGTGGAGACCCTGGGATGTAGGTTGGATTAAACTGTGATGGGTCCATCGGCGTCTTGACACAACACTAGGCTT"
    # cut=100
    # for step in range(args.noise_timesteps + 1):
    #     x1ts, x2ts, ts = forward(step, x10=100, x20=27, batch_size=args.batch_size)
    #     draw(x1ts, x2ts, f'zshit/forward_{step}.gif')
    #     reverse(x1ts[-1], x2ts[-1], ts[-1], ref, cut, step)
    

    logger.info("do inference")
    ref="AAAAAAAAAAAAAAAAAAAAAAAAGACGGCAGCCTTTTGACCTCCCAACCCCCCTATAGTCAGATAGTCAAGAAGGGCATTATCTGGCTTACCTGAATCGTCCCAAGAATTTTCTTCGGTGAGCATTTGTGGAGACCCTGGGATGTAGGTTGGATTAAACTGTGATGGGTCCATCGGCGTCTTGACACAACACTAGGCTT"
    cut=100
    batch = data_collector_inference(
        [{
            "ref": ref,
            "cut": cut
        }],
        noise_scheduler,
        pipe.stationary_sampler1,
        pipe.stationary_sampler2
    )
    x1ts, x2ts, ts = pipe(batch, batch_size=args.batch_size, record_path=True)
    draw(x1ts, x2ts, "zshit/inference.gif")
    
