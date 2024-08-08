#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)

# schedulers
# pipeline.scheduler.compatibles for compatible schedulers
# PNDMScheduler: default for stable diffusion
# DPMSolverMultistepScheduler: more performant

# Notes
# We strongly suggest always running your pipelines in float16, and so far, we’ve rarely seen any degradation in output quality (https://huggingface.co/docs/diffusers/stable_diffusion).

from accelerate import notebook_launcher
from train import train_loop

notebook_launcher(train_loop, num_processes=1)
