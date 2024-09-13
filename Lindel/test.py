#!/usr/bin/env python

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_Lindel", trust_remote_code=True)