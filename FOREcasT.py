#!/usr/bin/env python

from AI_models.FOREcasT.train import train
from AI_models.FOREcasT.test import test
from diffusers import DiffusionPipeline
from AI_models.config import args

# train()
test()

pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_FOREcasT", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_FOREcasT", MAX_DEL_SIZE=args.FOREcasT_MAX_DEL_SIZE)