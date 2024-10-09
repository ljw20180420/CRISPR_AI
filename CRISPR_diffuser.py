#!/usr/bin/env python

from AI_models.CRISPR_diffuser.train import train
from AI_models.CRISPR_diffuser.test import test
from diffusers import DiffusionPipeline

# train()
test()

pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_CRISPR_diffuser", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_CRISPR_diffuser")
