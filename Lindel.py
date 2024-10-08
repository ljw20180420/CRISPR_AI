#!/usr/bin/env python

from AI_models.Lindel.train import train
from AI_models.Lindel.test import test
from diffusers import DiffusionPipeline
from AI_models.config import args
from AI_models.proxy import *

# train()
test()

pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_Lindel", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_Lindel")
