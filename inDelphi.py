#!/usr/bin/env python

from AI_models.inDelphi.train import train_deletion, train_insertion
from AI_models.inDelphi.test import test
from diffusers import DiffusionPipeline

# train_deletion
# train_insertion
test()

pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_inDelphi", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_inDelphi")
