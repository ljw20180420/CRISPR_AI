#!/usr/bin/env python

# from AI_models.CRISPR_diffuser.train import train
# train()

# from AI_models.CRISPR_diffuser.test import test
# test()

from AI_models.CRISPR_diffuser.inference import inference
for x1ts, x2ts, ts in inference():
    pass
