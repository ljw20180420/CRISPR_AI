#!/usr/bin/env python

from datasets import load_dataset, Features, Value
from AI_models.config import args, logger

ds = load_dataset('json', data_files=args.inference_data, features=Features({
    'ref': Value('string'),
    'cut': Value('int16')
}))

from dataset.CRISPR_data import CRISPRData






from AI_models.inDelphi.train import train_deletion, train_insertion
from AI_models.inDelphi.test import test
from diffusers import DiffusionPipeline
from huggingface_hub import HfFileSystem
import pickle

# train_deletion()
# train_insertion()
breakpoint()
test()

fs = HfFileSystem()
with fs.open("ljw20180420/SX_spcas9_inDelphi/inDelphi_model/insertion_model.pkl", "rb") as fd:
    onebp_features, insert_probabilities, m654 = pickle.load(fd)
pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_inDelphi", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_inDelphi", onebp_features = onebp_features, insert_probabilities = insert_probabilities, m654 = m654)
