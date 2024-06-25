#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)

# 设置参数
import argparse
import pathlib
parser = argparse.ArgumentParser(description="use cascaded diffuser (https://doi.org/10.48550/arXiv.2106.15282) to predict CRIPSR/Cas9 editing")
parser.add_argument("--data_files", nargs='+', type=pathlib.Path, help="files after preprocess")
args = parser.parse_args()

# read data
from datasets import load_dataset, Features, Value, Sequence
alg_features = Features({
    'cut1': Value('int16'),
    'cut2': Value('int16'),
    'ref1': Value('string'),
    'ref2': Value('string'),
    'ref1_end': Sequence(Value('int16')),
    'ref2_start': Sequence(Value('int16')),
    'count': Sequence(Value('uint64'))
})

ds = load_dataset("json", data_files=args.data_files.as_posix(), num_proc=12, features=alg_features)



import torch
from diffusers import DiffusionPipeline

from transformers import AutoTokenizer, AutoModel

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
