#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)

# 设置参数
import argparse
import pathlib
parser = argparse.ArgumentParser(description="use cascaded diffuser (https://doi.org/10.48550/arXiv.2106.15282) to predict CRIPSR/Cas9 editing")
parser.add_argument("--data_files", nargs='+', type=pathlib.Path, help="files after preprocess")
args = parser.parse_args()

# read data
from datasets import load_dataset, Features, Value
alg_features = Features({
    'count': Value('uint64'),
    'score': Value('float32'),
    'ref_id': Value('uint32'),
    'up_dangle': Value('string'),
    'ref1_start': Value('int32'),
    'query1_start': Value('int32'),
    'ref1_end': Value('int32'),
    'query1_end': Value('int32'),
    'random_insert': Value('string'),
    'ref2_start': Value('int32'),
    'query2_start': Value('int32'),
    'ref2_end': Value('int32'),
    'query2_end': Value('int32'),
    'down_dangle': Value('string'),
    'cut1': Value('int32'),
    'cut2': Value('int32')
})
sx_data = load_dataset("csv", data_files="/home/ljw/sdc1/SX/algs/*", num_proc=12, delimiter="\t", column_names=['index', 'count', 'score', 'ref_id', 'up_dangle', 'ref1_start', 'query1_start', 'ref1_end', 'query1_end', 'random_insert', 'ref2_start', 'query2_start', 'ref2_end', 'query2_end', 'down_dangle', 'cut1', 'cut2', 'ref', 'query'], features=alg_features, keep_default_na=False, with_file_names=True)

import torch
from diffusers import DiffusionPipeline

from transformers import AutoTokenizer, AutoModel
# 从huggingface下载DNABERT的模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
