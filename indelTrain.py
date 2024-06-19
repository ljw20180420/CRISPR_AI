#!/usr/bin/env python
# DNABert (https://arxiv.org/pdf/2306.15006)


# 设置参数
import argparse
import pathlib
parser = argparse.ArgumentParser(description="use cascaded diffuser (https://doi.org/10.48550/arXiv.2106.15282) to predict CRIPSR/Cas9 editing")
parser.add_argument("--align_path", nargs='+', default='test/test.alg', type=pathlib.Path, help="file list stores alignments")
parser.add_argument("--ref_path", nargs='+', default='test/test.ref', type=pathlib.Path, help="file list stores reference plain sequences (one line for each reference)")
parser.add_argument("--condition_path", nargs='+', default='test/test.scaffold', type=pathlib.Path, help="file list stores condition plain sequences (one line for each condition)")
parser.add_argument("--max_length", default=None, help="maximal number of tokens for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
parser.add_argument("--truncation", default=True, help="truncation for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
parser.add_argument("--padding", default=True, help="padding for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
args = parser.parse_args()

# read data
from datasets import load_dataset, Features, Value
alg_features = Features({
    'index': Value('uint64'),
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
