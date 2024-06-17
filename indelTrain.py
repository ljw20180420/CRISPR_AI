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

# 读取数据，拆成训练集，检验集，测试集。
import datasets
import random
# 设置随机种子以便重复
random.seed(63036)
our_data_sets = ["test_dataset/sx_test_dataset.py"]
# 设置streaming=True流读取（https://hf-mirror.com/docs/datasets/v2.19.0/en/stream#stream），可以节约内存。
data_streams = [datasets.load_dataset(our_data_set, streaming=True)["train"] for our_data_set in our_data_sets]
# 将所有数据库的stream合并
combined_stream = datasets.interleave_datasets(data_streams)
# 打乱序列
combined_stream = combined_stream.shuffle(buffer_size=10_000, seed=random.randint(0, 999_999))
# 提取测试数据和检验数据，剩下的作为训练数据
test_size, valid_size = 50, 50
test_data_stream = combined_stream.take(test_size)
valid_data_stream = combined_stream.skip(test_size).take(valid_size)
train_data_stream = combined_stream.skip(test_size + valid_size)

import torch
from diffusers import DiffusionPipeline

from transformers import AutoTokenizer, AutoModel
# 从huggingface下载DNABERT的模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
