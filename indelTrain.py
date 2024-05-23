#!/usr/bin/env python

# 设置参数
import argparse
import pathlib
parser = argparse.ArgumentParser(description="用DNABert（https://arxiv.org/pdf/2306.15006）和cascaded扩散模型（https://doi.org/10.48550/arXiv.2106.15282）预测CRIPSR/Cas编辑结果。")
parser.add_argument("--align_path", nargs='+', default='test/test.alg', type=pathlib.Path, help="file list stores alignments")
parser.add_argument("--ref_path", nargs='+', default='test/test.ref', type=pathlib.Path, help="file list stores reference plain sequences (one line for each reference)")
parser.add_argument("--condition_path", nargs='+', default='test/test.scaffold', type=pathlib.Path, help="file list stores condition plain sequences (one line for each condition)")
parser.add_argument("--max_length", default=None, help="maximal number of tokens for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
parser.add_argument("--truncation", default=True, help="truncation for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
parser.add_argument("--padding", default=True, help="padding for DNABert tokenizer (https://pic4.zhimg.com/v2-bdfb30d0379a9e85c97821afe6fe983f_r.jpg)")
args = parser.parse_args()

# 读取数据 （使用batch=True来加速）
import datasets
data = datasets.load_dataset("test_dataset/sx_test_dataset.py")

import torch
from diffusers import DiffusionPipeline

from transformers import AutoTokenizer, AutoModel
# 从huggingface下载DNABERT的模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
