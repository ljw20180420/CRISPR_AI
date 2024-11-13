#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from AI_models.bench_lib import inDelphi_benchmark, inDelphi_original_benchmark, FOREcasT_benchmark, Lindel_benchmark

black_list = [[27], [100]]

def tensor_correlation(ten1: torch.Tensor, ten2: torch.Tensor):
    return (
        F.normalize(ten1.to(torch.float32), p=2., dim=0) * F.normalize(ten2.to(torch.float32), p=2., dim=0)
    ).sum()

inDelphi_likelihood, inDelphi_genotype_pearson, inDelphi_total_del_len_pearson, inDelphi_mhless_pearson, inDelphi_mhless_count_acc, inDelphi_total_del_len_count_acc, inDelphi_mhless_weight = inDelphi_benchmark(black_list)

inDelphi_mhless_pearson_acc = tensor_correlation(inDelphi_mhless_weight, inDelphi_mhless_count_acc)
inDelphi_mhless_total_pearson_acc = tensor_correlation(inDelphi_mhless_weight, inDelphi_total_del_len_count_acc)
fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0,0].scatter(inDelphi_mhless_weight, inDelphi_mhless_count_acc)
axs[0,1].scatter(range(1, len(inDelphi_mhless_count_acc) + 1), inDelphi_mhless_count_acc)
axs[0,2].scatter(range(1, len(inDelphi_mhless_weight) + 1), inDelphi_mhless_weight)
axs[1,0].scatter(inDelphi_mhless_weight, inDelphi_total_del_len_count_acc)
axs[1,1].scatter(range(1, len(inDelphi_total_del_len_count_acc) + 1), inDelphi_total_del_len_count_acc)
fig.savefig("zshit/inDelphi_corr.png")
plt.close(fig)

FOREcasT_likelihood = FOREcasT_benchmark(black_list)

Lindel_likelihood = Lindel_benchmark(black_list)

original_likelihood, original_genotype_pearson, original_total_del_len_pearson, original_mhless_pearson, original_mhless_weight, original_mhless_pearson_acc, original_mhless_total_pearson_acc = dict(), dict(), dict(), dict(), dict(), dict(), dict()
for celltype in ['mESC']: # ['mESC', 'U2OS', 'HEK293', 'HCT116', 'K562']
    original_likelihood[celltype], original_genotype_pearson[celltype], original_total_del_len_pearson[celltype], original_mhless_pearson[celltype], original_mhless_weight[celltype] = inDelphi_original_benchmark(black_list, celltype)
    original_mhless_pearson_acc[celltype] = tensor_correlation(original_mhless_weight[celltype], inDelphi_mhless_count_acc)
    original_mhless_total_pearson_acc[celltype] = tensor_correlation(original_mhless_weight[celltype], inDelphi_total_del_len_count_acc)

for celltype in ['mESC']:
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].scatter(original_mhless_weight[celltype], inDelphi_mhless_count_acc)
    axs[0,1].scatter(range(1, len(original_mhless_weight[celltype]) + 1), original_mhless_weight[celltype])
    axs[1,0].scatter(original_mhless_weight[celltype], inDelphi_total_del_len_count_acc)
    fig.savefig(f"zshit/original_{celltype}_corr.png")
    plt.close(fig)