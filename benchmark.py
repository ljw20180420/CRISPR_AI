#!/usr/bin/env python

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
import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0,0].scatter(inDelphi_mhless_weight, inDelphi_mhless_count_acc)
axs[0,1].scatter(range(1, len(inDelphi_mhless_count_acc) + 1), inDelphi_mhless_count_acc)
axs[0,2].scatter(range(1, len(inDelphi_mhless_weight) + 1), inDelphi_mhless_weight)
axs[1,0].scatter(inDelphi_mhless_weight, inDelphi_total_del_len_count_acc)
axs[1,1].scatter(range(1, len(inDelphi_total_del_len_count_acc) + 1), inDelphi_total_del_len_count_acc)
fig.savefig("zshit/inDelphi_corr.png")

FOREcasT_likelihood = FOREcasT_benchmark(black_list)

Lindel_likelihood = Lindel_benchmark(black_list)

inDelphi_original_likelihood, inDelphi_original_total_del_len_pearson = dict(), dict()
for celltype in ['mESC']: # ['mESC', 'U2OS', 'HEK293', 'HCT116', 'K562']
    inDelphi_original_likelihood[celltype], inDelphi_original_total_del_len_pearson[celltype] = inDelphi_original_benchmark(black_list, celltype)