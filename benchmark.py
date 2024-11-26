#!/usr/bin/env python

import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_col
from AI_models.benchmark.bench_lib import all_benchmark
from AI_models.proxy import proxy

proxy(url="socks5h://127.0.0.1:1080")

black_list = [[27], [100]]
for data_name in ["SX_spcas9", "SX_spymac", "SX_ispymac"]:
    all_benchmark(data_name, black_list)

    sns.pairplot(
        pd.read_csv(f"AI_models/benchmark/results/{data_name}_accum.csv")
    ).savefig(f"AI_models/benchmark/results/{data_name}_pair_correlation.png")

    benchmark_df = pd.read_csv(f"AI_models/benchmark/results/{data_name}_benchmark.csv")

    (
        ggplot(
            benchmark_df.query('pearson_type == "mhless_acc" or pearson_type == "total_acc"'),
            aes('pearson_type', 'value', fill='model')
        ) +
        geom_col(position="dodge")
    ).save(f"AI_models/benchmark/results/{data_name}_accum_correlation.png")

    (
        ggplot(
            benchmark_df
                .query('stat == "likelihood"')
                [['value', 'model']]
                .groupby(['model']).sum().reset_index(),
            aes('model', 'value', fill='model')
        ) +
        geom_col()
    ).save(f"AI_models/benchmark/results/{data_name}_likelihood.png")
