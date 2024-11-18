#!/usr/bin/env python

import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_col, geom_bar
from AI_models.benchmark.bench_lib import all_benchmark

data_name = 'SX_spcas9'
black_list = [[27], [100]]
all_benchmark(data_name, black_list)

sns.pairplot(
    pd.read_csv("AI_models/benchmark/accum.csv")
).savefig("AI_models/benchmark/pair_correlation.png")

benchmark_df = pd.read_csv("AI_models/benchmark/benchmark.csv")

(
    ggplot(
        benchmark_df.query('pearson_type == "mhless_acc" or pearson_type == "total_acc"'),
        aes('pearson_type', 'value', fill='model')
    ) +
    geom_col(position="dodge")
).save("AI_models/benchmark/accum_correlation.png")

(
    ggplot(
        benchmark_df
            .query('stat == "likelihood"')
            [['value', 'model']]
            .groupby(['model']).sum().reset_index(),
        aes('model', 'value', fill='model')
    ) +
    geom_col()
).save("AI_models/benchmark/likelihood.png")
