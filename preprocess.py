#!/usr/bin/env python

import argparse
import pathlib
import re
import numpy as np
import polars as pl
import tempfile
import io
parser = argparse.ArgumentParser(description="preprocess alignments")
parser.add_argument("--data_dir", type=pathlib.Path, help="directory containing alignment files")
parser.add_argument("--score_quantile", type=float, default=0.05, help="alignment score quantile to filter score")
parser.add_argument("--min_score", type=float, default=-np.inf, help="min alignment score threshold")
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
    'cut2': Value('int32'),
    'ref': Value('string'),
    'query': Value('string')
})
acgtn_pattern = re.compile('[acgtn]')
def get_ref1_len(examples):
    refs = [ref.replace("-", "") for ref in examples['ref']]
    ref1_lens = [acgtn_pattern.search(ref[1:]).span()[1] + 1 for ref in refs]
    ref2_starts = [ref2_start - ref1_len for ref2_start, ref1_len in zip(examples['ref2_start'], ref1_lens)]
    cut2s = [cut2 - ref1_len for cut2, ref1_len in zip(examples['cut2'], ref1_lens)]
    ref1s = [ref[:ref1_len].upper() for ref1_len, ref in zip(ref1_lens, refs)]
    ref2s = [ref[ref1_len:].upper() for ref1_len, ref in zip(ref1_lens, refs)]
    return {
        "ref": refs,
        "ref1_len": ref1_lens,
        "ref2_start": ref2_starts,
        "cut2": cut2s,
        "ref1": ref1s,
        "ref2": ref2s
    }
alg_ds = (
    load_dataset("csv", data_files=(args.data_dir / "*").as_posix(), num_proc=12, delimiter="\t", column_names=['index', 'count', 'score', 'ref_id', 'up_dangle', 'ref1_start', 'query1_start', 'ref1_end', 'query1_end', 'random_insert', 'ref2_start', 'query2_start', 'ref2_end', 'query2_end', 'down_dangle', 'cut1', 'cut2', 'ref', 'query'], features=alg_features, keep_default_na=False)
    .remove_columns(['index', 'ref_id', 'up_dangle', 'ref1_start', 'query1_start', 'query1_end', 'random_insert', 'query2_start', 'ref2_end', 'query2_end', 'down_dangle', 'query'])
    .map(get_ref1_len, batched=True)
    .remove_columns(["ref", "ref1_len"])
)
score_thres = max(
    args.min_score,
    np.quantile(alg_ds['train']['score'], args.score_quantile)
)
alg_ds = (
    alg_ds.filter(lambda examples: [score >= score_thres for score in examples["score"]], batched=True)
    .remove_columns(["score"])['train']
)
parquet_io = io.BytesIO()
alg_ds.to_parquet(parquet_io)
del alg_ds
parquet_io.seek(0)
(
    pl.scan_parquet(parquet_io)
    .group_by(["ref1_end", "ref2_start", "cut1", "cut2", "ref1", "ref2"])
    .agg(pl.col("count").sum())
    .sink_parquet((args.data_dir.parent / "dataset.parquet").as_posix())
)

pds = load_dataset("parquet", data_files="/home/ljw/sdc1/SJLJH/dataset.parquet", num_proc=12)