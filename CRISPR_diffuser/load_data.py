import torch
import torch.nn.functional as F
from config import args, alphacode, ref1len, ref2len, diag_indices, logger, device
from torch.utils.data import DataLoader
import itertools
import random
import numpy as np
import tqdm

def DNA2num(seq):
    # INPUT: string
    # OUTPUT: tensor dtype=uint8 shape=(len(seq),)
    seq = torch.frombuffer(seq.encode(), dtype=torch.uint8)
    cover = torch.full((len(seq),), False)
    for i in range(len(alphacode)):
        mask = seq == alphacode[i]
        seq[mask] = i
        cover = cover | mask
    assert torch.all(cover), "find unknow alphabet"
    return seq

def num2kmer(num):
    # INPUT: tensor dtype=uint8 shape=(len(num),) valueset=(0,1,2,...,base-1)
    # OUTPUT: tensor dtype=int64 shape=(len(num)+1,)
    # Pad kmer_size (base-1)s at the tail of DNA. Apply convolution kernel [base^0, base^1, base^2, ..., base^(kmer_size-1)]
    base = len(alphacode)
    DNACopy = torch.cat([num, torch.full((args.kmer_size,), base-1, dtype=torch.int64)]).view(1, 1, -1)
    return torch.ravel(F.conv1d(
        input=DNACopy,
        weight=torch.logspace(start=0, end=args.kmer_size-1, steps=args.kmer_size, base=base, dtype=torch.int64).view(1, 1, -1)
    ))

def kmer2one_hot1D(kmer1, kmer2):
    # INPUT: tensor dtype=int64 shape=1D
    # OUTPUT: one hot tensor dtype=int64 shape=(2*base^kmer_size, len(kmer2), len(kmer1))
    base_kmer = len(alphacode)**args.kmer_size
    return(
        torch.cat((
            F.one_hot(kmer1, num_classes=base_kmer).T[:, None, :].tile(dims=(1, len(kmer2), 1)),
            F.one_hot(kmer2, num_classes=base_kmer).T[:, :, None].tile(dims=(1, 1, len(kmer1)))
        ))
    )

def cut2one_hot(cut1, cut2):
    # OUTPUT: one hot tensor dtype=int64 shape=(ref2len+1, ref1len+1)
    # The layer representing 2D cut point
    cut_layer = torch.zeros(ref2len + 1, ref1len + 1, dtype=torch.int64)
    cut_layer[cut2, cut1] = 1
    return cut_layer

def split_train_valid_test(ds):
    # Divide ds's train split to valid and test splits. Both has proportion test_valid_ratio.
    ds = ds['train'].train_test_split(2*args.test_valid_ratio, shuffle=True, seed=args.seed) 
    ds_valid_test = ds['test'].train_test_split(test_size=0.5, shuffle=False)
    ds['valid'] = ds_valid_test.pop('train')
    ds['test'] = ds_valid_test.pop('test')
    return ds

def collect_data(examples):
    ref1nums = [DNA2num(example['ref1']) for example in examples]
    ref2nums = [DNA2num(example['ref2']) for example in examples]
    return {
        "condition": torch.stack([
            torch.cat((
                num2micro_homology(ref1nums[i], ref2nums[i], examples[i]['cut1'], examples[i]['cut2'])[None, :, :].clamp(0, args.max_micro_homology) / args.max_micro_homology,
                cut2one_hot(examples[i]['cut1'], examples[i]['cut2'])[None, :, :],
                kmer2one_hot1D(num2kmer(ref1nums[i]), num2kmer(ref2nums[i])) if not args.cross_reference else kmer2one_hot2D(num2kmer(ref1nums[i]), num2kmer(ref2nums[i]))
            ))
            for i in range(len(examples))
        ]).to(device),
        "observation": torch.stack([
            torch.sparse_coo_tensor([example['ref2_start'], example['ref1_end']], example['count'], (ref2len + 1, ref1len + 1)).to_dense()
            for example in examples
        ]).to(device)
    }

def sampler_without_put_back(counts):
    # sample index start from 0 according to counts without put-back
    counts = counts.copy()
    random.seed(args.seed)
    remain = sum(counts)
    population = list(range(len(counts)))
    while True:
        indices = random.sample(population, counts = counts, k= min(remain, args.batch_size))
        yield indices
        remain = remain - args.batch_size
        if remain <= 0:
            break
        for i in indices:
            counts[i] = counts[i] - 1

def sampler_with_put_back(weights):
    # sample index start from 0 according to weights with put-back
    rng = np.random.default_rng(args.seed)
    pvals = weights / sum(weights)
    while True:
        yield rng.choice(len(pvals), args.batch_size, p=pvals)

def unlist_cuts(examples):
    return {
        'ref1': [ref1 for i, ref1 in enumerate(examples['ref1']) for _ in examples['cut1'][i]],
        'ref2': [ref2 for i, ref2 in enumerate(examples['ref2']) for _ in examples['cut2'][i]],
        'cut1': list(itertools.chain.from_iterable(examples['cut1'])),
        'cut2': list(itertools.chain.from_iterable(examples['cut2'])),
        'ref1_end': list(itertools.chain.from_iterable(examples['ref1_end'])),
        'ref2_start': list(itertools.chain.from_iterable(examples['ref2_start'])),
        'random_insert': list(itertools.chain.from_iterable(examples['random_insert'])),
        'count': list(itertools.chain.from_iterable(examples['count']))
    }

from datasets import load_dataset, Features, Value, Sequence
alg_features = Features({
    'ref1': Value('string'),
    'ref2': Value('string'),
    'cut1': Sequence(Value('int16')),
    'cut2': Sequence(Value('int16')),
    'ref1_end': Sequence(Sequence(Value('int16'))),
    'ref2_start': Sequence(Sequence(Value('int16'))),
    'random_insert': Sequence(Sequence(Value('string'))),
    'count': Sequence(Sequence(Value('uint64')))
})

ds = load_dataset("json", data_files=args.data_file.as_posix(), num_proc=12, features=alg_features)
# for example in tqdm.tqdm(ds['train'], desc="checking reference lengths"):
#     try:
#         assert len(example['ref1']) == ref1len and len(example['ref2']) == ref2len, "references are not of the same length"
#     except Exception as err:
#         logger.exception(str(err))
#         raise
ds = ds.map(unlist_cuts, batched=True, desc="unlist cuts in dataset")
ds = split_train_valid_test(ds)

train_dataloader = DataLoader(dataset=ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collect_data)
valid_dataloader = DataLoader(dataset=ds["valid"], batch_size=args.batch_size, collate_fn=collect_data)

