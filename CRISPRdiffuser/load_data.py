import torch
import torch.nn.functional as F
from .config import args, alphacode, ref1len, ref2len, diag_indices, logger, device
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
            F.one_hot(kmer1, num_classes=base_kmer).T.unsqueeze(1).tile(dims=(1, len(kmer2), 1)),
            F.one_hot(kmer2, num_classes=base_kmer).T.unsqueeze(2).tile(dims=(1, 1, len(kmer1)))
        ))
    )

def kmer2one_hot2D(kmer1, kmer2):
    # INPUT: tensor dtype=int64 shape=1D
    # OUTPUT: one hot tensor dtype=int64 shape=(base^2kmer_size, len(kmer2), len(kmer1))
    base_kmer = len(alphacode)**args.kmer_size
    return F.one_hot(kmer1.view(1, -1) + kmer2.view(-1, 1) * base_kmer, num_classes=base_kmer**2).transpose(2, 0, 1)

def cut2one_hot(cut1, cut2):
    # OUTPUT: one hot tensor dtype=int64 shape=(ref2len+1, ref1len+1)
    # The layer representing 2D cut point
    cut_layer = torch.zeros(ref2len + 1, ref1len + 1, dtype=torch.int64)
    cut_layer[cut2, cut1] = 1
    return cut_layer

def num2micro_homology(ref1num, ref2num, cut1, cut2):
    # OUTPUT: micro-homology matrix shape=(ref2len+1, ref1len+1) dtype=int64
    indices_num = len(diag_indices[0])
    mh_matrix = F.pad((ref1num[:cut1].view(1, -1) == ref2num[cut2:].view(-1, 1)).to(torch.int64), pad=(0,ref1len-cut1+1,cut2,1), value=0)
    rep_num = torch.cat((
        torch.tensor([-1]),
        torch.where(mh_matrix[diag_indices].diff())[0],
        torch.tensor([indices_num-1])
    )).diff()
    rep_val = rep_num.clone()
    rep_val[0::2] = 0
    rep_num[1::2] = rep_num[1::2] + 1
    rep_num[2::2] = rep_num[2::2] - 1
    rep_val = rep_val.repeat_interleave(rep_num)
    mh_matrix[diag_indices] = rep_val
    return mh_matrix

def split_train_valid_test(ds):
    # Divide ds's train split to valid and test splits. Both has proportion test_valid_ratio.
    ds = ds['train'].train_test_split(2*args.test_valid_ratio, shuffle=True, seed=args.seed) 
    ds_valid_test = ds['test'].train_test_split(test_size=0.5, shuffle=False)
    ds['valid'] = ds_valid_test.pop('train')
    ds['test'] = ds_valid_test.pop('test')
    return ds

def collect_train_data(examples):
    ref1nums = [DNA2num(example['ref1']) for example in examples]
    ref2nums = [DNA2num(example['ref2']) for example in examples]
    return {
        "ref1_end": torch.tensor([example['ref1_end'] for example in examples], device=device),
        "ref2_start": torch.tensor([example['ref2_start'] for example in examples], device=device),
        "condition": torch.stack([
            torch.cat((
                num2micro_homology(ref1nums[i], ref2nums[i], examples[i]['cut1'], examples[i]['cut2']).unsqueeze(0).clamp(0, args.max_micro_homology) / args.max_micro_homology,
                cut2one_hot(examples[i]['cut1'], examples[i]['cut2']).unsqueeze(0),
                kmer2one_hot1D(num2kmer(ref1nums[i]), num2kmer(ref2nums[i])) if not args.cross_reference else kmer2one_hot2D(num2kmer(ref1nums[i]), num2kmer(ref2nums[i]))
            ))
            for i in range(len(examples))
        ]).to(device),
        # this is the gradient weight
        "weight": torch.tensor([
            example['count'] ** (1 - args.importance_sampling_factor)
            for example in examples
        ], device=device)
    }

def collect_valid_data(examples):
    ref1nums = [DNA2num(example['ref1']) for example in examples]
    ref2nums = [DNA2num(example['ref2']) for example in examples]
    return {
        "condition": torch.stack([
            torch.cat((
                num2micro_homology(ref1nums[i], ref2nums[i], examples[i]['cut1'], examples[i]['cut2']).unsqueeze(0).clamp(0, args.max_micro_homology) / args.max_micro_homology,
                cut2one_hot(examples[i]['cut1'], examples[i]['cut2']).unsqueeze(0),
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
        'count': list(itertools.chain.from_iterable(examples['count']))
    }

def unlist_ref1_end_ref2_start_count(examples):
    return {
        'ref1': [ref1 for i, ref1 in enumerate(examples['ref1']) for _ in examples['ref1_end'][i]],
        'ref2': [ref2 for i, ref2 in enumerate(examples['ref2']) for _ in examples['ref2_start'][i]],
        'cut1': [cut1 for i, cut1 in enumerate(examples['cut1']) for _ in examples['ref1_end'][i]],
        'cut2': [cut2 for i, cut2 in enumerate(examples['cut2']) for _ in examples['ref2_start'][i]],
        'ref1_end': list(itertools.chain.from_iterable(examples['ref1_end'])),
        'ref2_start': list(itertools.chain.from_iterable(examples['ref2_start'])),
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
# Split after unlist_cuts but before unlist_ref1_end_ref2_start_count, so that for samples in valid and test sets, at least one of ref1/2 and cut1/2 is new.
ds = split_train_valid_test(ds)
ds['train'] = ds['train'].map(unlist_ref1_end_ref2_start_count, batched=True, desc="unlist ref1_end, ref2_start and count in dataset")

train_dataloader = DataLoader(ds["train"], batch_sampler=sampler_with_put_back(
        np.array(ds["train"]["count"]) ** args.importance_sampling_factor
    ), collate_fn=collect_train_data)
valid_dataloader = DataLoader(ds["valid"], collate_fn=collect_valid_data)

