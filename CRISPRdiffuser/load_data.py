import torch
import torch.nn.functional as F
from .config import args, alphacode, reflen, diag_indices, logger
from torch.utils.data import DataLoader

def DNA2num(seq):
    # INPUT: string
    # OUTPUT: tensor dtype=uint8 shape=(len(seq),)
    global alphacode
    seq = torch.frombuffer(seq.encode(), dtype=torch.uint8)
    for i in range(len(alphacode)):
        seq[seq == alphacode[i]] = i
    return seq

def num2kmer(num):
    # INPUT: tensor dtype=uint8 shape=(len(num),) valueset=(0,1,2,...,base-1)
    # OUTPUT: tensor dtype=int64 shape=(len(num)+1,)
    # Pad kmer_size (base-1)s at the tail of DNA. Apply convolution kernel [base^0, base^1, base^2, ..., base^(kmer_size-1)]
    global alphacode, args
    base = len(alphacode)
    DNACopy = torch.cat([num, torch.full((args.kmer_size,), base-1, dtype=torch.int64)]).view(1, 1, -1)
    return torch.ravel(F.conv1d(
        input=DNACopy,
        weight=torch.logspace(start=0, end=args.kmer_size-1, steps=args.kmer_size, base=base, dtype=torch.int64).view(1, 1, -1)
    ))

def kmer2one_hot1D(kmer1, kmer2):
    # INPUT: tensor dtype=int64 shape=1D
    # OUTPUT: one hot tensor dtype=int64 shape=(len(kmer2), len(kmer1), 2*base^kmer_size)
    global alphacode, args
    base_kmer = len(alphacode)**args.kmer_size
    return(torch.cat((
        F.one_hot(kmer1, num_classes=base_kmer).unsqueeze(0).tile(dims=(len(kmer2), 1, 1)),
        F.one_hot(kmer2, num_classes=base_kmer).unsqueeze(1).tile(dims=(1, len(kmer1), 1))
    ), dim=2))

def kmer2one_hot2D(kmer1, kmer2):
    # INPUT: tensor dtype=int64 shape=1D
    # OUTPUT: one hot tensor dtype=int64 shape=(len(kmer2), len(kmer1), base^2kmer_size)
    global alphacode, args
    base_kmer = len(alphacode)**args.kmer_size
    return F.one_hot(kmer1.view(1, -1) + kmer2.view(-1, 1) * base_kmer, num_classes=base_kmer**2)

def cut2one_hot(cut1, cut2):
    # OUTPUT: one hot tensor dtype=int64 shape=(reflen+1, reflen+1, 2)
    global reflen
    return torch.stack((
        torch.cat((
            torch.ones(reflen + 1, cut1, dtype=torch.int64),
            torch.zeros(reflen + 1, reflen + 1 - cut1, dtype=torch.int64),
        ), dim=1),
        torch.cat((
            torch.zeros(cut2, reflen + 1, dtype=torch.int64),
            torch.zeros(reflen + 1 - cut2, reflen + 1, dtype=torch.int64),
        ), dim=0)
    ), dim=2)

def num2micro_homology(ref1num, ref2num):
    # OUTPUT: micro-homology matrix shape=(reflen+1, reflen+1) dtype=int64
    global diag_indices
    indices_num = len(diag_indices[0])
    mh_matrix = F.pad((ref1num.view(1, -1) == ref2num.view(-1, 1)).to(torch.int64), pad=(0,1,0,1), value=0)
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
    global args
    ds = ds['train'].train_test_split(2*args.test_valid_ratio, shuffle=True, seed=args.seed) 
    ds_valid_test = ds['test'].train_test_split(test_size=0.5, shuffle=False)
    ds['valid'] = ds_valid_test.pop('train')
    ds['test'] = ds_valid_test.pop('test')
    return ds

def collect_model_input(examples):
    ref1nums = [DNA2num(example['ref1']) for example in examples]
    ref2nums = [DNA2num(example['ref2']) for example in examples]
    return {
        "condition": torch.stack([
            torch.cat((
                num2micro_homology(ref1nums[i], ref2nums[i]).view(len(ref2nums[i])+1, len(ref1nums[i])+1, 1),
                cut2one_hot(examples[i]['cut1'], examples[i]['cut2']),
                kmer2one_hot1D(num2kmer(ref1nums[i]), num2kmer(ref2nums[i]))
            ), dim=2)
            for i in range(len(examples))
        ]),
        "ref1_end": torch.nn.utils.rnn.pad_sequence([
            torch.tensor(example['ref1_end'])
            for example in examples
        ], batch_first=True, padding_value=0),
        "ref2_start": torch.nn.utils.rnn.pad_sequence([
            torch.tensor(example['ref2_start'])
            for example in examples
        ], batch_first=True, padding_value=0),
        "count": torch.nn.utils.rnn.pad_sequence([
            torch.tensor(example['count'])
            for example in examples
        ], batch_first=True, padding_value=0),
    }

from datasets import load_dataset, Features, Value, Sequence
alg_features = Features({
    'cut1': Value('int16'),
    'cut2': Value('int16'),
    'ref1': Value('string'),
    'ref2': Value('string'),
    'ref1_end': Sequence(Value('int16')),
    'ref2_start': Sequence(Value('int16')),
    'count': Sequence(Value('uint64'))
})

ds = load_dataset("json", data_files=args.data_file.as_posix(), num_proc=12, features=alg_features)
for example in ds['train']:
    try:
        assert len(example['ref1']) == reflen, "references are not of the same length"
    except Exception as err:
        logger.exception(str(err))
        raise
ds = split_train_valid_test(ds)

train_dataloader = DataLoader(
    ds["train"], shuffle=True, batch_size=8, collate_fn=collect_model_input
)
num_training_steps = len(train_dataloader) * args.num_epochs
valid_dataloader = DataLoader(
    ds["valid"], shuffle=False, collate_fn=collect_model_input
)