import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from config import args


def data_collector(examples):
    def get_condition(example):
        mh_matrix = torch.zeros(ref2len + 1, ref1len + 1)
        mh_matrix[example['mh_ref2'], example['mh_ref1']] = example['mh_val']
        mh_matrix = mh_matrix.clamp(0, args.max_micro_homology) / max_micro_homology
        one_hot_cut = torch.zeros(ref2len + 1, ref1len + 1)
        one_hot_cut[example['cut2'], example['cut1']] = 1.0
        one_hot_ref1 = F.one_hot(
            (torch.frombuffer((example['ref1'] + "N").encode(), dtype=torch.int8) % base).to(torch.int64),
            num_classes=base
        ).T[:, None, :].expand(-1, ref2len + 1, -1)
        one_hot_ref2 = F.one_hot(
            (torch.frombuffer((example['ref2'] + "N").encode(), dtype=torch.int8) % base).to(torch.int64),
            num_classes=base
        ).T[:, :, None].expand(-1, -1, ref1len + 1)
        return torch.cat([
            mh_matrix[None, :, :],
            one_hot_cut[None, :, :],
            one_hot_ref1,
            one_hot_ref2
        ])

    observation = 

    batch_size, ref1len, ref2len = len(examples), len(examples[0]['ref1']), len(examples[0]['ref2'])
    base = len("ACGTN")
    conditions = torch.tensor(batch_size, 12, ref2len + 1, ref1len + 1)
    observations = torch.tensor(batch_size, ref2len + 1, ref1len + 1)
    

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

