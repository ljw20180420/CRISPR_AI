import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import os
sys.path.append(os.getcwd())
from config import args

def data_collector(examples, noise_scheduler, stationary_sampler1, stationary_sampler2):
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

    def get_observation(example):
        observation = torch.tensor(ref2len + 1, ref1len + 1)
        observation[example['ob_ref2'], example['ob_ref1']] = example['ob_val']
        return observation

    batch_size, ref1len, ref2len = len(examples), len(examples[0]['ref1']), len(examples[0]['ref2'])
    base = len("ACGTN")
    conditions = torch.stack([
        get_condition(example)
        for example in examples
    ])
    observations = torch.stack([
        get_observation(example)
        for example in examples
    ])
    x_cross0 = Categorical(probs=observations.view(batch_size, -1)).sample()
    x20 = x_cross0 // (ref1len + 1)
    x10 = x_cross0 % (ref1len + 1)
    t = noise_scheduler.step_to_time(torch.rand(batch_size, device=device) * args.noise_timesteps)
    x1t, x2t = noise_scheduler.add_noise(x10, x20, t, stationary_sampler1, stationary_sampler2)
    return {
        "x1t": x1t,
        "x2t": x2t,
        "t": t,
        "condition": conditions,
        "observation": observations
    }
