import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from ..config import get_config

args = get_config(config_file="config_CRISPR_diffuser.ini")

outputs_train = ["x1t_x2t_t", "condition", "observation"]
outputs_test = ["condition", "observation"]
outputs_inference = ["condition"]

@torch.no_grad()
def data_collector(examples, noise_scheduler, stationary_sampler1, stationary_sampler2, outputs):
    def get_condition(example):
        mh_matrix = torch.zeros(ref2len + 1, ref1len + 1)
        mh_matrix[example['mh_ref2'], example['mh_ref1']] = torch.tensor(example['mh_val'], dtype=mh_matrix.dtype)
        mh_matrix = mh_matrix.clamp(0, args.max_micro_homology) / args.max_micro_homology
        one_hot_cut = torch.zeros(ref2len + 1, ref1len + 1)
        one_hot_cut[example['cut2'], example['cut1']] = 1.0
        one_hot_ref1 = F.one_hot(
            torch.from_numpy(
                (np.frombuffer((example['ref1'] + "N").encode(), dtype=np.int8) % 5).clip(max=3).astype(np.int64)
            ),
            num_classes=4
        ).T[:, None, :].expand(-1, ref2len + 1, -1)
        one_hot_ref2 = F.one_hot(
            torch.from_numpy(
                (np.frombuffer((example['ref2'] + "N").encode(), dtype=np.int8) % 5).clip(max=3).astype(np.int64)
            ),
            num_classes=4
        ).T[:, :, None].expand(-1, -1, ref1len + 1)
        return torch.cat([
            mh_matrix[None, :, :],
            one_hot_cut[None, :, :],
            one_hot_ref1,
            one_hot_ref2
        ])

    def get_observation(example):
        observation = torch.zeros(ref2len + 1, ref1len + 1)
        observation[example['ob_ref2'], example['ob_ref1']] = torch.tensor(example['ob_val'], dtype=observation.dtype)
        return observation

    batch_size, ref1len, ref2len = len(examples), len(examples[0]['ref1']), len(examples[0]['ref2'])
    results = dict()
    if "condition" in outputs:
        results["condition"] = torch.stack([
            get_condition(example)
            for example in examples
        ])
    if "x1t_x2t_t" in outputs or "observation" in outputs:
        results["observation"] = torch.stack([
            get_observation(example)
            for example in examples
        ])
    if "x1t_x2t_t" in outputs:
        x_cross0 = Categorical(probs=results["observation"].view(batch_size, -1)).sample()
        x20 = x_cross0 // (ref1len + 1)
        x10 = x_cross0 % (ref1len + 1)
        t = noise_scheduler.step_to_time(torch.rand(batch_size) * args.noise_timesteps)
        x1t, x2t = noise_scheduler.add_noise(x10, x20, t, stationary_sampler1, stationary_sampler2)
        results["x1t_x2t_t"] = {
            "x1t": x1t,
            "x2t": x2t,
            "t": t,
        }
    return results
