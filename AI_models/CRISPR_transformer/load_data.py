import torch
import numpy as np
from ..config import get_config

args = get_config(config_file="config_CRISPR_transformer.ini")

outputs_train = ["refcode", "observation"]
outputs_test = ["refcode", "observation"]
outputs_inference = ["refcode"]

@torch.no_grad()
def data_collector(examples, outputs):
    def get_observation(example):
        observation = torch.zeros(ref2len + 1, ref1len + 1)
        observation[example['ob_ref2'], example['ob_ref1']] = torch.tensor(example['ob_val'], dtype=observation.dtype)
        return observation

    batch_size, ref1len, ref2len = len(examples), len(examples[0]['ref1']), len(examples[0]['ref2'])
    results = dict()
    if "refcode" in outputs:
        results["refcode"] = torch.stack([
            torch.from_numpy(
                np.frombuffer(
                    (example['ref1'] + example['ref2']).encode(),
                    dtype=np.int8
                ) % 5
            ).clamp_max(3).to(torch.int64)
            for example in examples
        ])

    if "observation" in outputs:
        results["observation"] = torch.stack([
            get_observation(example)
            for example in examples
        ])

    return results
