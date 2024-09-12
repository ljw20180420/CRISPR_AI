import itertools
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
import tqdm
from Predictor import gen_indel, onehotencoder, create_feature_array, features
from config import args, device

@torch.no_grad()
def data_collector(examples, Lindel_mh_len=4, model="indel"):
    def onehotencoder(guide):
        guideVal = (torch.frombuffer(guide.encode(), torch.int8) % 5).clamp(0, 3)
        return torch.cat([
            F.one_hot(guideVal, num_classes=4).flatten(),
            F.one_hot(guideVal[:-1] + guideVal[1:] * 4, num_classes=16).flatten()
        ])

    def get_feature(example):
        Lindel_dlen = (-7 + (49 + 4 * (8 + 2 * len(example["del_count"]))) ** 0.5) / 2
        dstarts = torch.tensor(example["dstart"])
        del_lens = torch.tensor(example["del_len"])
        mh_lens = torch.tensor(example["mh_len"])
        features = len(example["del_count"]) * mh_lens + (Lindel_dlen - 1 + 4 + del_lens + 1 + 4) * (Lindel_dlen - del_lens - 1) / 2 + dstarts + del_lens + 1
        one_hot = torch.zeros((Lindel_mh_len + 1) * len(example["del_count"]), torch.int64)
        one_hot[features] = 1
        return one_hot

    if model in ["indel", "del"]:
        input_indels = torch.stack([
            onehotencoder(example["ref"][example["cut"] - 17:example["cut"] + 3])
            for example in examples
        ]),
        if model == "indel":
            return {
                "input_indel": input_indels,
                "indel_count": torch.tensor([
                    [sum(example["del_count"]), sum(example["ins_count"])]
                    for example in examples
                ])        
            }
        return {
            "input_del": torch.cat([
                torch.stack([
                    get_feature(example)
                    for example in examples
                ]),
                input_indels
            ], dim=1),
            "del_count": torch.tensor([
                example["del_count"]
                for example in examples
            ])
        }
    return {
        "input_ins": torch.stack([
            onehotencoder(example["ref"][example["cut"] - 3:example["cut"] + 3])
            for example in examples
        ]),
        "ins_count": torch.tensor([
            example["ins_count"]
            for example in examples
        ])
    }