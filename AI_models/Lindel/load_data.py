import torch
import torch.nn.functional as F

@torch.no_grad()
def data_collector(examples, Lindel_mh_len=4, model=None):
    def onehotencoder(guide):
        guideVal = (torch.frombuffer(guide.encode(), dtype=torch.int8) % 5).clamp(0, 3).to(torch.int64)
        return torch.cat([
            F.one_hot(guideVal, num_classes=4).flatten(),
            F.one_hot(guideVal[:-1] + guideVal[1:] * 4, num_classes=16).flatten()
        ]).to(torch.float32)

    def get_feature(example):
        Lindel_dlen = int(round((-7 + (49 + 4 * (8 + 2 * len(example["del_count"]))) ** 0.5) / 2))
        dstarts = torch.tensor(example["dstart"])
        del_lens = torch.tensor(example["del_len"])
        mh_lens = torch.tensor(example["mh_len"])
        features = len(example["del_count"]) * mh_lens + (Lindel_dlen - 1 + 4 + del_lens + 1 + 4) * (Lindel_dlen - del_lens - 1) // 2 + dstarts + del_lens + 1
        one_hot = torch.zeros((Lindel_mh_len + 1) * len(example["del_count"]))
        one_hot[features] = 1.0
        return one_hot

    if model != "ins":
        input_indel = torch.stack([
            onehotencoder(example["ref"][example["cut"] - 17:example["cut"] + 3])
            for example in examples
        ])
    if model != "ins" and model != "del":
        count_indel = torch.tensor([
            [sum(example["del_count"]), sum(example["ins_count"])]
            for example in examples
        ])
    if model != "ins" and model != "indel":
        input_del = torch.cat([
            torch.stack([
                get_feature(example)
                for example in examples
            ]),
            input_indel
        ], dim=1)
        count_del = torch.tensor([
            example["del_count"]
            for example in examples
        ])
    if model != "del" and model != "indel":
        input_ins = torch.stack([
            onehotencoder(example["ref"][example["cut"] - 3:example["cut"] + 3])
            for example in examples
        ])
        count_ins = torch.tensor([
            example["ins_count"]
            for example in examples
        ])
    if model == "indel":
        return {
            "input": input_indel,
            "count": count_indel
        }
    if model == "ins":
        return {
            "input": input_ins,
            "count": count_ins
        }
    if model == "del":
        return {
            "input": input_del,
            "count": count_del
        }
    return {
        "ref": [
            example["ref"]
            for example in examples
        ],
        "cut": [
            example["cut"]
            for example in examples
        ],
        "input_indel": input_indel,
        "count_indel": count_indel,
        "input_ins": input_ins,
        "count_ins": count_ins,
        "input_del": input_del,
        "count_del": count_del
    }