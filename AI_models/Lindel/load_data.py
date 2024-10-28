import torch
import torch.nn.functional as F

outputs_train_ins = {
    "input": "input_ins",
    "count": "count_ins"
}
outputs_train_del = {
    "input": "input_del",
    "count": "count_del"
}
outputs_train_indel = {
    "input": "input_indel",
    "count": "count_indel"
}
outputs_test = {
    "input_indel": "input_indel",
    "count_indel": "count_indel",
    "input_ins": "input_ins",
    "count_ins": "count_ins",
    "input_del": "input_del",
    "count_del": "count_del"
}
outputs_inference = {
    "input_indel": "input_indel",
    "input_ins": "input_ins",
    "input_del": "input_del"
}

@torch.no_grad()
def data_collector(examples, Lindel_dlen, Lindel_mh_len, outputs):
    def onehotencoder(guide):
        guideVal = (torch.frombuffer(guide.encode(), dtype=torch.int8) % 5).clamp(0, 3).to(torch.int64)
        return torch.cat([
            F.one_hot(guideVal, num_classes=4).flatten(),
            F.one_hot(guideVal[:-1] + guideVal[1:] * 4, num_classes=16).flatten()
        ]).to(torch.float32)

    def get_feature(example):
        del_dim = (4 + 1 + 4 + Lindel_dlen - 1) * (Lindel_dlen - 1) // 2
        dstarts = torch.tensor(example["dstart"])
        del_lens = torch.tensor(example["del_len"])
        mh_lens = torch.tensor(example["mh_len"])
        features = del_dim * mh_lens + (Lindel_dlen - 1 + 4 + del_lens + 1 + 4) * (Lindel_dlen - del_lens - 1) // 2 + dstarts + del_lens + 1
        one_hot = torch.zeros((Lindel_mh_len + 1) * del_dim)
        one_hot[features.to(torch.int64)] = 1.0
        return one_hot

    results = dict()
    if "input_del" in outputs.values() or "input_indel" in outputs.values():
        results["input_indel"] = torch.stack([
            onehotencoder(example["ref"][example["cut"] - 17:example["cut"] + 3])
            for example in examples
        ])
    if "count_indel" in outputs.values():
        results["count_indel"] = torch.tensor([
            [sum(example["del_count"]), sum(example["ins_count"])]
            for example in examples
        ])
    if "input_del" in outputs.values():
        results["input_del"] = torch.cat([
            torch.stack([
                get_feature(example)
                for example in examples
            ]),
            results["input_indel"]
        ], dim=1)
    if "count_del" in outputs.values():
        results["count_del"] = torch.tensor([
            example["del_count"]
            for example in examples
        ])
    if "input_ins" in outputs.values():
        results["input_ins"] = torch.stack([
            onehotencoder(example["ref"][example["cut"] - 3:example["cut"] + 3])
            for example in examples
        ])
    if "count_ins" in outputs.values():
        results["count_ins"] = torch.tensor([
            example["ins_count"]
            for example in examples
        ])

    return {
        key: results[value]
        for key, value in outputs.items()
    }
