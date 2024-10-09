import torch

def kmer2int(kmer):
    return (
        (torch.frombuffer(kmer.encode(), dtype=torch.int8) % 5) * (5 ** torch.arange(len(kmer)))
    ).sum().item()

@torch.no_grad()
def data_collector(examples, epsilon=1e-6, mode="train_deletion"):
    DELLEN_LIMIT = len(examples[0]['mhless_count']) + 1
    max_mh_genotype = max([len(example['mh_mh_len']) for example in examples])
    mh_inputs = torch.tensor([
        [[mh_mh_len, mh_gc_frac] for mh_mh_len, mh_gc_frac in zip(example['mh_mh_len'], example['mh_gc_frac'])] + [[0.0, 0.0]] * (max_mh_genotype - len(example['mh_mh_len']))
        for example in examples
    ], dtype=torch.float32)
    mh_del_lens = torch.tensor([
        example['mh_del_len'] + [DELLEN_LIMIT] * (max_mh_genotype - len(example['mh_del_len']))
        for example in examples
    ], dtype=torch.int64)
    if mode=="train_deletion":
        return {
            "mh_input": mh_inputs,
            "mh_del_len": mh_del_lens,
            "genotype_count": torch.tensor([
                example['mh_count'] + [0] * (max_mh_genotype - len(example['mh_count'])) + example['mhless_count']
                for example in examples
            ], dtype=torch.float32),
            "total_del_len_count": torch.stack([
                torch.tensor(example['mhless_count'], dtype=torch.float32).scatter_add(
                    dim=0,
                    index=torch.tensor(example['mh_del_len'], dtype=torch.int64) - 1,
                    src=torch.tensor(example['mh_count'], dtype=torch.float32)
                )
                for example in examples
            ])
        }
    else:
        onebp_features = torch.stack([
            torch.eye(4)[
                (torch.frombuffer(example['ref'][example['cut'] - 1:example['cut'] + 1].encode(), dtype=torch.int8) % 5).clamp(0, 3).to(torch.int64)
            ].flatten()
            for example in examples
        ])
        m654s = torch.tensor([
            kmer2int(example['ref'][example['cut'] - 3:example['cut']])
            for example in examples
        ], dtype=torch.int64)
        insert_probabilities = [
            sum(example['insert_1bp']) / (sum(example['insert_1bp']) + sum(example['mh_count']) + sum(example['mhless_count']) + epsilon)
            for example in examples
        ]
        insert_1bps = [
            example['insert_1bp']
            for example in examples
        ]
        if mode=="train_insertion":
            return {
                "mh_input": mh_inputs,
                "mh_del_len": mh_del_lens,
                "onebp_feature": onebp_features,
                "m654": m654s,
                "insert_probability": insert_probabilities,
                "insert_1bp": torch.tensor(insert_1bps, dtype=torch.float32)
            }
        else:
            return {
                "mh_input": mh_inputs,
                "mh_del_len": mh_del_lens,
                "onebp_feature": onebp_features,
                "m654": m654s,
                "ref": [example["ref"] for example in examples],
                "cut": [example["cut"] for example in examples],
                "mh_gt_pos": [example["mh_gt_pos"] for example in examples],
                "mh_count": [example['mh_count'] for example in examples],
                "mhless_count": [example['mhless_count'] for example in examples],
                "total_del_len_count": [
                    torch.tensor(example['mhless_count'], dtype=torch.float32).scatter_add(
                        dim=0,
                        index=torch.tensor(example['mh_del_len'], dtype=torch.int64) - 1,
                        src=torch.tensor(example['mh_count'], dtype=torch.float32)
                    ).to(torch.int64).tolist()
                    for example in examples
                ],
                "insert_probability": insert_probabilities,
                "insert_1bp": insert_1bps
            }