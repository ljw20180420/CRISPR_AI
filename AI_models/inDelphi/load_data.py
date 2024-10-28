import torch

def kmer2int(kmer):
    return (
        (torch.frombuffer(kmer.encode(), dtype=torch.int8) % 5) * (5 ** torch.arange(len(kmer)))
    ).sum().item()

outputs_train_deletion = ["mh_input", "mh_del_len", "genotype_count", "total_del_len_count"]
outputs_train_insertion = ["mh_input", "mh_del_len", "onebp_feature", "m654", "insert_probability", "insert_1bp"]
outputs_test = ["mh_input", "mh_del_len", "onebp_feature", "m654", "total_del_len_count", "insert_probability"]
outputs_inference = ["mh_input", "mh_del_len", "onebp_feature", "m654"]

@torch.no_grad()
def data_collector(examples, DELLEN_LIMIT, outputs, epsilon=1e-6):
    max_mh_genotype = max([len(example['mh_mh_len']) for example in examples])
    results = dict()
    if "mh_input" in outputs:
        results["mh_input"] = torch.tensor([
            [[mh_mh_len, mh_gc_frac] for mh_mh_len, mh_gc_frac in zip(example['mh_mh_len'], example['mh_gc_frac'])] + [[0.0, 0.0]] * (max_mh_genotype - len(example['mh_mh_len']))
            for example in examples
        ], dtype=torch.float32)
    if "mh_del_len" in outputs:
        results["mh_del_len"] = torch.tensor([
            example['mh_del_len'] + [DELLEN_LIMIT] * (max_mh_genotype - len(example['mh_del_len']))
            for example in examples
        ], dtype=torch.int64)
    if "genotype_count" in outputs:
        results["genotype_count"] = torch.tensor([
            example['mh_count'] + [0] * (max_mh_genotype - len(example['mh_count'])) + example['mhless_count']
            for example in examples
        ], dtype=torch.float32)
    if "total_del_len_count" in outputs:
        results["total_del_len_count"] = torch.stack([
            torch.tensor(example['mhless_count'], dtype=torch.float32).scatter_add(
                dim=0,
                index=torch.tensor(example['mh_del_len'], dtype=torch.int64) - 1,
                src=torch.tensor(example['mh_count'], dtype=torch.float32)
            )
            for example in examples
        ])
    if "onebp_feature" in outputs:
        results["onebp_feature"] = torch.stack([
            torch.eye(4)[
                (torch.frombuffer(example['ref'][example['cut'] - 1:example['cut'] + 1].encode(), dtype=torch.int8) % 5).clamp(0, 3).to(torch.int64)
            ].flatten()
            for example in examples
        ])
    if "m654" in outputs:
        results["m654"] = torch.tensor([
            kmer2int(example['ref'][example['cut'] - 3:example['cut']])
            for example in examples
        ], dtype=torch.int64)
    if "insert_probability" in outputs:
        results["insert_probability"] = [
            sum(example['insert_1bp']) / (sum(example['insert_1bp']) + sum(example['mh_count']) + sum(example['mhless_count']) + epsilon)
            for example in examples
        ]
    if "insert_1bp" in outputs:
        results["insert_1bp"] = torch.tensor(
            [
                example['insert_1bp']
                for example in examples
            ],
            dtype=torch.float32
        )
    return results