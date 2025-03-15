import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import torch.nn.functional as F
from datasets import load_dataset, Split
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from subprocess import Popen, PIPE
import pandas as pd
from ..config import get_config

args = get_config()

base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

zero_logit = -100


def load_ds_ob(data_name):
    return load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_CRISPR_diffuser",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )


def black_list_likelihood(observation, logit, black_list):
    observation = observation.clone()
    observation[*black_list] = 0
    logit = logit.clone()
    logit[*black_list] = zero_logit
    return (
        (
            observation.flatten()
            * F.log_softmax(logit.flatten(), dim=0).to(observation.device)
        )
        .sum()
        .cpu()
        .item()
    )


def detect_templated_insert(left, right, ins_seq, ref1, ref2):
    for r in range(len(ins_seq) + 1):
        if r == len(ins_seq) or ins_seq[len(ins_seq) - r - 1] != ref2[right - r - 1]:
            break
    for l in range(len(ins_seq) - r + 1):
        if l == len(ins_seq) - r or ins_seq[l] != ref1[left + l]:
            break
    return left + l, right - r


def inDelphi_logit(
    ref: str,
    mh_gt_pos: List,
    mh_del_len: List,
    mh_mh_len: List,
    mh_weight: torch.Tensor,
    mhless_weight: torch.Tensor,
    pre_insert_probability: float,
    pre_insert_1bp: np.array,
):
    pre_template_insert_probability = (
        pre_insert_1bp[base_map[ref[args.cut1 - 1]]] * pre_insert_probability
    )
    if ref[args.cut1] != ref[args.cut1 - 1]:
        pre_template_insert_probability += (
            pre_insert_1bp[base_map[ref[args.cut1]]] * pre_insert_probability
        )
    pre_delete_probability = F.normalize(
        torch.cat([mh_weight, mhless_weight]), p=1.0, dim=0
    ) * (1 - pre_template_insert_probability)
    mh_weight = pre_delete_probability[: len(mh_weight)]
    mhless_weight = pre_delete_probability[-len(mhless_weight) :]

    mh_gt_pos = torch.tensor(mh_gt_pos)
    mh_del_len = torch.tensor(mh_del_len)
    mh_mh_len_p1 = torch.tensor(mh_mh_len) + 1
    ref2start = mh_gt_pos + args.ref2len - len(ref)
    ref1end = mh_gt_pos - mh_del_len
    ref2start = ref2start.repeat_interleave(mh_mh_len_p1)
    ref1end = ref1end.repeat_interleave(mh_mh_len_p1)
    mh_shift = torch.cat([torch.arange(mh_len_p1) for mh_len_p1 in mh_mh_len_p1])
    ref2start -= mh_shift
    ref1end -= mh_shift

    pre_probability = torch.zeros(args.ref2len + 1, args.ref1len + 1)
    pre_probability[ref2start, ref1end] += (mh_weight / mh_mh_len_p1).repeat_interleave(
        mh_mh_len_p1
    )
    pre_probability[inDelphi_ref2start_mhless, inDelphi_ref1end_mhless] += (
        mhless_weight / inDelphi_del_len_p1
    ).repeat_interleave(inDelphi_del_len_p1)
    pre_probability[args.cut2 - 1, args.cut1] = (
        pre_insert_1bp[base_map[ref[args.cut1 - 1]]] * pre_insert_probability
    )
    if ref[args.cut1] != ref[args.cut1 - 1]:
        pre_probability[args.cut2, args.cut1 + 1] = (
            pre_insert_1bp[base_map[ref[args.cut1]]] * pre_insert_probability
        )
    return pre_probability.log().clamp_min(zero_logit)


inDelphi_del_len_p1 = torch.arange(2, args.DELLEN_LIMIT + 1)
inDelphi_ref2start_mhless = (
    torch.cat([torch.arange(del_len_p1) for del_len_p1 in inDelphi_del_len_p1])
    + args.cut2
)
inDelphi_ref1end_mhless = args.cut1 - torch.cat(
    [torch.arange(del_len_p1 - 1, -1, -1) for del_len_p1 in inDelphi_del_len_p1]
)


@torch.no_grad()
def inDelphi_benchmark(data_name: str, black_list: List):
    from ..inDelphi.model import inDelphiModel, inDelphiConfig
    from ..inDelphi.pipeline import inDelphiPipeline
    from ..inDelphi.load_data import data_collector, outputs_test

    inDelphi_model = inDelphiModel.from_pretrained(
        args.output_dir
        / inDelphiConfig.model_type
        / f"{data_name}_{inDelphiConfig.model_type}"
    )

    with open(
        args.output_dir
        / inDelphiConfig.model_type
        / f"{data_name}_{inDelphiConfig.model_type}"
        / "insertion_model.pkl",
        "rb",
    ) as fd:
        onebp_features, insert_probabilities, m654 = pickle.load(fd)
    pipe = inDelphiPipeline(inDelphi_model, onebp_features, insert_probabilities, m654)
    pipe.inDelphi_model.to(args.device)

    ds_ob = load_ds_ob(data_name)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{inDelphiConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        DELLEN_LIMIT=args.DELLEN_LIMIT,
    )

    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(
            examples, args.DELLEN_LIMIT, outputs_test + ["genotype_count"]
        ),
    )

    (
        likelihood,
        genotype_pearson,
        total_del_len_pearson,
        mhless_pearson,
        mhless_count_acc,
        total_del_len_count_acc,
    ) = (
        [],
        [],
        [],
        [],
        torch.zeros(args.DELLEN_LIMIT - 1, dtype=torch.int64),
        torch.zeros(args.DELLEN_LIMIT - 1, dtype=torch.int64),
    )
    for examples, batch, ob_examples in tqdm(
        zip(
            ds.iter(batch_size=args.batch_size),
            test_dataloader,
            ds_ob.iter(batch_size=args.batch_size),
        ),
        desc="inDelphi",
        total=len(ds_ob) / args.batch_size,
    ):
        mhless_count = torch.tensor(examples["mhless_count"], dtype=torch.int64)
        mhless_count_acc += mhless_count.sum(dim=0)
        total_del_len_count_acc += (
            batch["total_del_len_count"].to(torch.int64).sum(dim=0)
        )

        output = pipe(batch)
        max_mh_len = batch["genotype_count"].shape[1] - output["mhless_weight"].shape[0]
        mh_weight = torch.stack(
            [
                torch.cat([mw, torch.zeros(max_mh_len - len(mw), device=mw.device)])
                for mw in output["mh_weight"]
            ]
        )
        batch_size = mh_weight.shape[0]
        genotype_pearson.extend(
            (
                F.normalize(
                    torch.cat(
                        (mh_weight, output["mhless_weight"].expand(batch_size, -1)),
                        dim=1,
                    ),
                    p=2.0,
                    dim=1,
                )
                * F.normalize(batch["genotype_count"], p=2.0, dim=1).to(
                    mh_weight.device
                )
            )
            .sum(dim=1)
            .tolist()
        )
        total_del_len_pearson.extend(
            (
                F.normalize(output["total_del_len_weight"], p=2.0, dim=1)
                * F.normalize(batch["total_del_len_count"], p=2.0, dim=1).to(
                    output["total_del_len_weight"].device
                )
            )
            .sum(dim=1)
            .tolist()
        )
        mhless_pearson.extend(
            (
                F.normalize(output["mhless_weight"], p=2.0, dim=0)
                * F.normalize(mhless_count.to(torch.float32), p=2.0, dim=1).to(
                    output["mhless_weight"].device
                )
            )
            .sum(dim=1)
            .tolist()
        )

        for i in range(len(examples["ref"])):
            observation = torch.zeros(args.ref2len + 1, args.ref1len + 1)
            observation[ob_examples["ob_ref2"][i], ob_examples["ob_ref1"][i]] = (
                torch.tensor(ob_examples["ob_val"][i], dtype=observation.dtype)
            )

            logit = inDelphi_logit(
                examples["ref"][i],
                examples["mh_gt_pos"][i],
                examples["mh_del_len"][i],
                examples["mh_mh_len"][i],
                output["mh_weight"][i].cpu(),
                output["mhless_weight"].cpu(),
                output["pre_insert_probability"][i],
                output["pre_insert_1bp"][i],
            )

            likelihood.append(black_list_likelihood(observation, logit, black_list))

    return (
        likelihood,
        genotype_pearson,
        total_del_len_pearson,
        mhless_pearson,
        mhless_count_acc,
        total_del_len_count_acc,
        output["mhless_weight"].cpu(),
    )


@torch.no_grad()
def inDelphi_original_benchmark(
    data_name: str, black_list: List, celltype: str
):  # Supported cell types are ['mESC', 'U2OS', 'HEK293', 'HCT116', 'K562']
    from ..inDelphi.model import inDelphiConfig
    from ..inDelphi.load_data import data_collector, outputs_test

    ds_ob = load_ds_ob(data_name)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{inDelphiConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        DELLEN_LIMIT=args.DELLEN_LIMIT,
    )

    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(
            examples, args.DELLEN_LIMIT, outputs_test
        ),
    )

    inDelphi_original_proc = Popen(
        f"./benchmark_inDelphi_original.py {celltype}",
        shell=True,
        executable="/bin/bash",
        stdin=PIPE,
    )
    for i in tqdm(range(len(ds_ob)), desc=f"inDelphi original {celltype}, logit"):
        ref = (
            ds_ob[i]["ref1"][args.cut1 - args.DELLEN_LIMIT : args.cut1]
            + ds_ob[i]["ref2"][args.cut2 : args.cut2 + args.DELLEN_LIMIT]
        )
        inDelphi_original_proc.stdin.write((f"{ref}\t{60}\n").encode())
    inDelphi_original_proc.stdin.close()
    inDelphi_original_proc.wait()

    pred_df_all = pd.read_csv(
        "AI_models/inDelphi/reference/temp.csv",
        names=[
            "Category",
            "Genotype position",
            "Inserted Bases",
            "Length",
            "Predicted frequency",
        ],
        header=None,
    )
    bounds = [0] + (
        pred_df_all.index[pred_df_all["Inserted Bases"] == "G"] + 1
    ).values.tolist()

    likelihood, genotype_pearson, total_del_len_pearson, mhless_pearson = [], [], [], []
    bi = 0
    for examples, batch, ob_examples in tqdm(
        zip(
            ds.iter(batch_size=args.batch_size),
            test_dataloader,
            ds_ob.iter(batch_size=args.batch_size),
        ),
        desc=f"inDelphi original {celltype}, likelihood",
        total=len(ds_ob) / args.batch_size,
    ):
        for i in range(len(ob_examples["ref1"])):
            ref = (
                ob_examples["ref1"][i][: args.cut1]
                + ob_examples["ref2"][i][args.cut2 :]
            )
            pred_df = pred_df_all[bounds[bi] : bounds[bi + 1]]
            bi += 1
            pre_insert_1bp = np.empty(4)
            pre_insert_1bp[[0, 1, 3, 2]] = pred_df["Predicted frequency"][-4:] / 100
            pre_insert_probability = pre_insert_1bp.sum()
            pre_insert_1bp /= pre_insert_probability

            mh_gt_pos = (
                pred_df["Genotype position"][: -args.DELLEN_LIMIT + 1 - 4]
                .values.astype(int)
                .tolist()
            )
            mh_del_len = pred_df["Length"][: len(mh_gt_pos)].values.tolist()
            mh_mh_len = []
            for gp, dl in zip(mh_gt_pos, mh_del_len):
                gp += args.cut1
                for sf in range(gp - dl + 1):
                    if sf == gp - dl or ref[gp - dl - sf - 1] != ref[gp - sf - 1]:
                        break
                mh_mh_len.append(sf)

            mh_weight = torch.from_numpy(
                pred_df["Predicted frequency"][: len(mh_gt_pos)].values.copy()
            )
            mhless_weight = torch.from_numpy(
                pred_df["Predicted frequency"][
                    -args.DELLEN_LIMIT + 1 - 4 : -4
                ].values.copy()
            )

            idx = (
                pd.DataFrame(
                    {
                        "mh_del_len": examples["mh_del_len"][i],
                        "mh_gt_pos": examples["mh_gt_pos"][i],
                    }
                )
                .sort_values(by=["mh_del_len", "mh_gt_pos"])
                .index.values
            )

            genotype_pearson.append(
                (
                    F.normalize(torch.cat([mh_weight, mhless_weight]), p=2.0, dim=0)
                    * F.normalize(
                        torch.cat(
                            [
                                torch.tensor(examples["mh_count"][i])[idx],
                                torch.tensor(examples["mhless_count"][i]),
                            ]
                        ).to(torch.float64),
                        p=2.0,
                        dim=0,
                    )
                )
                .sum()
                .cpu()
                .item()
            )

            total_del_len_weight = mhless_weight.clone().scatter_add(
                dim=0,
                index=torch.from_numpy(pred_df["Length"][: len(mh_gt_pos)].values - 1),
                src=mh_weight,
            )

            total_del_len_pearson.append(
                (
                    F.normalize(total_del_len_weight, p=2.0, dim=0)
                    * F.normalize(batch["total_del_len_count"][i], p=2.0, dim=0)
                )
                .sum()
                .cpu()
                .item()
            )
            mhless_pearson.append(
                (
                    F.normalize(mhless_weight, p=2.0, dim=0)
                    * F.normalize(
                        torch.tensor(examples["mhless_count"][i], dtype=torch.float32),
                        p=2.0,
                        dim=0,
                    )
                )
                .sum()
                .cpu()
                .item()
            )

            observation = torch.zeros(args.ref2len + 1, args.ref1len + 1)
            observation[ob_examples["ob_ref2"][i], ob_examples["ob_ref1"][i]] = (
                torch.tensor(ob_examples["ob_val"][i], dtype=observation.dtype)
            )

            logit = inDelphi_logit(
                ref,
                mh_gt_pos,
                mh_del_len,
                mh_mh_len,
                mh_weight,
                mhless_weight,
                pre_insert_probability,
                pre_insert_1bp,
            )

            likelihood.append(black_list_likelihood(observation, logit, black_list))

    return (
        likelihood,
        genotype_pearson,
        total_del_len_pearson,
        mhless_pearson,
        mhless_weight,
    )


def FOREcasT_benchmark(data_name: str, black_list: List):
    from ..FOREcasT.model import FOREcasTModel, FOREcasTConfig
    from ..FOREcasT.pipeline import FOREcasTPipeline
    from ..FOREcasT.load_data import data_collector

    FOREcasT_model = FOREcasTModel.from_pretrained(
        args.output_dir
        / FOREcasTConfig.model_type
        / f"{data_name}_{FOREcasTConfig.model_type}"
    )
    pipe = FOREcasTPipeline(FOREcasT_model, args.FOREcasT_MAX_DEL_SIZE)
    pipe.FOREcasT_model.to(args.device)

    ds_ob = load_ds_ob(data_name)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{FOREcasTConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        FOREcasT_MAX_DEL_SIZE=args.FOREcasT_MAX_DEL_SIZE,
    )

    test_dataloader = DataLoader(
        dataset=ds, batch_size=args.batch_size, collate_fn=data_collector
    )

    likelihood = []
    for batch, ob_examples in tqdm(
        zip(test_dataloader, ds_ob.iter(batch_size=args.batch_size)),
        desc="FOREcasT",
        total=len(ds_ob) / args.batch_size,
    ):
        output = pipe(batch)
        for i in range(len(ob_examples["ref1"])):
            observation = torch.zeros(args.ref2len + 1, args.ref1len + 1)
            observation[ob_examples["ob_ref2"][i], ob_examples["ob_ref1"][i]] = (
                torch.tensor(ob_examples["ob_val"][i], dtype=observation.dtype)
            )

            ref1, ref2, cut1, cut2 = (
                ob_examples["ref1"][i],
                ob_examples["ref2"][i],
                ob_examples["cut1"][i],
                ob_examples["cut2"][i],
            )
            lefts, rights = [], []
            for left, right, ins_seq in zip(
                output["left"], output["right"], output["ins_seq"]
            ):
                templated_left, templated_right = detect_templated_insert(
                    left + cut1, right + cut2, ins_seq, ref1, ref2
                )
                lefts.append(templated_left)
                rights.append(templated_right)

            pre_probability = torch.sparse_coo_tensor(
                [rights, lefts],
                output["proba"][i].cpu(),
                (args.ref2len + 1, args.ref1len + 1),
            ).to_dense()

            logit = pre_probability.log().clamp_min(zero_logit)

            likelihood.append(black_list_likelihood(observation, logit, black_list))

    return likelihood


def Lindel_benchmark(data_name: str, black_list: List):
    from ..Lindel.model import LindelModel, LindelConfig
    from ..Lindel.pipeline import LindelPipeline
    from ..Lindel.load_data import data_collector, outputs_test

    Lindel_models = {
        f"{model}_model": LindelModel.from_pretrained(
            args.output_dir
            / LindelConfig.model_type
            / f"{data_name}_{LindelConfig.model_type}_{model}"
        )
        for model in ["indel", "ins", "del"]
    }

    pipe = LindelPipeline(**Lindel_models)
    pipe.indel_model.to(args.device)
    pipe.ins_model.to(args.device)
    pipe.del_model.to(args.device)

    ds_ob = load_ds_ob(data_name)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{LindelConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        Lindel_dlen=args.Lindel_dlen,
        Lindel_mh_len=args.Lindel_mh_len,
    )

    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(
            examples, args.Lindel_dlen, args.Lindel_mh_len, outputs_test
        ),
    )

    likelihood = []
    for batch, ob_examples in tqdm(
        zip(test_dataloader, ds_ob.iter(batch_size=args.batch_size)),
        desc="Lindel",
        total=len(ds_ob) / args.batch_size,
    ):
        output = pipe(batch)
        dlefts = [dstart + args.cut1 for dstart in output["dstart"]]
        drights = [dend + args.cut2 for dend in output["dend"]]
        for i in range(len(ob_examples["ref1"])):
            observation = torch.zeros(args.ref2len + 1, args.ref1len + 1)
            observation[ob_examples["ob_ref2"][i], ob_examples["ob_ref1"][i]] = (
                torch.tensor(ob_examples["ob_val"][i], dtype=observation.dtype)
            )

            ref1, ref2, cut1, cut2 = (
                ob_examples["ref1"][i],
                ob_examples["ref2"][i],
                ob_examples["cut1"][i],
                ob_examples["cut2"][i],
            )
            ilefts, irights = [], []
            for ins_seq in output["ins_base"][:-1]:
                templated_left, templated_right = detect_templated_insert(
                    cut1, cut2, ins_seq, ref1, ref2
                )
                ilefts.append(templated_left)
                irights.append(templated_right)

            all_proba = F.normalize(
                torch.cat(
                    [
                        output["del_pos_proba"][i] * output["del_proba"][i],
                        output["ins_base_proba"][i][:-1] * output["ins_proba"][i],
                    ]
                ),
                p=1,
                dim=0,
            )

            pre_probability = (
                torch.sparse_coo_tensor(
                    [drights + irights, dlefts + ilefts],
                    all_proba,
                    (args.ref2len + 1, args.ref1len + 1),
                )
                .to_dense()
                .cpu()
            )

            logit = pre_probability.log().clamp_min(zero_logit)

            likelihood.append(black_list_likelihood(observation, logit, black_list))

    return likelihood


def CRISPR_transformer_benchmark(data_name: str, black_list: List):
    from ..CRISPR_transformer.model import (
        CRISPRTransformerModel,
        CRISPRTransformerConfig,
    )
    from ..CRISPR_transformer.pipeline import CRISPRTransformerPipeline
    from ..CRISPR_transformer.load_data import data_collector, outputs_test

    CRISPR_transformer_model = CRISPRTransformerModel.from_pretrained(
        args.output_dir
        / CRISPRTransformerConfig.model_type
        / f"{data_name}_{CRISPRTransformerConfig.model_type}"
    )

    pipe = CRISPRTransformerPipeline(CRISPR_transformer_model)
    pipe.CRISPR_transformer_model.to(args.device)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{CRISPRTransformerConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(examples, outputs_test),
    )

    likelihood = []
    for batch in tqdm(test_dataloader, desc="CRISPR transformer"):
        output = pipe(batch)
        for i in range(batch["observation"].shape[0]):
            observation = batch["observation"][i]
            logit = output["logit"][i]

            likelihood.append(black_list_likelihood(observation, logit, black_list))

    return likelihood


def all_benchmark(data_name: str, black_list: List):
    benchmark_df_rows = list()
    accum = dict()
    accum["del_len"] = list(range(1, 60))

    def tensor_correlation(ten1: torch.Tensor, ten2: torch.Tensor):
        return (
            (
                F.normalize(ten1.to(torch.float32), p=2.0, dim=0)
                * F.normalize(ten2.to(torch.float32), p=2.0, dim=0)
            )
            .sum()
            .cpu()
            .item()
        )

    ###################
    # CRISPR transformer
    ###################
    likelihood = CRISPR_transformer_benchmark(data_name, black_list)
    benchmark_df_rows.extend(
        [
            {
                "value": likelihood,
                "stat": "likelihood",
                "model": "CRISPR_transformer",
                "pearson_type": ".",
            }
        ]
    )

    ####################
    # inDelphi
    ####################
    (
        likelihood,
        pearson_genotype,
        pearson_total,
        pearson_mhless,
        accum["mhless_count"],
        accum["total_count"],
        accum["inDelphi_mhless_weight"],
    ) = inDelphi_benchmark(data_name, black_list)
    benchmark_df_rows.extend(
        [
            {
                "value": likelihood,
                "stat": "likelihood",
                "model": "inDelphi",
                "pearson_type": ".",
            },
            {
                "value": pearson_genotype,
                "stat": "pearson",
                "model": "inDelphi",
                "pearson_type": "genotype",
            },
            {
                "value": pearson_total,
                "stat": "pearson",
                "model": "inDelphi",
                "pearson_type": "total",
            },
            {
                "value": pearson_mhless,
                "stat": "pearson",
                "model": "inDelphi",
                "pearson_type": "mhless",
            },
            {
                "value": [
                    tensor_correlation(
                        accum["inDelphi_mhless_weight"], accum["mhless_count"]
                    )
                ],
                "stat": "pearson",
                "model": "inDelphi",
                "pearson_type": "mhless_acc",
            },
            {
                "value": [
                    tensor_correlation(
                        accum["inDelphi_mhless_weight"], accum["total_count"]
                    )
                ],
                "stat": "pearson",
                "model": "inDelphi",
                "pearson_type": "total_acc",
            },
        ]
    )

    ####################
    # original
    ####################
    for celltype in ["mESC"]:  # ['mESC', 'U2OS', 'HEK293', 'HCT116', 'K562']
        (
            likelihood,
            pearson_genotype,
            pearson_total,
            pearson_mhless,
            accum[f"original_mhless_weight_{celltype}"],
        ) = inDelphi_original_benchmark(data_name, black_list, celltype)
        benchmark_df_rows.extend(
            [
                {
                    "value": likelihood,
                    "stat": "likelihood",
                    "model": f"original_{celltype}",
                    "pearson_type": ".",
                },
                {
                    "value": pearson_genotype,
                    "stat": "pearson",
                    "model": f"original_{celltype}",
                    "pearson_type": "genotype",
                },
                {
                    "value": pearson_total,
                    "stat": "pearson",
                    "model": f"original_{celltype}",
                    "pearson_type": "total",
                },
                {
                    "value": pearson_mhless,
                    "stat": "pearson",
                    "model": f"original_{celltype}",
                    "pearson_type": "mhless",
                },
                {
                    "value": [
                        tensor_correlation(
                            accum[f"original_mhless_weight_{celltype}"],
                            accum["mhless_count"],
                        )
                    ],
                    "stat": "pearson",
                    "model": f"original_{celltype}",
                    "pearson_type": "mhless_acc",
                },
                {
                    "value": [
                        tensor_correlation(
                            accum[f"original_mhless_weight_{celltype}"],
                            accum["total_count"],
                        )
                    ],
                    "stat": "pearson",
                    "model": f"original_{celltype}",
                    "pearson_type": "total_acc",
                },
            ]
        )

    ###################
    # FOREcasT
    ###################
    likelihood = FOREcasT_benchmark(data_name, black_list)
    benchmark_df_rows.extend(
        [
            {
                "value": likelihood,
                "stat": "likelihood",
                "model": "FOREcasT",
                "pearson_type": ".",
            }
        ]
    )

    ###################
    # Lindel
    ###################
    likelihood = Lindel_benchmark(data_name, black_list)
    benchmark_df_rows.extend(
        [
            {
                "value": likelihood,
                "stat": "likelihood",
                "model": "Lindel",
                "pearson_type": ".",
            }
        ]
    )

    ####################
    # save
    ####################
    pd.DataFrame(benchmark_df_rows).explode("value", ignore_index=True).to_csv(
        f"AI_models/benchmark/results/{data_name}_benchmark.csv", index=False
    )
    pd.DataFrame(accum).to_csv(
        f"AI_models/benchmark/results/{data_name}_accum.csv", index=False
    )


def virsualize_observation_and_prediction(
    data_name: str, file_name: str, test_idx=0, vmin=0, vmax=0.01
):
    from ..CRISPR_transformer.model import (
        CRISPRTransformerModel,
        CRISPRTransformerConfig,
    )
    from ..CRISPR_transformer.pipeline import CRISPRTransformerPipeline
    from ..CRISPR_transformer.load_data import data_collector, outputs_test

    CRISPR_transformer_model = CRISPRTransformerModel.from_pretrained(
        args.output_dir
        / CRISPRTransformerConfig.model_type
        / f"{data_name}_{CRISPRTransformerConfig.model_type}"
    )

    pipe = CRISPRTransformerPipeline(CRISPR_transformer_model)
    pipe.CRISPR_transformer_model.to(args.device)

    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{CRISPRTransformerConfig.model_type}",
        split=Split.TEST,
        trust_remote_code=True,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(examples, outputs_test),
    )

    likelihood = []
    for batch in tqdm(test_dataloader, desc="CRISPR transformer"):
        break
    output = pipe(batch)

    fig, ax = plt.subplots()
    im = ax.imshow(
        (batch["observation"][test_idx] / batch["observation"][test_idx].sum())
        .cpu()
        .numpy(),
        cmap=cm.get_cmap("Reds"),
        vmax=vmax,
        vmin=vmin,
    )
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{file_name}_observe.png")

    fig, ax = plt.subplots()
    im = ax.imshow(
        F.softmax(output["logit"][test_idx].flatten())
        .view(output["logit"][test_idx].shape)
        .cpu()
        .numpy(),
        cmap=cm.get_cmap("Reds"),
        vmax=vmax,
        vmin=vmin,
    )
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{file_name}_predict.png")
