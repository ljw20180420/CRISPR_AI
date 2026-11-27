import json
from numbers import Number

import numpy as np
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.lr_scheduler import MyLrScheduler
from common_ai.optimizer import MyOptimizer
from common_ai.utils import SeqTokenizer, split_train_valid_test
from datasets import Dataset
from tqdm import tqdm

from AI.metric import MSE, GreatestCommonCrossEntropy
from AI.preprocess.CRIfuser.model import CRIfuser
from AI.preprocess.FOREcasT.model import FOREcasT
from AI.preprocess.inDelphi.model import inDelphi
from AI.preprocess.Lindel.model import Lindel


def get_dataset(cfg: dict, data_name: str) -> Dataset:
    df_ref = pd.read_csv(
        cfg["output_dir"] / "rearr.ref",
        header=None,
        names=["ref1_start", "ref1", "cut1", "cut2", "ref2", "ref2_end"],
        sep="\t",
    )
    ref_len = cfg["extend"] + cfg["short_extend"]

    data_param = cfg["train"]["datasets"][data_name]
    df_alg = pd.DataFrame(
        pd.read_csv(
            cfg["output_dir"] / f"{data_name}.correct", header=None
        ).values.reshape(-1, 3),
        columns=["header", "refline", "queryline"],
    )
    df_expand = (
        df_alg["header"]
        .str.split("\t", expand=True)
        .rename(
            columns={
                0: "idx",
                1: "count",
                2: "score",
                3: "ref_id",
                4: "udangle",
                5: "rstart1",
                6: "qstart1",
                7: "rend1",
                8: "qend1",
                9: "random",
                10: "ref1+rstart2",
                11: "qstart2",
                12: "rend2",
                13: "qend2",
                14: "ddangle",
                15: "cut1",
                16: "ref1+cut2",
            }
        )
        .astype({"ref_id": int, "rend1": int, "ref1+rstart2": int, "count": int})
        .assign(
            rstart2=lambda df: df["ref1+rstart2"] - ref_len,
            random_len=lambda df: df["random"].map(
                lambda random: min(len(random), data_param["random_insert_uplimit"] + 1)
            ),
        )[["ref_id", "rend1", "rstart2", "random", "random_len", "count"]]
    )

    ref1s = []
    ref2s = []
    cut1s = []
    cut2s = []
    scaffolds = []
    ob_idxs = []
    ob_vals = []
    insert_countss = []
    insert_count_longs = []
    for ref_id, (ref1, ref2, cut1, cut2) in enumerate(
        zip(df_ref["ref1"], df_ref["ref2"], df_ref["cut1"], df_ref["cut2"])
    ):
        ref1 = df_ref.loc[ref_id, "ref1"]
        ref2 = df_ref.loc[ref_id, "ref2"]
        cut1 = df_ref.loc[ref_id, "cut1"]
        cut2 = df_ref.loc[ref_id, "cut2"]
        ref1s.append(ref1)
        ref2s.append(ref2)
        cut1s.append(cut1)
        cut2s.append(cut2)

        ob_sparse = (
            df_expand.query("ref_id == @ref_id")
            .groupby(
                [
                    "rend1",
                    "rstart2",
                    "random_len",
                ],
                as_index=False,
            )["count"]
            .sum()
        )
        observations = np.zeros(
            [data_param["random_insert_uplimit"] + 2, len(ref2) + 1, len(ref1) + 1],
            dtype=float,
        )
        observations[
            ob_sparse["random_len"], ob_sparse["rstart2"], ob_sparse["rend1"]
        ] = ob_sparse["count"]
        observations = observations.flatten()
        (ob_idx,) = observations.nonzero()
        ob_idxs.append(ob_idx)
        ob_vals.append(observations[ob_idx])

        df_insert = (
            df_expand.query("ref_id == @ref_id")
            .groupby(
                [
                    "rend1",
                    "rstart2",
                    "random",
                ],
                as_index=False,
            )["count"]
            .sum()
        )
        alphabet = "ACGT"
        base_cutoff = len(alphabet) ** np.arange(data_param["insert_uplimit"] + 1)
        base_cutoff[0] = 0
        base_cutoff = np.cumsum(base_cutoff)
        # ACGT to 0123
        DNA_tokenizer = SeqTokenizer(alphabet)
        insert_counts = np.zeros(base_cutoff[-1])
        insert_count_long = 0
        for rend1, rstart2, random_insert, count in zip(
            df_insert["rend1"],
            df_insert["rstart2"],
            df_insert["random"],
            df_insert["count"],
        ):
            if rend1 < cut1 or rstart2 > cut2:  # deletion or indel
                continue
            insert = ref1[cut1:rend1] + random_insert + ref2[rstart2:cut2]
            if not insert:  # wildtype
                continue
            if insert.find("N") != -1:  # ambitious insertion
                continue
            if (
                len(insert) > data_param["insert_uplimit"]
            ):  # insertion length beyond insert_uplimit
                insert_count_long += count
                continue
            vec = DNA_tokenizer(insert) * (
                len(alphabet) ** np.arange(len(insert) - 1, -1, -1)
            )
            idx = base_cutoff[len(insert) - 1] + vec.sum()
            insert_counts[idx] += count
        insert_countss.append(insert_counts)
        insert_count_longs.append(insert_count_long)

    ds = Dataset.from_dict({
        "ref1": ref1s,
        "ref2": ref2s,
        "cut1": cut1s,
        "cut2": cut2s,
        "scaffold": ["NNNNNN"]
        * len(df_ref),  # dummy value, not used for CRIfuser, inDelphi, FOREcasT, Lindel
        "random_insert_uplimit": [data_param["random_insert_uplimit"]] * len(df_ref),
        "ob_idx": ob_idxs,
        "ob_val": ob_vals,
        "insert_count": insert_countss,
        "insert_count_long": insert_count_longs,
    })

    ds = ds.train_test_split(
        test_size=data_param["validation_ratio"],
        seed=data_param["seed"],
    )
    ds["validation"] = ds.pop("test")

    return ds


def train_model(cfg: dict, model_name: str, data_name: str) -> None:
    checkpoint = None
    last_epoch = -1
    for epoch in range(cfg["train"]["num_epochs"] - 1, -1, -1):
        pt_path = (
            cfg["output_dir"]
            / "trains"
            / model_name
            / data_name
            / f"checkpoint-{epoch}"
            / "checkpoint.pt"
        )
        if pt_path.exists():
            checkpoint = torch.load(
                pt_path,
                weights_only=False,
            )
            last_epoch = epoch
            break

    my_generator = MyGenerator(**cfg["train"]["generator"])
    if checkpoint is not None:
        my_generator.load_state_dict(checkpoint["generator"])

    model = eval(model_name)(**cfg["train"]["models"][model_name])
    # Move model to the device before setup optimizer because some optimizers like Adagrad use device information.
    setattr(model, "device", cfg["train"]["device"])
    model.to(model.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        my_initializer = MyInitializer(**cfg["train"]["initializer"])
        if hasattr(model, "my_initialize_model"):
            model.my_initialize_model(my_initializer, my_generator)
        else:
            my_initializer(model, my_generator)

    if isinstance(model, torch.nn.Module):
        my_optimizer = MyOptimizer(**cfg["train"]["optimizer"])
        my_optimizer(model)
        my_lr_scheduler = MyLrScheduler(**cfg["train"]["lr_scheduler"])
        my_lr_scheduler(my_optimizer)
        if checkpoint is not None:
            my_optimizer.load_state_dict(checkpoint["optimizer"])
            my_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    metrics = {
        name: eval(name)(**param) for name, param in cfg["train"]["metrics"].items()
    }

    dataset = get_dataset(cfg, data_name)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["train"],
        batch_size=cfg["train"]["batch_size"],
        collate_fn=lambda examples: examples,
        shuffle=True,
        generator=my_generator.torch_c_rng,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["validation"],
        batch_size=cfg["train"]["batch_size"],
        collate_fn=lambda examples: examples,
    )

    for epoch in tqdm(range(last_epoch + 1, cfg["train"]["num_epochs"])):
        # train
        model.train()
        model.zero_grad()  # optimizer.zero_grad() is different when multiple models share a common optimizer
        train_loss, train_loss_num, grad_norm = 0.0, 0.0, 0.0
        for step, examples in tqdm(enumerate(train_dataloader)):
            batch = model.data_collator(
                examples, output_label=True, my_generator=my_generator
            )

            result = model(
                input=batch["input"],
                label=batch["label"],
                my_generator=my_generator,
            )

            result["loss"].backward()
            if (step + 1) % cfg["train"]["accumulate_steps"] == 0 or step == len(
                train_dataloader
            ):
                grad_norm += torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=cfg["train"]["clip_value"]
                ).item()
                my_optimizer.step()
                model.zero_grad()

            train_loss += (
                result["loss"].item()
                if not isinstance(result["loss"], Number)
                else result["loss"]
            )
            train_loss_num += (
                result["loss_num"].item()
                if not isinstance(result["loss_num"], Number)
                else result["loss_num"]
            )

        if model_name == "inDelphi":
            model.train_knn(train_dataloader, eval_dataloader, my_generator)

        # eval
        with torch.no_grad():
            if isinstance(model, torch.nn.Module):
                model.eval()
            eval_loss, eval_loss_num = 0.0, 0.0
            for examples in tqdm(eval_dataloader):
                batch = model.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                result = model(
                    input=batch["input"],
                    label=batch["label"],
                    my_generator=my_generator,
                )

                eval_loss += (
                    result["loss"].item()
                    if not isinstance(result["loss"], Number)
                    else result["loss"]
                )
                eval_loss_num += (
                    result["loss_num"].item()
                    if not isinstance(result["loss_num"], Number)
                    else result["loss_num"]
                )
                df = model.eval_output(examples, batch, my_generator)
                for metric_name, metric_fun in metrics.items():
                    metric_fun.step(
                        df=df,
                        examples=examples,
                        batch=batch,
                    )

            metric_loss_dict = {
                metric_name: metric_fun.epoch()
                for metric_name, metric_fun in metrics.items()
            }

        # log
        loginfo = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_num": train_loss_num,
            "grad_norm": grad_norm,
            "eval_loss": eval_loss,
            "eval_loss_num": eval_loss_num,
            **metric_loss_dict,
        }

        if isinstance(model, torch.nn.Module):
            loginfo["learning_rate"] = my_lr_scheduler.get_last_lr()[0]
            my_lr_scheduler.step(eval_loss / eval_loss_num)

        (cfg["output_dir"] / "trains" / model_name / data_name).mkdir(
            exist_ok=True, parents=True
        )
        with open(
            cfg["output_dir"] / "trains" / model_name / data_name / "info.log", "a"
        ) as fd:
            json.dump(loginfo, fd)
            fd.write("\n")
            fd.flush()

        # save
        (
            cfg["output_dir"]
            / "trains"
            / model_name
            / data_name
            / f"checkpoint-{epoch}"
        ).mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "model": model.state_dict(),
            "generator": my_generator.state_dict(),
        }
        if isinstance(model, torch.nn.Module):
            checkpoint.update({
                "optimizer": my_optimizer.state_dict(),
                "lr_scheduler": my_lr_scheduler.state_dict(),
            })

        torch.save(
            obj=checkpoint,
            f=cfg["output_dir"]
            / "trains"
            / model_name
            / data_name
            / f"checkpoint-{epoch}"
            / "checkpoint.pt",
        )
