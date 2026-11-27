import pandas as pd
import torch
from common_ai.config import get_train_parser
from jsonargparse import Namespace

from AI.inference import MyInference
from AI.metric import MSE, GreatestCommonCrossEntropy

from . import config


def predict_by_AI_models(cfg: dict):
    black_list_sites = cfg["black_list_sites"]
    infer_df = (
        pd.read_csv(cfg["output_dir"] / "target_sites.csv", header=0)
        .query("not site.isin(@black_list_sites)")
        .reset_index(drop=True)[["ref"]]
        .assign(cut=cfg["extend"], scaffold="spcas9")
    )

    data_name = cfg["predict"]["data_name"]
    for model in cfg["predict"]["models"]:
        checkpoints_path = config.get_checkpoints_path(model, data_name)
        logs_path = config.get_logs_path(model, data_name)
        infer_out = MyInference(**cfg["inference"])(
            infer_df,
            test_cfg=Namespace(
                checkpoints_path=checkpoints_path,
                logs_path=logs_path,
                target=cfg["predict"]["target"],
                maximize_target=cfg["predict"]["maximize_target"],
                overwrite={},
            ),
            train_parser=get_train_parser(),
        )

        (cfg["output_dir"] / "predicts").mkdir(exist_ok=True, parents=True)
        infer_out.to_csv(
            cfg["output_dir"] / "predicts" / f"{model}_{data_name}.csv", index=False
        )


def get_metrics(cfg: dict):
    df_ref = pd.read_csv(
        cfg["output_dir"] / "rearr.ref",
        header=None,
        names=["ref1_start", "ref1", "cut1", "cut2", "ref2", "ref2_end"],
        sep="\t",
    )
    examples = [
        {
            "cut1": cut1,
            "cut2": cut2,
        }
        for cut1, cut2 in zip(df_ref["cut1"], df_ref["cut2"])
    ]
    ref_len = cfg["extend"] + cfg["short_extend"]
    nll = GreatestCommonCrossEntropy(**cfg["inference"])
    mse = MSE(**cfg["inference"])

    cass = []
    models = []
    metrics = []
    values = []
    for cas in cfg["cas_proteins"]:
        df_alg = pd.DataFrame(
            pd.read_csv(
                cfg["output_dir"] / f"{cas}.correct", header=None
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
            .assign(rstart2=lambda df: df["ref1+rstart2"] - ref_len)[
                ["ref_id", "rend1", "rstart2", "count"]
            ]
            .assign(
                refline=df_alg["refline"],
                queryline=df_alg["queryline"],
            )
        )

        observations = []
        for ref_id in range(len(df_ref)):
            ob = torch.from_numpy(
                df_expand.query("ref_id == @ref_id")
                .pivot_table(
                    values="count", index="rstart2", columns="rend1", fill_value=0
                )
                .reindex(
                    index=range(ref_len + 1),
                    columns=range(ref_len + 1),
                    fill_value=0,
                )
                .to_numpy()
                .copy()
            )
            observations.append(ob)

        observations = torch.stack(observations)
        batch = {"label": {"observation": observations}}

        data_name = cfg["predict"]["data_name"]
        for model in cfg["predict"]["models"]:
            df_predict = pd.read_csv(
                cfg["output_dir"] / "predicts" / f"{model}_{data_name}.csv", header=0
            )

            print(cas, model, "nll")
            nll.step(df_predict, examples, batch)
            cass.append(cas)
            models.append(model)
            metrics.append("nll")
            values.append(nll.epoch())

            print(cas, model, "mse")
            mse.step(df_predict, examples, batch)
            cass.append(cas)
            models.append(model)
            metrics.append("mse")
            values.append(mse.epoch())

    pd.DataFrame({
        "cas": cass,
        "model": models,
        "metric": metrics,
        "value": values,
    }).to_csv(cfg["output_dir"] / "bench.csv", index=False)
