import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
import pickle
from sklearn.neighbors import KNeighborsRegressor
import os
import datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, repeat
from .data_collator import DataCollator


class inDelphiConfig(PretrainedConfig):
    model_type = "inDelphi"

    def __init__(
        self,
        DELLEN_LIMIT: int,
        mid_dim: int,
        **kwargs,
    ) -> None:
        """inDelphi paramters.

        Args:
            DELLEN_LIMIT: the upper limit of deletion length (strictly less than DELLEN_LIMIT).
            mid_dim: the size of middle layer of MLP.
        """
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.mid_dim = mid_dim
        super().__init__(**kwargs)


class inDelphiModel(PreTrainedModel):
    config_class = inDelphiConfig

    def __init__(self, config: inDelphiConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(DELLEN_LIMIT=config.DELLEN_LIMIT)
        self.DELLEN_LIMIT = config.DELLEN_LIMIT
        self.register_buffer(
            "del_lens", torch.arange(1, config.DELLEN_LIMIT, dtype=torch.float32)
        )
        self.mh_in_layer = nn.Linear(in_features=2, out_features=config.mid_dim)
        self.mh_mid_layer = nn.Linear(
            in_features=config.mid_dim, out_features=config.mid_dim
        )
        self.mh_out_layer = nn.Linear(in_features=config.mid_dim, out_features=1)
        self.mhless_in_layer = nn.Linear(in_features=1, out_features=config.mid_dim)
        self.mhless_mid_layer = nn.Linear(
            in_features=config.mid_dim, out_features=config.mid_dim
        )
        self.mhless_out_layer = nn.Linear(in_features=config.mid_dim, out_features=1)
        self.mid_active = self._sigmoid
        self.out_active = self._logit_to_weight

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (F.tanh(x) + 1)

    def _logit_to_weight(
        self, logits: torch.Tensor, del_lens: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(logits.squeeze() - 0.25 * del_lens) * (
            del_lens < self.DELLEN_LIMIT
        )

    def forward(self, input: dict, label: Optional[dict] = None) -> dict:
        batch_size = input["mh_input"].shape[0]
        mh_weight = self.mh_in_layer(input["mh_input"])
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_mid_layer(mh_weight)
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_out_layer(mh_weight)
        mh_weight = self.out_active(mh_weight, input["mh_del_len"])

        mhless_weight = self.mhless_in_layer(self.del_lens[:, None])
        mhless_weight = self.mid_active(mhless_weight)
        mhless_weight = self.mhless_mid_layer(mhless_weight)
        mhless_weight = self.mid_active(mhless_weight)
        mhless_weight = self.mhless_out_layer(mhless_weight)
        mhless_weight = self.out_active(mhless_weight, self.del_lens)

        total_del_len_weight = (
            torch.zeros(
                batch_size,
                mhless_weight.shape[0] + 1,
                dtype=mh_weight.dtype,
                device=mh_weight.device,
            ).scatter_add(dim=1, index=input["mh_del_len"] - 1, src=mh_weight)[:, :-1]
            + mhless_weight
        )
        if label is not None:
            loss, loss_num = self.loss_fun(
                mh_weight,
                mhless_weight,
                total_del_len_weight,
                label["genotype_count"],
                label["total_del_len_count"],
            )
            return {
                "mh_weight": mh_weight,
                "mhless_weight": mhless_weight,
                "total_del_len_weight": total_del_len_weight,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {
            "mh_weight": mh_weight,
            "mhless_weight": mhless_weight,
            "total_del_len_weight": total_del_len_weight,
        }

    def loss_fun(
        self,
        mh_weight: torch.Tensor,
        mhless_weight: torch.Tensor,
        total_del_len_weight: torch.Tensor,
        genotype_count: torch.Tensor,
        total_del_len_count: torch.Tensor,
    ) -> float:
        # negative correlation
        batch_size = mh_weight.shape[0]
        genotype_pearson = einsum(
            F.normalize(
                torch.cat(
                    (
                        mh_weight,
                        repeat(mhless_weight, "w -> b w", b=batch_size),
                    ),
                    dim=1,
                ),
                p=2.0,
                dim=1,
            ),
            F.normalize(genotype_count, p=2.0, dim=1),
            "b w, b w ->",
        )

        total_del_len_pearson = einsum(
            F.normalize(total_del_len_weight, p=2.0, dim=1),
            F.normalize(total_del_len_count, p=2.0, dim=1),
            "b w, b w ->",
        )

        loss = -genotype_pearson - total_del_len_pearson
        loss_num = batch_size
        return loss, loss_num

    def eval_output(
        self,
        examples: list[dict],
        batch: dict,
    ) -> pd.DataFrame:
        result = self(batch["input"])
        knn_feature = self._get_knn_feature(
            result["total_del_len_weight"], batch["input"]["onebp_features"]
        )
        insert_probabilities = self.knn.predict(
            (knn_feature - self.knn_feature_mean) / self.knn_feature_std
        )
        insert_1bps = self.m4s[
            batch["input"]["m654"] % 4
        ]  # insert_1bps = self.m654s[batch["input"]["m654"]]

        batch_size = result["mh_weight"].shape[0]
        delete_probabilities = einsum(
            F.normalize(
                torch.cat(
                    (
                        result["mh_weight"],
                        repeat(result["mhless_weight"], "w -> b w", b=batch_size),
                    ),
                    dim=1,
                ),
                p=1.0,
                dim=1,
            )
            .cpu()
            .numpy(),
            1 - insert_probabilities,
            "b w, b -> b w",
        )
        DELLEN_LIMIT = self.config.DELLEN_LIMIT
        sample_idxs, probas, rpos1s, rpos2s, random_inss = [], [], [], [], []
        for sample_idx, (
            delete_probability,
            insert_probability,
            insert_1bp,
            rightest,
            mh_mh_len,
            mh_del_len,
            example,
        ) in enumerate(
            zip(
                delete_probabilities,
                insert_probabilities,
                insert_1bps,
                batch["rightest"],
                batch["mh_input"][:, :, 0].cpu().numpy(),
                batch["mh_del_len"].cpu().numpy(),
                examples,
            )
        ):
            mh_mh_len = mh_mh_len[mh_mh_len > 0]
            mh_del_len = mh_del_len[mh_del_len < self.config.DELLEN_LIMIT]
            probas.append(
                (delete_probability[: len(mh_mh_len)] / (mh_mh_len + 1)).repeat(
                    mh_mh_len + 1
                )
            )
            rpos2_mh = np.concatenate(
                [np.arange(rt, rt - ml - 1, -1) for rt, ml in zip(rightest, mh_mh_len)]
            )
            rpos1_mh = rpos2_mh - mh_del_len.repeat(mh_mh_len + 1)
            rpos1s.append(rpos1_mh)
            rpos2s.append(rpos2_mh)
            probas.append(
                (
                    delete_probability[-(DELLEN_LIMIT - 1) :]
                    / np.arange(2, DELLEN_LIMIT + 1)
                ).repeat(np.arange(2, DELLEN_LIMIT + 1))
            )
            rpos1s.append(
                np.concatenate(
                    [np.arange(-DEL_SIZE, 1) for DEL_SIZE in range(1, DELLEN_LIMIT)]
                )
            )
            rpos2s.append(
                np.concatenate(
                    [np.arange(0, DEL_SIZE + 1) for DEL_SIZE in range(1, DELLEN_LIMIT)]
                )
            )
            random_inss.append(
                np.array([""] * (len(probas[-1]) + len(probas[-2])), dtype="1U")
            )
            probas.append(insert_1bp / insert_1bp.sum() * insert_probability)
            tr2 = example["ref2"][example["cut2"] - 1]
            rpos2_ins = np.array(
                [-1 if base == tr2 else 0 for base in ["A", "C", "G", "T"]]
            )
            tr1 = example["ref1"][example["cut1"]]
            rpos1_ins = np.array(
                [1 if base == tr1 else 0 for base in ["A", "C", "G", "T"]]
            )
            rpos1_ins = np.minimum(rpos1_ins, 1 + rpos2_ins)
            rpos1s.append(rpos1_ins)
            rpos2s.append(rpos2_ins)
            random_inss.append(
                np.array(
                    [
                        "" if rpos1_ins[i] != 0 or rpos2_ins[i] != 0 else base
                        for i, base in enumerate(["A", "C", "G", "T"])
                    ],
                    dtype="1U",
                )
            )
            sample_idxs.extend(
                (len(probas[-1]) + len(probas[-2]) + len(probas[-3])) * [sample_idx]
            )

        df = (
            pd.DataFrame(
                {
                    "sample_idx": sample_idxs,
                    "proba": np.concatenate(probas, axis=0),
                    "rpos1": np.concatenate(rpos1s, axis=0),
                    "rpos2": np.concatenate(rpos2s, axis=0),
                    "random_ins": np.concatenate(random_inss, axis=0),
                }
            )
            .groupby(by=["sample_idx", "rpos1", "rpos2", "random_ins"])["proba"]
            .sum()
            .reset_index()
        )

        return df

    def state_dict(self) -> dict:
        return {
            "pytorch_state_dict": super().state_dict(),
            "scikit_learn_state_dict": {
                component: getattr(self, component)
                for component in [
                    "knn",
                    "knn_feature_mean",
                    "knn_feature_std",
                    "m654s",
                    "m4s",
                ]
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict["pytorch_state_dict"])
        for component in [
            "knn",
            "knn_feature_mean",
            "knn_feature_std",
            "m654s",
            "m4s",
        ]:
            setattr(self, component, state_dict["scikit_learn_state_dict"][component])

    @torch.no_grad()
    def train_scikit_learn(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        self.eval()
        knn_features = []
        insert_probabilities = []
        m654s = np.zeros((4**3, 4), dtype=int)
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(examples, output_label=True)
            result = self(batch["input"])
            knn_features.append(
                self._get_knn_feature(
                    result["total_del_len_weight"],
                    batch["input"]["onebp_feature"],
                )
            )
            np.add.at(
                m654s,
                batch["input"]["m654"],
                batch["label"]["insert_1bp"],
            )
            insert_probabilities.append(batch["label"]["insert_probability"])
        knn_features = np.concatenate(knn_features, axis=0)
        insert_probabilities = np.concatenate(insert_probabilities, axis=0)

        self.knn_feature_mean = knn_features.mean(axis=0)
        self.knn_feature_std = knn_features.std(axis=0)
        self.knn = KNeighborsRegressor(weights="distance").fit(
            (knn_features - self.knn_feature_mean) / self.knn_feature_std,
            insert_probabilities,
        )
        self.m654s = m654s / np.maximum(
            np.linalg.norm(m654s, ord=1, axis=1, keepdims=True),
            1e-6,
        )
        self.m4s = m654s.reshape(16, 4, 4).sum(axis=0)
        self.m4s = self.m4s / np.maximum(
            np.linalg.norm(self.m4s, ord=1, axis=1, keepdims=True),
            1e-6,
        )

    def _get_knn_feature(
        self, total_del_len_weights: torch.Tensor, onebp_features: np.ndarray
    ) -> np.ndarray:
        log_total_weights = total_del_len_weights.sum(dim=1, keepdim=True).log()
        precisions = 1 - torch.distributions.Categorical(
            total_del_len_weights[:, :28]
        ).entropy() / torch.log(torch.tensor(28))

        return np.concatenate(
            [
                onebp_features,
                precisions[:, None].cpu().numpy(),
                log_total_weights.cpu().numpy(),
            ],
            axis=1,
        )
