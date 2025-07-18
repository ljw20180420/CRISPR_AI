from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
import pickle
from sklearn.neighbors import KNeighborsRegressor
from huggingface_hub import HfFileSystem
import os
import pathlib
import logging
import datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, repeat


class inDelphiConfig(PretrainedConfig):
    model_type = "inDelphi"
    label_names = ["label"]

    def __init__(
        self,
        DELLEN_LIMIT: Optional[int] = None,
        mid_dim: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """inDelphi paramters.

        Args:
            DELLEN_LIMIT: the upper limit of deletion length (strictly less than DELLEN_LIMIT).
            mid_dim: the size of middle layer of MLP.
        """
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.mid_dim = mid_dim
        self.seed = seed
        super().__init__(**kwargs)


class inDelphiModel(PreTrainedModel):
    config_class = inDelphiConfig

    def __init__(self, config: inDelphiConfig) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
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
        self._initialize_model_layer_weights()

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (F.tanh(x) + 1)

    def _logit_to_weight(
        self, logits: torch.Tensor, del_lens: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(logits.squeeze() - 0.25 * del_lens) * (
            del_lens < self.DELLEN_LIMIT
        )

    # huggingface use the name initialize_weights, use another name here.
    def _initialize_model_layer_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            loss = self.loss_fun(
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

        return -genotype_pearson - total_del_len_pearson

    def eval_output(
        self,
        examples: list[dict],
        batch: dict,
        auxilary_model: PreTrainedModel,
    ) -> pd.DataFrame:
        result = self(batch["input"])
        insert_probabilities, insert_1bps = auxilary_model(
            result["total_del_len_weight"],
            batch["onebp_feature"],
            batch["m654"],
            use_m654=False,
        )
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


class inDelphiAuxilary(PreTrainedModel):
    config_class = inDelphiConfig

    def __init__(self, config: inDelphiConfig) -> None:
        super().__init__(config)
        self.workaround_to_save_the_model = nn.Parameter(torch.zeros(1))

    def load_auxilary(self, model_pickle_file: str | pathlib.Path) -> None:
        if os.path.exists(model_pickle_file):
            with open(model_pickle_file, "rb") as fd:
                knn_features, insert_probabilities, m654s = pickle.load(fd)
        else:
            fs = HfFileSystem()
            with fs.open(model_pickle_file, "rb") as fd:
                knn_features, insert_probabilities, m654s = pickle.load(fd)

        self._compose_auxilary(knn_features, insert_probabilities, m654s)

    def _compose_auxilary(
        self,
        knn_features: np.ndarray,
        insert_probabilities: np.ndarray,
        m654s: np.ndarray,
    ) -> None:
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

    def __call__(
        self,
        total_del_len_weights: torch.Tensor,
        onebp_features: np.ndarray,
        m654s: np.ndarray,
        use_m654: bool,
    ) -> tuple[np.ndarray]:
        knn_feature = self._get_knn_feature(total_del_len_weights, onebp_features)
        insert_probabilities = self.knn.predict(
            (knn_feature - self.knn_feature_mean) / self.knn_feature_std
        )

        if use_m654:
            insert_1bps = self.m654s[m654s]
        else:
            insert_1bps = self.m4s[m654s % 4]

        return insert_probabilities, insert_1bps

    def train_auxilary(
        self,
        preprocess: str,
        model_name: str,
        data_collator,  # Type hint make model.py depends on load_data.py, thereby utils.py.
        data_name: str,
        ds: datasets.Dataset,
        output_dir: pathlib.Path,
        batch_size: int,
        device: str,
        logger: logging.Logger,
    ) -> None:
        logger.info("loading model")
        model = inDelphiModel.from_pretrained(
            output_dir / preprocess / model_name / data_name / "core_model"
        ).to(device)

        logger.info("loading data")
        dl = DataLoader(
            dataset=ds["train"],
            batch_size=batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("train inDelphi insertion model")
        with torch.no_grad():
            knn_features = []
            insert_probabilities = []
            m654s = np.zeros((4**3, 4), dtype=int)
            for examples in tqdm(dl):
                data_collator.output_label = True
                batch = data_collator(examples)
                result = model(
                    batch["mh_input"].to(model.device),
                    batch["mh_del_len"].to(model.device),
                )
                knn_features.append(
                    self._get_knn_feature(
                        result["total_del_len_weight"],
                        batch["onebp_feature"],
                    )
                )
                np.add.at(
                    m654s,
                    batch["m654"],
                    batch["insert_1bp"],
                )
                insert_probabilities.append(batch["insert_probability"])
            knn_features = np.concatenate(knn_features, axis=0)
            insert_probabilities = np.concatenate(insert_probabilities, axis=0)

        logger.info("save")
        self.save_pretrained(
            output_dir / preprocess / model_name / data_name / "auxilary_model"
        )
        with open(
            output_dir
            / preprocess
            / model_name
            / data_name
            / "auxilary_model"
            / "auxilary.pkl",
            "wb",
        ) as fd:
            pickle.dump([knn_features, insert_probabilities, m654s], fd)

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
