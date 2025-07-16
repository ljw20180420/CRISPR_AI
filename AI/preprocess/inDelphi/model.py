from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from huggingface_hub import HfFileSystem
import os
import pathlib
import logging
import datasets
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle
from tqdm import tqdm


class inDelphiConfig(PretrainedConfig):
    model_type = "inDelphi"
    label_names = ["genotype_count", "total_del_len_count"]

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
        self.mid_active = self.sigmoid
        self.out_active = self.logit_to_weight
        self.initialize_weights()

    def initialize_weights(self):
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

    def forward(
        self, mh_input, mh_del_len, genotype_count=None, total_del_len_count=None
    ):
        batch_size = mh_input.shape[0]
        mh_weight = self.mh_in_layer(mh_input)
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_mid_layer(mh_weight)
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_out_layer(mh_weight)
        mh_weight = self.out_active(mh_weight, mh_del_len)

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
            ).scatter_add(dim=1, index=mh_del_len - 1, src=mh_weight)[:, :-1]
            + mhless_weight
        )
        if genotype_count is not None and total_del_len_count is not None:
            loss = self.negative_correlation(
                mh_weight,
                mhless_weight,
                total_del_len_weight,
                genotype_count,
                total_del_len_count,
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

    def logit_to_weight(self, logits, del_lens):
        return torch.exp(logits.squeeze() - 0.25 * del_lens) * (
            del_lens < self.DELLEN_LIMIT
        )

    def sigmoid(self, x):
        return 0.5 * (F.tanh(x) + 1)

    def negative_correlation(
        self,
        mh_weight,
        mhless_weight,
        total_del_len_weight,
        genotype_count,
        total_del_len_count,
    ):
        batch_size = mh_weight.shape[0]
        genotype_pearson = (
            F.normalize(
                torch.cat((mh_weight, mhless_weight.expand(batch_size, -1)), dim=1),
                p=2.0,
                dim=1,
            )
            * F.normalize(genotype_count, p=2.0, dim=1)
        ).sum()

        total_del_len_pearson = (
            F.normalize(total_del_len_weight, p=2.0, dim=1)
            * F.normalize(total_del_len_count, p=2.0, dim=1)
        ).sum()

        return -genotype_pearson - total_del_len_pearson


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

        self.compose_auxilary(knn_features, insert_probabilities, m654s)

    def compose_auxilary(
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
        knn_feature = self.get_knn_feature(total_del_len_weights, onebp_features)
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
                data_collator.output_label = False
                batch = data_collator(examples)
                result = model(
                    batch["mh_input"].to(model.device),
                    batch["mh_del_len"].to(model.device),
                )
                data_collator.output_label = True
                insert_batch = data_collator.insert_call(examples)
                knn_features.append(
                    self.get_knn_feature(
                        result["total_del_len_weight"],
                        insert_batch["onebp_feature"],
                    )
                )
                np.add.at(
                    m654s,
                    insert_batch["m654"],
                    insert_batch["insert_1bp"],
                )
                insert_probabilities.append(insert_batch["insert_probability"])
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

    def get_knn_feature(
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
